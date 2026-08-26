"""Validation and loading for candidate-editable training-only artifacts."""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class TrainingInputs:
    data_dir: Path
    prompt_file: Path
    reward_file: Path
    hashes: dict[str, str]


def resolve_scheduler_horizon(run_steps: int, requested_horizon: int | None) -> int:
    if run_steps <= 0:
        raise ValueError("run_steps must be positive")
    horizon = run_steps if requested_horizon is None else requested_horizon
    if horizon < run_steps:
        raise ValueError(f"scheduler horizon {horizon} is smaller than run_steps {run_steps}")
    return horizon


def _require_below(root: Path, path: Path, *, kind: str) -> Path:
    root_resolved = root.resolve(strict=True)
    if path.is_symlink():
        raise ValueError(f"{kind} may not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(f"{kind} must stay below training artifact root {root_resolved}: {resolved}") from error
    cursor = root_resolved
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"{kind} contains a symlink: {cursor}")
    return resolved


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_hash(path: Path) -> str:
    if path.is_file():
        return _file_hash(path)
    if not path.is_dir():
        raise ValueError(f"training input is neither a file nor directory: {path}")
    digest = hashlib.sha256()
    for directory, dirnames, filenames in os.walk(path, followlinks=False):
        directory_path = Path(directory)
        for name in dirnames:
            if (directory_path / name).is_symlink():
                raise ValueError(f"training input contains a symlink: {directory_path / name}")
        for name in sorted(filenames):
            file_path = directory_path / name
            if file_path.is_symlink():
                raise ValueError(f"training input contains a symlink: {file_path}")
            digest.update(file_path.relative_to(path).as_posix().encode())
            digest.update(b"\0")
            digest.update(bytes.fromhex(_file_hash(file_path)))
    return digest.hexdigest()


def load_training_inputs(
    artifact_root: Path,
    data_dir: Path,
    prompt_file: Path,
    reward_file: Path,
) -> TrainingInputs:
    root = artifact_root.resolve(strict=True)
    data = _require_below(root, data_dir, kind="training data")
    prompt = _require_below(root, prompt_file, kind="training prompt")
    reward = _require_below(root, reward_file, kind="training reward")
    if not data.is_dir():
        raise ValueError("training data path must be a directory")
    if not prompt.is_file() or not prompt.read_text().strip():
        raise ValueError("training prompt must be a nonempty regular file")
    if not reward.is_file() or not reward.read_text().strip():
        raise ValueError("training reward must be a nonempty regular file")
    return TrainingInputs(
        data_dir=data,
        prompt_file=prompt,
        reward_file=reward,
        hashes={
            "training_data_sha256": _path_hash(data),
            "training_prompt_sha256": _path_hash(prompt),
            "training_reward_sha256": _path_hash(reward),
        },
    )


def _load_reward_hook(reward_file: Path, artifact_root: Path) -> Callable[[dict[str, Any]], list[float]]:
    path = _require_below(artifact_root, reward_file, kind="training reward")
    module_name = f"tmax_training_reward_{_file_hash(path)}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load training reward module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hook = getattr(module, "shape_rewards", None)
    if not callable(hook):
        raise ValueError("training reward module must define callable shape_rewards(context)")
    return hook


def apply_training_reward_shaping(
    *,
    reward_file: Path,
    artifact_root: Path,
    base_rewards: list[float],
    decoded_responses: list[str],
    finish_reasons: list[str],
    rollout_states: list[dict[str, Any]],
) -> list[float]:
    lengths = {len(base_rewards), len(decoded_responses), len(finish_reasons), len(rollout_states)}
    if len(lengths) != 1:
        raise ValueError("training reward inputs must have equal lengths")
    hook = _load_reward_hook(reward_file, artifact_root)
    result = hook(
        {
            "base_rewards": list(base_rewards),
            "decoded_responses": list(decoded_responses),
            "finish_reasons": list(finish_reasons),
            "rollout_states": [dict(state) for state in rollout_states],
        }
    )
    if not isinstance(result, list) or len(result) != len(base_rewards):
        raise ValueError("shape_rewards must return one reward per response")
    try:
        shaped = [float(value) for value in result]
    except (TypeError, ValueError) as error:
        raise ValueError("shape_rewards must return numeric rewards") from error
    if not all(math.isfinite(value) for value in shaped):
        raise ValueError("shape_rewards must return finite rewards")
    return shaped
