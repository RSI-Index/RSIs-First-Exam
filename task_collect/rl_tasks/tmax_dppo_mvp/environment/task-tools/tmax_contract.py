#!/usr/bin/env python3
"""Trusted input contracts shared by the TMAX task tools."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


FINAL_OPTIMIZER_UPDATES = 200
TRAJECTORIES_PER_UPDATE = 256
FINAL_TRAINING_TRAJECTORIES = FINAL_OPTIMIZER_UPDATES * TRAJECTORIES_PER_UPDATE
SCIENTIFIC_HASH_FIELDS = (
    "source_sha256",
    "training_data_sha256",
    "training_prompt_sha256",
    "training_reward_sha256",
    "rl_recipe_sha256",
)
IGNORED_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "wandb",
}
_ATTEMPT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


@dataclass(frozen=True)
class InputManifest:
    attempt_id: str
    max_updates: int
    scheduler_horizon_updates: int
    source_sha256: str
    training_data_sha256: str
    training_prompt_sha256: str
    training_reward_sha256: str
    rl_recipe_sha256: str
    source_path: str
    training_data_path: str
    training_prompt_path: str
    training_reward_path: str
    rl_recipe_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_max_updates(value: object) -> int:
    """Return an attempt length in [1, 200], rejecting coercive inputs."""

    if isinstance(value, bool) or value is None:
        raise ValueError("max_updates must be an integer from 1 through 200")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.isascii() and value.isdigit():
        parsed = int(value)
    else:
        raise ValueError("max_updates must be an integer from 1 through 200")
    if not 1 <= parsed <= FINAL_OPTIMIZER_UPDATES:
        raise ValueError("max_updates must be an integer from 1 through 200")
    return parsed


def validate_attempt_id(value: object) -> str:
    if not isinstance(value, str) or not _ATTEMPT_RE.fullmatch(value):
        raise ValueError("attempt_id must contain 1-96 safe identifier characters")
    return value


def sha256_file(path: Path) -> str:
    if path.is_symlink():
        raise ValueError(f"symlink is forbidden in scientific inputs: {path}")
    if not path.is_file():
        raise ValueError(f"required scientific input file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scientific_files(root: Path) -> Iterable[Path]:
    if root.is_symlink():
        raise ValueError(f"symlink is forbidden in scientific inputs: {root}")
    if root.is_file():
        yield root
        return
    if not root.is_dir():
        raise ValueError(f"required scientific input path is missing: {root}")
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        retained_dirs: list[str] = []
        for name in sorted(dirnames):
            child = directory_path / name
            if name in IGNORED_NAMES:
                continue
            if child.is_symlink():
                raise ValueError(f"symlink is forbidden in scientific inputs: {child}")
            retained_dirs.append(name)
        dirnames[:] = retained_dirs
        for name in sorted(filenames):
            if name in IGNORED_NAMES or name.endswith((".pyc", ".pyo")):
                continue
            child = directory_path / name
            if child.is_symlink():
                raise ValueError(f"symlink is forbidden in scientific inputs: {child}")
            yield child


def tree_digest(path: Path | str) -> str:
    root = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    if root.is_file():
        digest.update(b"file\0")
        digest.update(root.name.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(root)))
        return digest.hexdigest()
    for file_path in _scientific_files(root):
        relative = file_path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(file_path)))
        digest.update(b"\0")
    return digest.hexdigest()


def build_input_manifest(
    *,
    attempt_id: str,
    max_updates: int | str,
    source: Path,
    training_data: Path,
    training_prompt: Path,
    training_reward: Path,
    rl_recipe: Path,
) -> InputManifest:
    attempt = validate_attempt_id(attempt_id)
    updates = validate_max_updates(max_updates)
    return InputManifest(
        attempt_id=attempt,
        max_updates=updates,
        scheduler_horizon_updates=FINAL_OPTIMIZER_UPDATES,
        source_sha256=tree_digest(source),
        training_data_sha256=tree_digest(training_data),
        training_prompt_sha256=tree_digest(training_prompt),
        training_reward_sha256=tree_digest(training_reward),
        rl_recipe_sha256=tree_digest(rl_recipe),
        source_path=str(Path(source).resolve(strict=True)),
        training_data_path=str(Path(training_data).resolve(strict=True)),
        training_prompt_path=str(Path(training_prompt).resolve(strict=True)),
        training_reward_path=str(Path(training_reward).resolve(strict=True)),
        rl_recipe_path=str(Path(rl_recipe).resolve(strict=True)),
    )


def assert_resume_compatible(prior: dict[str, Any], current: dict[str, Any]) -> None:
    for field in SCIENTIFIC_HASH_FIELDS:
        if prior.get(field) != current.get(field):
            raise ValueError(f"resume rejected: {field} does not match the checkpoint contract")


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
