#!/usr/bin/env python3
"""Launch only the frozen eight-rank Magpie SFT recipe and verify its export."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
from typing import Any, Sequence

try:  # Package imports for CPU contract tests.
    from .task_contract import CONTRACT, RUNTIME_PATHS, accumulation_for_world_size
except ImportError:  # Flat /tests execution in the Harbor image.
    from task_contract import CONTRACT, RUNTIME_PATHS, accumulation_for_world_size


ACCELERATE = "/opt/train-venv/bin/accelerate"
_DATASET_MARKER = "__CANONICAL_DATASET__"
_OUTPUT_MARKER = "__VERIFIER_OUTPUT__"
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
_NUMBERED_SHARD = re.compile(r"^model-(\d{5})-of-(\d{5})\.safetensors$")
_CHECKPOINT_DIR = re.compile(r"^checkpoint-(\d+)$")
_VOLATILE_CONFIG_FIELDS = frozenset({"_name_or_path", "_commit_hash", "transformers_version"})


def _ordinary_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink() and not any(parent.is_symlink() for parent in path.parents)


def _ordinary_directory(path: Path) -> bool:
    return path.is_dir() and not path.is_symlink() and not any(parent.is_symlink() for parent in path.parents)


def _ensure_private_directory(path: Path) -> None:
    path = Path(path)
    if path.exists():
        if not _ordinary_directory(path):
            raise ValueError("verifier output parent must be an ordinary directory")
        metadata = path.stat()
        if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
            raise ValueError("verifier output parent must be private to the verifier")
        return
    if not _ordinary_directory(path.parent):
        raise ValueError("verifier output parent must have an ordinary parent")
    path.mkdir(mode=0o700)
    os.chmod(path, 0o700)
    _ensure_private_directory(path)


def _recipe_path() -> Path:
    local = Path(__file__).resolve().parent / "frozen_train.yaml"
    if local.is_file():  # Flat verifier staging may place recipe beside this module.
        return local
    return Path(__file__).resolve().parents[1] / "environment" / "frozen_train.yaml"


def _frozen_recipe() -> str:
    recipe = _recipe_path()
    if not _ordinary_file(recipe):
        raise ValueError("frozen training recipe must be an ordinary file")
    rendered = recipe.read_text(encoding="utf-8")
    if rendered.count(_DATASET_MARKER) != 1 or rendered.count(_OUTPUT_MARKER) != 1:
        raise ValueError("frozen training recipe must contain exactly two runtime placeholders")
    if accumulation_for_world_size(8) != 4:
        raise ValueError("frozen recipe no longer preserves global batch 32")
    return rendered


def write_effective_recipe(dataset: Path, output: Path, destination: Path) -> Path:
    """Write the immutable recipe after replacing exactly its two allowed values."""

    dataset, output, destination = Path(dataset), Path(output), Path(destination)
    if not _ordinary_file(dataset):
        raise ValueError("canonical dataset must be an ordinary file")
    if destination.exists() and (destination.is_symlink() or not destination.is_file()):
        raise ValueError("effective recipe destination must be an ordinary file")
    rendered = _frozen_recipe().replace(_DATASET_MARKER, str(dataset)).replace(_OUTPUT_MARKER, str(output))
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
        os.chmod(destination, 0o600)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def training_command(effective_recipe: Path) -> list[str]:
    return [ACCELERATE, "launch", "--num_processes", "8", "-m", "axolotl.cli.train", str(Path(effective_recipe))]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_file(path: Path, name: str) -> dict[str, Any]:
    if not _ordinary_file(path):
        raise ValueError(f"{name} must be an ordinary file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must contain valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _finite_loss(state: dict[str, Any]) -> float:
    candidates: list[object] = [state.get("train_loss"), state.get("loss")]
    history = state.get("log_history")
    if isinstance(history, list):
        for entry in reversed(history):
            if isinstance(entry, dict):
                candidates.extend((entry.get("train_loss"), entry.get("loss")))
    for value in candidates:
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return float(value)
    raise ValueError("trainer state must contain a finite loss")


def _checkpoint_loss(output: Path) -> float:
    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    for path in output.iterdir():
        if not path.name.startswith("checkpoint-"):
            continue
        if not _ordinary_directory(path):
            raise ValueError("checkpoint directory must be ordinary")
        if not _CHECKPOINT_DIR.fullmatch(path.name):
            raise ValueError("checkpoint directory has an invalid name")
        state = _json_file(path / "trainer_state.json", "trainer state")
        global_step = state.get("global_step")
        if not isinstance(global_step, int) or isinstance(global_step, bool) or global_step < 0:
            raise ValueError("trainer state must contain a valid global_step")
        _finite_loss(state)
        candidates.append((global_step, path, state))
    if not candidates:
        raise ValueError("checkpoint is missing trainer state evidence")
    highest = max(item[0] for item in candidates)
    final = [item for item in candidates if item[0] == highest]
    if len(final) != 1:
        raise ValueError("trainer state has an ambiguous final global_step")
    return _finite_loss(final[0][2])


def _ordinary_tree(path: Path) -> None:
    for current, directories, files in os.walk(path, followlinks=False):
        current_path = Path(current)
        for name in [*directories, *files]:
            child = current_path / name
            metadata = child.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)):
                raise ValueError("checkpoint contains a symlink or non-regular file")


def _safe_open(path: Path):
    """Open with the official reader installed in the pinned inference runtime."""

    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("official safetensors reader is required in /opt/infer-venv") from exc
    return safe_open(str(path), framework="pt", device="cpu")


def _safe_tensor_metadata(path: Path) -> dict[str, tuple[str, tuple[int, ...]]]:
    if not _ordinary_file(path):
        raise ValueError("model shard must be an ordinary safetensors file")
    tensors: dict[str, tuple[str, tuple[int, ...]]] = {}
    try:
        with _safe_open(path) as reader:
            keys = list(reader.keys())
            if not keys or not all(isinstance(name, str) and name for name in keys):
                raise ValueError("safetensors shard contains no tensors")
            for name in keys:
                tensor_slice = reader.get_slice(name)
                dtype, shape = tensor_slice.get_dtype(), tensor_slice.get_shape()
                if not isinstance(dtype, str) or not isinstance(shape, (list, tuple)):
                    raise ValueError("safetensors shard has invalid tensor metadata")
                if not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in shape):
                    raise ValueError("safetensors shard has invalid tensor shape")
                tensors[name] = (dtype, tuple(shape))
    except Exception as exc:
        raise ValueError("official safetensors reader rejected shard") from exc
    if not tensors:
        raise ValueError("safetensors shard contains no tensors")
    return tensors


def _model_shards(directory: Path) -> list[Path]:
    root_files = [path for path in directory.iterdir() if path.is_file() and not path.is_symlink()]
    forbidden = [path.name for path in root_files if path.suffix in {".bin", ".pt", ".pth"} or path.name.startswith("pytorch_model")]
    if forbidden:
        raise ValueError("checkpoint contains a forbidden non-safetensors weight format")
    index = directory / "model.safetensors.index.json"
    numbered = sorted(path for path in root_files if _NUMBERED_SHARD.fullmatch(path.name))
    single = directory / "model.safetensors"
    permitted = {"model.safetensors", "model.safetensors.index.json", *(path.name for path in numbered)}
    unexpected = [path.name for path in root_files if path.name.endswith(".safetensors") and path.name not in permitted]
    if unexpected:
        raise ValueError("checkpoint contains an unexpected safetensors weight file")
    if index.exists():
        index_payload = _json_file(index, "safetensors index")
        weights = index_payload.get("weight_map")
        if not isinstance(weights, dict) or not weights:
            raise ValueError("safetensors index must contain a non-empty weight_map")
        referenced: set[str] = set()
        for name in weights.values():
            if not isinstance(name, str) or Path(name).name != name or not _NUMBERED_SHARD.fullmatch(name):
                raise ValueError("safetensors index references an unsafe shard")
            referenced.add(name)
        actual = {path.name for path in numbered}
        if referenced != actual:
            raise ValueError("safetensors index shard set is incomplete or has extras")
        totals = {_NUMBERED_SHARD.fullmatch(path.name).group(2) for path in numbered}  # type: ignore[union-attr]
        positions = {_NUMBERED_SHARD.fullmatch(path.name).group(1) for path in numbered}  # type: ignore[union-attr]
        if len(totals) != 1 or len(numbered) != int(next(iter(totals))) or positions != {f"{value:05d}" for value in range(1, len(numbered) + 1)}:
            raise ValueError("safetensors index shards are not a complete numbered set")
        if single.exists():
            raise ValueError("checkpoint cannot contain both single and indexed weights")
        return numbered
    if numbered or not _ordinary_file(single):
        raise ValueError("checkpoint shard layout without an index must contain exactly model.safetensors")
    return [single]


def _tensor_map(shards: list[Path]) -> dict[str, tuple[str, tuple[int, ...]]]:
    tensors: dict[str, tuple[str, tuple[int, ...]]] = {}
    for shard in shards:
        for name, metadata in _safe_tensor_metadata(shard).items():
            if name in tensors:
                raise ValueError("safetensors shards duplicate a tensor")
            tensors[name] = metadata
    return tensors


def _normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key not in _VOLATILE_CONFIG_FIELDS}


def validate_checkpoint(output: Path, *, base_model: Path = Path(RUNTIME_PATHS["student_model"])) -> dict[str, object]:
    """Fail closed unless a full ordinary Llama checkpoint matches the pinned base."""

    candidate, base = Path(output), Path(base_model)
    if not _ordinary_directory(candidate) or not _ordinary_directory(base):
        raise ValueError("checkpoint and pinned base must be ordinary directories")
    _ordinary_tree(candidate)
    model_config = _json_file(candidate / "config.json", "checkpoint config")
    base_config = _json_file(base / "config.json", "base model config")
    architectures = model_config.get("architectures")
    if not isinstance(architectures, list) or "LlamaForCausalLM" not in architectures:
        raise ValueError("checkpoint config must declare LlamaForCausalLM")
    if _normalized_config(model_config) != _normalized_config(base_config):
        raise ValueError("checkpoint config must fully match the pinned base")
    tokenizer_hashes: dict[str, str] = {}
    for name in _TOKENIZER_FILES:
        expected, actual = base / name, candidate / name
        if not _ordinary_file(expected) or not _ordinary_file(actual):
            raise ValueError(f"checkpoint tokenizer file is missing: {name}")
        tokenizer_hashes[name] = _sha256_file(actual)
        if tokenizer_hashes[name] != _sha256_file(expected):
            raise ValueError(f"checkpoint tokenizer hash differs from pinned base: {name}")
    shards = _model_shards(candidate)
    base_shards = _model_shards(base)
    if _tensor_map(shards) != _tensor_map(base_shards):
        raise ValueError("checkpoint tensor metadata must match the pinned base")
    loss = _checkpoint_loss(candidate)
    digest = hashlib.sha256()
    for path in [candidate / "config.json", *[candidate / name for name in _TOKENIZER_FILES], *shards]:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return {"loss": loss, "checkpoint_sha256": digest.hexdigest(), "tokenizer_sha256": tokenizer_hashes}


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))
            handle.write("\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def train_candidate(dataset: Path, output: Path, *, base_model: Path = Path(RUNTIME_PATHS["student_model"])) -> dict[str, object]:
    """Run exactly one fixed accelerator invocation, then write a finite summary."""

    dataset, output = Path(dataset), Path(output)
    if not _ordinary_file(dataset):
        raise ValueError("canonical dataset must be an ordinary file")
    if output.is_symlink():
        raise ValueError("checkpoint output must not be a symlink")
    _ensure_private_directory(output.parent)
    if output.exists() and not _ordinary_directory(output):
        raise ValueError("checkpoint output must be an ordinary directory")
    effective = write_effective_recipe(dataset, output, output.parent / "effective_train.yaml")
    command = training_command(effective)
    environment = os.environ.copy()
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    subprocess.run(command, check=True, env=environment)
    checked = validate_checkpoint(output, base_model=base_model)
    summary = {
        **checked,
        "command_sha256": hashlib.sha256("\0".join(command).encode("utf-8")).hexdigest(),
        "config_sha256": _sha256_file(effective),
        "dataset_sha256": _sha256_file(dataset),
        "world_size": 8,
        "global_batch_size": CONTRACT.global_batch_size,
    }
    _atomic_json(output.parent / "train_summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = train_candidate(args.dataset, args.output)
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
