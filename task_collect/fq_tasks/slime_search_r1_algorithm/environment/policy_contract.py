"""Fail-closed provenance checks for the Slime Search-R1 submission."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from task_contract import (
        BM25_INDEX_REVISION,
        DATASET_REVISION,
        MODEL_ID,
        MODEL_REVISION,
        NUM_ROLLOUT,
        SEARCH_R1_COMMIT,
        SLIME_COMMIT,
        algorithm_source_sha256,
    )
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from task_contract import (
        BM25_INDEX_REVISION,
        DATASET_REVISION,
        MODEL_ID,
        MODEL_REVISION,
        NUM_ROLLOUT,
        SEARCH_R1_COMMIT,
        SLIME_COMMIT,
        algorithm_source_sha256,
    )


REQUIRED_PROVENANCE_FIELDS = (
    "algorithm_sha256",
    "base_model",
    "base_model_revision",
    "bm25_index_revision",
    "dataset_revision",
    "num_rollout",
    "optimization_attempts",
    "search_r1_commit",
    "selected_attempt",
    "slime_commit",
    "training_command",
    "training_command_sha256",
    "web_search",
    "world_size",
)


def _command_sha256(command: list[str]) -> str:
    return hashlib.sha256(json.dumps(command, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_provenance(payload: dict[str, Any], algorithm_root: str | Path | None = None) -> list[str]:
    errors = []
    if not isinstance(payload, dict):
        return ["provenance must be a JSON object"]
    for field in REQUIRED_PROVENANCE_FIELDS:
        if field not in payload:
            errors.append(f"missing provenance field: {field}")
    if errors:
        return errors

    expected = {
        "base_model": MODEL_ID,
        "base_model_revision": MODEL_REVISION,
        "bm25_index_revision": BM25_INDEX_REVISION,
        "dataset_revision": DATASET_REVISION,
        "num_rollout": NUM_ROLLOUT,
        "search_r1_commit": SEARCH_R1_COMMIT,
        "slime_commit": SLIME_COMMIT,
        "web_search": "disabled",
        "world_size": 8,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            errors.append(f"provenance {field} does not match the frozen contract")
    command = payload.get("training_command")
    if not isinstance(command, list) or not command or not all(isinstance(part, str) for part in command):
        errors.append("training_command must be a non-empty string list")
    elif payload.get("training_command_sha256") != _command_sha256(command):
        errors.append("training command hash mismatch")
    attempts = payload.get("optimization_attempts")
    if not isinstance(attempts, list) or not attempts or payload.get("selected_attempt") not in attempts:
        errors.append("selected attempt is absent from optimization_attempts")
    if algorithm_root is not None:
        try:
            actual_hash = algorithm_source_sha256(algorithm_root)
        except (OSError, ValueError) as exc:
            errors.append(f"invalid algorithm source tree: {exc}")
        else:
            if payload.get("algorithm_sha256") != actual_hash:
                errors.append("algorithm source hash mismatch")
    return errors


def tree_sha256(path: str | Path) -> str:
    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("frozen root must be a regular directory")
    digest = hashlib.sha256()
    for entry in sorted(root.rglob("*")):
        if entry.is_symlink():
            raise ValueError("frozen tree contains a symlink")
        if not entry.is_file():
            continue
        relative = entry.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(entry.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
