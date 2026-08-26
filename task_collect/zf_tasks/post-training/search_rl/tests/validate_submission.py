#!/usr/bin/env python3
"""Validate the portable search-rl Layer 1 submission and its provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


EXPECTED_TREE = "9238a96476fa1607fc572f8d0e62d52cdd441c851a9d51b31c5d76000a7e780c"
REQUIRED_LAUNCHER = Path(
    "training_scripts/rl/recipe/deepresearch/run_deepresearch_fully_async_megatron.sh"
)


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def tree_sha256(root: Path) -> str:
    records: list[list[str]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if ".git" in relative.parts or "__pycache__" in relative.parts or path.is_dir():
            continue
        if path.is_symlink():
            records.append([relative.as_posix(), "symlink", os.readlink(path)])
        elif path.is_file():
            records.append([relative.as_posix(), "file", hashlib.sha256(path.read_bytes()).hexdigest()])
    encoded = json.dumps(records, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def within(relative: str, root: Path, label: str) -> Path:
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    if candidate == resolved_root or resolved_root not in candidate.parents:
        raise ValueError(f"{label} escapes or aliases its output root")
    return candidate


def validate(output: Path, project: Path, task: str) -> dict[str, Any]:
    if task != "search-rl":
        raise ValueError("unexpected task identity")
    output = output.resolve(strict=True)
    project = project.resolve(strict=True)
    if (project / "evaluation").exists() or not (project / REQUIRED_LAUNCHER).is_file():
        raise ValueError("candidate source redaction or launcher contract failed")
    if tree_sha256(project) != EXPECTED_TREE:
        raise ValueError("candidate source differs from the pinned redacted tree")

    submission = read_object(output / "submission.json")
    if submission.get("schema_version") != 1 or submission.get("task") != task:
        raise ValueError("submission identity or schema is invalid")
    if submission.get("validation_mode") != "layer1-smoke":
        raise ValueError("submission must identify the Layer 1 smoke mode")
    if submission.get("scientific_validation") != "not-run":
        raise ValueError("Layer 1 must not claim scientific validation")
    if submission.get("source_tree_sha256") != EXPECTED_TREE:
        raise ValueError("submission source digest is invalid")
    attempt_id = submission.get("attempt_id")
    if not isinstance(attempt_id, str) or not attempt_id:
        raise ValueError("submission attempt_id is missing")
    run_dir_raw = submission.get("run_dir")
    if not isinstance(run_dir_raw, str):
        raise ValueError("submission run_dir is missing")
    run_dir = within(run_dir_raw, output, "run_dir")
    if run_dir.parent != output / "attempts" or run_dir.name != attempt_id:
        raise ValueError("submission run_dir does not match attempt_id")

    staged = read_object(output / "staged_candidate.json")
    candidate = read_object(run_dir / "candidate_result.json")
    evaluation = read_object(run_dir / "evaluation.json")
    if any(item.get("attempt_id") != attempt_id for item in (staged, candidate, evaluation)):
        raise ValueError("attempt identity is inconsistent across artifacts")
    if staged.get("run_dir") != run_dir_raw:
        raise ValueError("staged run path does not match submission")
    if candidate.get("status") != "completed" or candidate.get("source_tree_sha256") != EXPECTED_TREE:
        raise ValueError("candidate result is incomplete or unpinned")
    checks = candidate.get("wiring_checks")
    if not isinstance(checks, dict) or not checks or not all(value is True for value in checks.values()):
        raise ValueError("candidate wiring checks are incomplete")
    if evaluation.get("passed") is not True:
        raise ValueError("smoke evaluation did not pass")
    for item in (staged, candidate, evaluation):
        if item.get("validation_mode") != "layer1-smoke" or item.get("scientific_validation") != "not-run":
            raise ValueError("an artifact makes an invalid validation claim")
    return submission


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    try:
        validate(args.output, args.project, args.task)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
