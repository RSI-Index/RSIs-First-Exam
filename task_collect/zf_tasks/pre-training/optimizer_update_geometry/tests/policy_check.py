#!/usr/bin/env python3
"""Deterministic source-scope gate for the optimizer-encoding task."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


def ignored(relative: str) -> bool:
    parts = Path(relative).parts
    generated_dataset_helper = bool(
        Path(relative).parent.as_posix() == "3rdparty/Megatron-LM/megatron/core/datasets"
        and Path(relative).name.startswith("helpers_cpp.")
        and Path(relative).suffix == ".so"
    )
    return (
        ".git" in parts
        or "__pycache__" in parts
        or relative.endswith(".pyc")
        or generated_dataset_helper
    )


def inventory(root: Path) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if ignored(relative) or path.is_dir():
            continue
        if path.is_symlink():
            result[relative] = ("symlink", os.readlink(path))
        elif path.is_file():
            result[relative] = ("file", hashlib.sha256(path.read_bytes()).hexdigest())
        else:
            result[relative] = ("other", "")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    policy: dict[str, Any] = yaml.safe_load(args.policy.read_text())
    allowed = tuple(str(item) for item in policy["allowed_source_globs"])
    clean = inventory(args.clean)
    candidate = inventory(args.candidate)
    changed = sorted(path for path in clean.keys() | candidate.keys() if clean.get(path) != candidate.get(path))
    violations = [
        path
        for path in changed
        if not any(fnmatch.fnmatchcase(path, pattern) for pattern in allowed)
    ]
    report = {
        "policy_gate": 0 if violations else 1,
        "changed_paths": changed,
        "allowed_source_globs": list(allowed),
        "violations": violations,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if violations:
        raise SystemExit("source-scope violations: " + ", ".join(violations))


if __name__ == "__main__":
    main()
