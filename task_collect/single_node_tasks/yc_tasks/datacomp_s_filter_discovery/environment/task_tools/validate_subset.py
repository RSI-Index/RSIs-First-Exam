#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


UID_DTYPE = np.dtype("u8,u8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_uid_array(path: Path, label: str) -> np.ndarray:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular file")
    value = np.load(path, allow_pickle=False)
    if value.ndim != 1:
        raise ValueError(f"{label} must be rank one")
    if value.dtype != UID_DTYPE or value.dtype.hasobject:
        raise ValueError(f"{label} must have exact non-object dtype {UID_DTYPE}")
    return value


def strictly_sorted(value: np.ndarray) -> bool:
    if len(value) < 2:
        return True
    left0, left1 = value["f0"][:-1], value["f1"][:-1]
    right0, right1 = value["f0"][1:], value["f1"][1:]
    return bool(np.all((left0 < right0) | ((left0 == right0) & (left1 < right1))))


def validate_subset(subset_path: Path | str, universe_path: Path | str) -> dict[str, Any]:
    subset_path, universe_path = Path(subset_path), Path(universe_path)
    subset = load_uid_array(subset_path, "subset")
    universe = load_uid_array(universe_path, "UID universe")
    if len(subset) == 0:
        raise ValueError("subset must be non-empty")
    if not strictly_sorted(subset):
        raise ValueError("subset must be unique and lexicographically ascending")
    if not strictly_sorted(universe):
        raise ValueError("UID universe must be unique and lexicographically ascending")
    indices = np.searchsorted(universe, subset)
    present = indices < len(universe)
    present[present] &= universe[indices[present]] == subset[present]
    if not bool(np.all(present)):
        raise ValueError("subset contains one or more out-of-universe UIDs")
    return {
        "schema_version": 1,
        "status": "pass",
        "count": int(len(subset)),
        "dtype": str(subset.dtype),
        "subset_sha256": sha256(subset_path),
        "universe_sha256": sha256(universe_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    record = validate_subset(args.subset, args.universe)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()

