#!/usr/bin/env python3
"""Export numeric model parameters from the fixed trainer checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_digest(arrays: dict[str, np.ndarray], prefix: str = "") -> str:
    digest = hashlib.sha256()
    selected = [(key, value) for key, value in arrays.items() if key.startswith(prefix)]
    if not selected:
        raise ValueError(f"no arrays match prefix {prefix!r}")
    for key, value in sorted(selected):
        array = np.ascontiguousarray(value)
        digest.update(key.encode() + b"\0")
        digest.update(array.dtype.str.encode() + b"\0")
        digest.update(json.dumps(array.shape).encode() + b"\0")
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def export(checkpoint: Path, output: Path) -> dict:
    arrays: dict[str, np.ndarray] = {}
    with np.load(checkpoint, allow_pickle=False) as source:
        for key in source.files:
            if not key.startswith("params/"):
                continue
            value = source[key]
            if value.dtype.hasobject:
                raise ValueError(f"object array in model parameters: {key}")
            arrays[key.removeprefix("params/")] = value
    if not arrays or not any(key.startswith("img/") for key in arrays) or not any(key.startswith("txt/") for key in arrays):
        raise ValueError("checkpoint does not contain complete img/txt parameter trees")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)
    return {
        "model_sha256": sha256(output),
        "image_parameter_digest": array_digest(arrays, "img/"),
        "array_count": len(arrays),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    record = export(args.checkpoint, args.output)
    args.record.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()

