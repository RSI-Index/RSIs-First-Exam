#!/usr/bin/env python3
"""Official L/14 top-30% starter; replace this file with a research selector."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.dataset as pads


UID_DTYPE = np.dtype("u8,u8")
SCORE_COLUMN = "clip_l14_similarity_score"


def encode_uids(values: list[str]) -> np.ndarray:
    output = np.empty(len(values), dtype=UID_DTYPE)
    for index, value in enumerate(values):
        uid = str(value)
        if len(uid) != 32:
            raise ValueError(f"UID must contain exactly 32 hexadecimal characters: {uid!r}")
        output[index] = (int(uid[:16], 16), int(uid[16:], 16))
    output.sort()
    return output


def select(metadata_dir: Path, fraction: float = 0.30) -> np.ndarray:
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must lie strictly between zero and one")
    dataset = pads.dataset(metadata_dir, format="parquet")
    table = dataset.to_table(columns=["uid", SCORE_COLUMN])
    scores = np.asarray(table[SCORE_COLUMN].to_numpy(zero_copy_only=False))
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("score column must be finite and rank one")
    threshold_index = int(len(scores) * fraction)
    threshold = np.sort(scores)[::-1][threshold_index]
    mask = scores >= threshold
    uids = table["uid"].to_pylist()
    selected = encode_uids([uid for uid, keep in zip(uids, mask, strict=True) if keep])
    if len(selected) == 0:
        raise ValueError("selector produced an empty subset")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fraction", type=float, default=0.30)
    args = parser.parse_args()
    del args.features_dir, args.seed
    selected = select(args.metadata_dir, args.fraction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, selected, allow_pickle=False)


if __name__ == "__main__":
    main()

