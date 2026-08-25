#!/usr/bin/env python3
"""Fail-closed aggregate scorer for normalized Magpie pair scores."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable, Sequence

try:
    from .evaluation_contract import aggregate_reward
except ImportError:
    from evaluation_contract import aggregate_reward


def _atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _load_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("pair scores must be an ordinary JSONL file")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("pair score row must be an object")
            rows.append(value)
    return rows


def write_score(rows: Iterable[object], reward_dir: Path, artifact_dir: Path) -> float:
    """Write only scalar aggregate output; hidden prompts never enter details."""

    values: list[float] = []
    identifiers: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "score"}:
            raise ValueError("pair score row must contain only id and score")
        identifier = row["id"]
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("pair score ids must be unique")
        identifiers.add(identifier)
        raw_score = row["score"]
        if not isinstance(raw_score, (int, float)) or isinstance(raw_score, bool) or not math.isfinite(raw_score) or float(raw_score) not in {0.0, 0.5, 1.0}:
            raise ValueError("pair scores must be finite numeric win/tie/loss values")
        values.append(float(raw_score))
    reward = aggregate_reward(values)
    details = {"count": len(values), "wins": sum(score == 1.0 for score in values), "ties": sum(score == 0.5 for score in values), "losses": sum(score == 0.0 for score in values)}
    reward_destination = Path(reward_dir)
    artifact_destination = Path(artifact_dir)
    _atomic(reward_destination / "reward.txt", f"{reward}\n".encode("ascii"))
    _atomic(reward_destination / "reward.json", (json.dumps({"reward": reward}, separators=(",", ":"), allow_nan=False) + "\n").encode())
    _atomic(artifact_destination / "score_details.json", (json.dumps(details, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())
    return reward


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--reward-output", type=Path, required=True)
    parser.add_argument("--artifact-output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps({"reward": write_score(_load_rows(args.pairs), args.reward_output, args.artifact_output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
