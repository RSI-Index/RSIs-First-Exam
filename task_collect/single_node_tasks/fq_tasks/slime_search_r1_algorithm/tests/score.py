#!/usr/bin/env python3
"""Publish a finite Harbor reward from frozen aggregate metrics."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, "/task-tools")

from evaluation_contract import normalized_reward


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregates", type=Path, default=Path("/app/output/verifier/aggregates.json"))
    parser.add_argument("--reward", type=Path, default=Path("/logs/verifier/reward.json"))
    args = parser.parse_args()
    payload = json.loads(args.aggregates.read_text(encoding="utf-8"))
    if set(payload) != {"base", "candidate", "reference"}:
        raise ValueError("aggregate metrics must contain exactly base, candidate, and reference")
    values = {key: float(value) for key, value in payload.items()}
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values.values()):
        raise ValueError("aggregate metrics must be finite values in [0, 1]")
    reward = normalized_reward(values["candidate"], values["base"], values["reference"])
    document = {
        "base_score": values["base"],
        "candidate_score": values["candidate"],
        "reference_score": values["reference"],
        "reward": reward,
    }
    args.reward.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.reward.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.reward)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
