#!/usr/bin/env python3
"""Write a finite Layer 1 contract reward without scientific score claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_reward(logs: Path, passed: bool, reason: str) -> None:
    reward = 1.0 if passed else 0.0
    logs.mkdir(parents=True, exist_ok=True)
    payload = {
        "reward": reward,
        "reason": reason,
        "validation_mode": "layer1-smoke",
        "scientific_validation": "not-run",
    }
    (logs / "reward.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (logs / "reward.txt").write_text(f"{reward:.1f}\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--passed", action="store_true")
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    write_reward(args.logs, args.passed, args.reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
