#!/usr/bin/env python3
"""Evaluate a Layer 1 candidate result without claiming scientific success."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import read_json, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        candidate = read_json(args.run_dir / "candidate_result.json")
        if candidate.get("status") != "completed":
            raise ValueError("candidate result is incomplete")
        if candidate.get("validation_mode") != "layer1-smoke":
            raise ValueError("only layer1-smoke results are accepted locally")
        checks = candidate.get("wiring_checks")
        if not isinstance(checks, dict) or not checks or not all(checks.values()):
            raise ValueError("candidate wiring checks did not all pass")
        write_json(
            args.run_dir / "evaluation.json",
            {
                "schema_version": 1,
                "task": "search-rl",
                "attempt_id": candidate.get("attempt_id"),
                "passed": True,
                "validation_mode": "layer1-smoke",
                "scientific_validation": "not-run",
                "reason": "redacted source and artifact wiring passed",
            },
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"smoke evaluation passed: {candidate.get('attempt_id')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
