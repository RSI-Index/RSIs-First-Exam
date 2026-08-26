#!/usr/bin/env python3
"""Run the generic Layer 1 verifier."""

from __future__ import annotations

import argparse
from pathlib import Path

from policy_check import check
from score import write_reward
from validate_submission import validate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--policy", type=Path)
    args = parser.parse_args()
    try:
        check(args.project, args.policy, args.task)
        validate(args.output, args.project, args.task)
    except (OSError, ValueError, KeyError) as error:
        write_reward(args.logs, False, str(error))
        return 1
    write_reward(args.logs, True, "Layer 1 control and artifact contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
