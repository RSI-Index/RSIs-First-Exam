#!/usr/bin/env python3
"""Validate one search-RL attempt before launch."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from common import output_root, read_json, require_within, validate_attempt_id, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--hypothesis-file", required=True)
    parser.add_argument("--output-root")
    args = parser.parse_args()
    try:
        attempt_id = validate_attempt_id(args.attempt_id)
        root = output_root(args.output_root)
        hypothesis = require_within(Path(args.hypothesis_file), root / "hypotheses", "hypothesis")
        payload = read_json(hypothesis)
        if payload.get("attempt_id") != attempt_id:
            raise ValueError("hypothesis attempt_id does not match --attempt-id")
        project_raw = os.environ.get("TASK_PROJECT")
        if project_raw:
            project = Path(project_raw)
            if (project / "evaluation").exists():
                raise ValueError("candidate project exposes forbidden evaluation/")
            launcher = project / "training_scripts/rl/recipe/deepresearch/run_deepresearch_fully_async_megatron.sh"
            if not launcher.is_file():
                raise ValueError("pinned QUEST training launcher is missing")
        run_dir = root / "attempts" / attempt_id
        write_json(
            run_dir / "preflight.json",
            {
                "schema_version": 1,
                "attempt_id": attempt_id,
                "status": "passed",
                "validation_mode": "layer1-smoke",
                "safe_to_retry": True,
            },
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"preflight passed: {attempt_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
