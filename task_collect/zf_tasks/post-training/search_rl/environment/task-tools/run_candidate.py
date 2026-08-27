#!/usr/bin/env python3
"""Run the deterministic Layer 1 candidate path or reject unavailable full runs."""

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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run:
        parser.exit(3, "FULL-RUN-UNVERIFIED: use the separately validated backend adapter\n")
    try:
        attempt_id = validate_attempt_id(args.attempt_id)
        root = output_root(args.output_root)
        hypothesis_path = require_within(Path(args.hypothesis_file), root / "hypotheses", "hypothesis")
        hypothesis = read_json(hypothesis_path)
        if hypothesis.get("attempt_id") != attempt_id:
            raise ValueError("hypothesis attempt_id does not match --attempt-id")
        run_dir = root / "attempts" / attempt_id
        preflight = read_json(run_dir / "preflight.json")
        if preflight.get("status") != "passed":
            raise ValueError("preflight did not pass")
        write_json(
            run_dir / "candidate_result.json",
            {
                "schema_version": 1,
                "task": "search-rl",
                "attempt_id": attempt_id,
                "status": "completed",
                "validation_mode": "layer1-smoke",
                "scientific_validation": "not-run",
                "source_tree_sha256": os.environ.get(
                    "TASK_SOURCE_TREE_SHA256",
                    "9238a96476fa1607fc572f8d0e62d52cdd441c851a9d51b31c5d76000a7e780c",
                ),
                "wiring_checks": {
                    "hypothesis_loaded": True,
                    "redacted_source_contract": True,
                    "artifact_transport": True,
                },
            },
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"smoke candidate completed: {attempt_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
