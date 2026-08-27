#!/usr/bin/env python3
"""Record, summarize, stage, restore, and prune durable search-RL attempt state."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from common import output_root, read_json, require_within, validate_attempt_id, write_json


def ledger_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def record(run_dir: Path, root: Path) -> None:
    evaluation = read_json(run_dir / "evaluation.json")
    attempt_id = validate_attempt_id(str(evaluation.get("attempt_id", "")))
    row = {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "status": "valid-smoke" if evaluation.get("passed") else "failed",
        "validation_mode": evaluation.get("validation_mode"),
        "scientific_validation": evaluation.get("scientific_validation"),
        "run_dir": str(run_dir.relative_to(root)),
    }
    ledger = root / "experiments.jsonl"
    rows = ledger_rows(ledger)
    existing = next((item for item in rows if item.get("attempt_id") == attempt_id), None)
    if existing is not None:
        if existing != row:
            raise ValueError("attempt ID already exists with different state")
        return
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("record", "stage"):
        command = subcommands.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
        command.add_argument("--output-root")
    summarize = subcommands.add_parser("summarize")
    summarize.add_argument("--output-root")
    restore = subcommands.add_parser("restore")
    restore.add_argument("--output-root")
    prune = subcommands.add_parser("prune")
    prune.add_argument("--attempt-id", required=True)
    prune.add_argument("--output-root")
    args = parser.parse_args()
    try:
        root = output_root(args.output_root)
        if args.command in ("record", "stage"):
            run_dir = require_within(args.run_dir, root / "attempts", "run directory")
        if args.command == "record":
            record(run_dir, root)
        elif args.command == "stage":
            evaluation = read_json(run_dir / "evaluation.json")
            if not evaluation.get("passed") or evaluation.get("validation_mode") != "layer1-smoke":
                raise ValueError("only a valid completed smoke run may be staged locally")
            write_json(
                root / "staged_candidate.json",
                {
                    "schema_version": 1,
                    "attempt_id": evaluation["attempt_id"],
                    "run_dir": str(run_dir.relative_to(root)),
                    "validation_mode": "layer1-smoke",
                    "scientific_validation": "not-run",
                },
            )
        elif args.command == "summarize":
            rows = ledger_rows(root / "experiments.jsonl")
            summary = {"schema_version": 1, "attempts": len(rows), "rows": rows}
            write_json(root / "summary.json", summary)
            print(json.dumps(summary, sort_keys=True))
        elif args.command == "restore":
            print(json.dumps(read_json(root / "staged_candidate.json"), sort_keys=True))
        elif args.command == "prune":
            attempt_id = validate_attempt_id(args.attempt_id)
            staged_path = root / "staged_candidate.json"
            if staged_path.is_file() and read_json(staged_path).get("attempt_id") == attempt_id:
                raise ValueError("refusing to prune the staged attempt")
            target = require_within(root / "attempts" / attempt_id, root / "attempts", "attempt")
            if target.is_dir():
                shutil.rmtree(target)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
