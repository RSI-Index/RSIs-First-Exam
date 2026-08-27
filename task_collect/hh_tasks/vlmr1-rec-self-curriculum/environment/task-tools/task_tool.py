#!/usr/bin/env python3
"""task_tool.py -- stage / audit / summarize. The agent-facing half of the submission path.

  summarize  read a finished attempt's real counters and report them (no claims, only files)
  stage      copy one selected checkpoint into /app/output/submission as an HF-format dir
  audit      the checks the verifier will run, run early so the agent is not surprised

`audit` deliberately duplicates verifier logic. The agent should be able to fail locally for
free rather than burn a round discovering that the budget counter was off by one step.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REQUIRED_PROVENANCE = [
    "base_model", "base_weight_sha256", "round_index", "improver_diff_sha256",
    "training_rows_selected", "training_rows_generated", "generator_model",
    "optimizer_steps", "completions_consumed", "tool_gpu_seconds", "training_command",
    "upstream_commits", "downloads", "evaluation_commands", "carry_manifest_sha256",
    "web_search",
]
CONTRACT_STEPS = 150


def cmd_summarize(args: argparse.Namespace) -> int:
    attempt = Path(args.attempt)
    states = sorted(attempt.rglob("trainer_state.json"))
    if not states:
        print(json.dumps({"attempt": str(attempt), "state": "no trainer_state.json found",
                          "hint": "the run has not reached a checkpoint yet"}, indent=2))
        return 0
    state = json.loads(states[-1].read_text())
    log = state.get("log_history", [])
    print(json.dumps({
        "attempt": str(attempt),
        "global_step": state.get("global_step"),
        "epoch": state.get("epoch"),
        "last_logged": log[-1] if log else None,
        "reward_trend": [row.get("reward") for row in log if "reward" in row][-10:],
        "checkpoints": [p.parent.name for p in states],
    }, indent=2))
    return 0


def cmd_stage(args: argparse.Namespace) -> int:
    src, dest = Path(args.checkpoint), Path(args.submission)
    if not src.is_dir():
        print(f"not a directory: {src}")
        return 2
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for path in src.iterdir():
        if path.is_file():
            shutil.copy2(path, dest / path.name)
    trainer_state = src / "trainer_state.json"
    if trainer_state.is_file():
        shutil.copy2(trainer_state, dest / "trainer_state.json")
    have = {p.name for p in dest.iterdir()}
    missing = [n for n in ("config.json", "trainer_state.json") if n not in have]
    print(json.dumps({"staged": str(dest), "files": sorted(have), "missing": missing}, indent=2))
    return 0 if not missing else 3


def cmd_audit(args: argparse.Namespace) -> int:
    root = Path(args.output_root)
    problems = []
    submission = root / "submission"
    if not submission.is_dir():
        problems.append("no /app/output/submission -- nothing would be scored")
    else:
        ts = submission / "trainer_state.json"
        if not ts.is_file():
            problems.append("submission lacks trainer_state.json; the verifier reads the budget from it")
        else:
            step = int(json.loads(ts.read_text()).get("global_step", -1))
            if step != CONTRACT_STEPS:
                problems.append(f"global_step={step}, contract requires exactly {CONTRACT_STEPS}")
    curriculum = root / "curriculum.jsonl"
    if not curriculum.is_file():
        problems.append("no curriculum.jsonl -- the round's training rows must be submitted")
    prov_path = root / "provenance.json"
    if not prov_path.is_file():
        problems.append("no provenance.json")
    else:
        prov = json.loads(prov_path.read_text())
        for key in REQUIRED_PROVENANCE:
            if key not in prov:
                problems.append(f"provenance.json missing required field: {key}")
        if prov.get("web_search") != "disabled":
            problems.append('provenance.json web_search must be "disabled"')
    print(json.dumps({"ok": not problems, "problems": problems}, indent=2))
    return 0 if not problems else 3


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("summarize"); a.add_argument("--attempt", required=True); a.set_defaults(fn=cmd_summarize)
    b = sub.add_parser("stage")
    b.add_argument("--checkpoint", required=True)
    b.add_argument("--submission", default="/app/output/submission")
    b.set_defaults(fn=cmd_stage)
    c = sub.add_parser("audit"); c.add_argument("--output-root", default="/app/output"); c.set_defaults(fn=cmd_audit)
    args = parser.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
