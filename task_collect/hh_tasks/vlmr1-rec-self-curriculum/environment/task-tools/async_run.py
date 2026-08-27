#!/usr/bin/env python3
"""async_run.py -- submit long training asynchronously, end the agent's turn, resume on terminal state.

This is the loop the reference template described in instruction.md and did not ship. The
contract with the harness:

  submit            launch the command detached, write status.json, return immediately
  status            print status.json (no model call; this is what the poller uses)
  complete-control  hand control back to the harness after the round's work is staged

The poll interval is deliberately long (default 1800 s) and the poller must not invoke a
model: a 6-round chain with per-round training measured in hours cannot afford an LLM call
per poll, and the reference design explicitly kept the poller model-free.

Preemption: on this cluster `-q normal` is preemptible and a preempted job RESTARTS FROM
ZERO, appending to the same .out. So `submit` records a resume directory and `status` judges
from the trainer's own last checkpoint, never from the presence of the log file.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

RUNS = Path("/app/output/attempts")


def status_path(attempt: str) -> Path:
    return RUNS / attempt / "status.json"


def write_status(attempt: str, **fields) -> dict:
    path = status_path(attempt)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(path.read_text()) if path.is_file() else {}
    state.update(fields)
    state["updated_at_monotonic"] = time.monotonic()
    path.write_text(json.dumps(state, indent=2) + "\n")
    return state


def cmd_submit(args: argparse.Namespace) -> int:
    if not args.command:
        print("nothing to run: put the training command after `--`")
        return 2
    run_dir = RUNS / args.attempt_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log = run_dir / "train.out"
    env = os.environ.copy()
    env["RESUME_RUN_DIR"] = str(run_dir)
    # start_new_session so the trainer survives this process exiting -- the agent's turn ends
    # while training continues.
    with log.open("ab") as handle:
        proc = subprocess.Popen(args.command, stdout=handle, stderr=subprocess.STDOUT,
                                cwd=args.cwd, env=env, start_new_session=True)
    state = write_status(args.attempt_id, attempt_id=args.attempt_id, pid=proc.pid,
                         state="running", terminal=False, command=args.command,
                         log=str(log), run_dir=str(run_dir),
                         poll_seconds=args.poll_seconds, restarts=0)
    print(json.dumps({"submitted": args.attempt_id, "pid": proc.pid,
                      "poll_seconds": args.poll_seconds,
                      "note": "end your turn now; do not sleep or poll"}, indent=2))
    return 0 if state else 1


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def cmd_status(args: argparse.Namespace) -> int:
    path = status_path(args.attempt_id)
    if not path.is_file():
        print(json.dumps({"state": "unknown", "terminal": False}))
        return 0
    state = json.loads(path.read_text())
    if state.get("state") == "running" and not _alive(int(state.get("pid", -1))):
        # Judge from the trainer's own artefacts, not from the log's existence: a preempted
        # rerun appends to the same file and would otherwise look like progress.
        run_dir = Path(state["run_dir"])
        finished = any(run_dir.rglob("trainer_state.json"))
        state = write_status(args.attempt_id,
                             state="succeeded" if finished else "failed",
                             terminal=True)
    print(json.dumps(state, indent=2))
    return 0


def cmd_complete_control(args: argparse.Namespace) -> int:
    marker = Path("/app/output/CONTROL_RELEASED")
    marker.write_text(json.dumps({"round": args.round, "released": True}, indent=2) + "\n")
    print(json.dumps({"control": "released", "round": args.round}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit")
    s.add_argument("--attempt-id", required=True)
    s.add_argument("--poll-seconds", type=int, default=1800)
    s.add_argument("--cwd", default="/app/project/src/open-r1-multimodal")
    s.add_argument("command", nargs=argparse.REMAINDER)
    s.set_defaults(fn=cmd_submit)

    t = sub.add_parser("status"); t.add_argument("--attempt-id", required=True); t.set_defaults(fn=cmd_status)
    c = sub.add_parser("complete-control"); c.add_argument("--round", type=int, default=0); c.set_defaults(fn=cmd_complete_control)

    args = parser.parse_args()
    if getattr(args, "command", None) and args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
