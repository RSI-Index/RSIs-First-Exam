#!/usr/bin/env python3
"""Run one trusted TMAX attempt while preserving one agent tool call."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Sequence

from tmax_contract import atomic_write_json
from tmax_task_tool import finish_attempt


def _active_control(attempt: Path) -> bool:
    control = attempt / "control.json"
    if not control.is_file():
        return False
    try:
        state = json.loads(control.read_text())
        pid = int(state["pid"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError, TypeError):
        return False
    if state.get("status") != "running" or pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def run_process(*, attempt: Path, command: Sequence[str], poll_seconds: float = 30.0) -> dict:
    """Run one process group, poll it, and atomically publish terminal state."""
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    if _active_control(attempt):
        raise ValueError(f"active attempt already exists: {attempt}")
    if not command:
        raise ValueError("command must not be empty")

    process = subprocess.Popen(list(command), start_new_session=True)
    control = {
        "pid": process.pid,
        "process_group_id": process.pid,
        "status": "running",
        "started_at_epoch": time.time(),
        "command": list(command),
    }
    atomic_write_json(attempt / "control.json", control)
    poll_count = 0
    try:
        while process.poll() is None:
            time.sleep(poll_seconds)
            poll_count += 1
            print(f"TMAX attempt heartbeat pid={process.pid} polls={poll_count}", flush=True)
    except BaseException:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise

    exit_code = process.wait()
    state = {
        "status": "completed" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "poll_count": poll_count,
        "pid": process.pid,
        "finished_at_epoch": time.time(),
    }
    atomic_write_json(attempt / "async_status.json", state)
    atomic_write_json(attempt / "control.json", {**control, **state})
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    submit = subparsers.add_parser("submit")
    submit.add_argument("--attempt", required=True, type=Path)
    submit.add_argument("--host", required=True)
    submit.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()

    if args.command == "submit":
        contract = args.attempt / "run_contract.json"
        if not contract.is_file():
            parser.error(f"attempt contract is missing: {contract}")
        blaunch = os.environ.get("BLAUNCH", "blaunch")
        state = run_process(
            attempt=args.attempt,
            command=[blaunch, "-z", args.host, "/task-tools/run_tmax_attempt.sh", str(contract)],
            poll_seconds=args.poll_seconds,
        )
        counters = args.attempt / "trusted_counters.json"
        if not counters.is_file():
            atomic_write_json(
                counters,
                {
                    "optimizer_updates": 0,
                    "training_trajectories": 0,
                    "generated_tokens": 0,
                    "sandbox_steps": 0,
                    "gpu_count": 64,
                    "metrics": {},
                },
            )
        terminal = finish_attempt(attempt=args.attempt, exit_code=int(state["exit_code"]), counters_file=counters)
        print(json.dumps({**state, "attempt": terminal}, sort_keys=True))
        return int(state["exit_code"])
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
