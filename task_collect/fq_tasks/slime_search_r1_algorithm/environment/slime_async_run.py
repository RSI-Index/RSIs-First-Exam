#!/usr/bin/env python3
"""Durable same-session experiment lifecycle for long Slime training attempts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from async_contract import append_jsonl, atomic_write_json, read_events, read_json, transition_status, utc_now, validate_attempt_id
    from task_contract import (
        BM25_INDEX_REVISION,
        DATASET_REVISION,
        MODEL_ID,
        MODEL_REVISION,
        NUM_ROLLOUT,
        SEARCH_R1_COMMIT,
        SLIME_COMMIT,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from async_contract import append_jsonl, atomic_write_json, read_events, read_json, transition_status, utc_now, validate_attempt_id
    from task_contract import (
        BM25_INDEX_REVISION,
        DATASET_REVISION,
        MODEL_ID,
        MODEL_REVISION,
        NUM_ROLLOUT,
        SEARCH_R1_COMMIT,
        SLIME_COMMIT,
    )


DEFAULT_OUTPUT_ROOT = Path("/app/output")
TRUSTED_RUN_ROOT = Path("/var/lib/slime-task/runs")


def _attempt_dir(root: Path, attempt_id: str) -> Path:
    return root / "attempts" / validate_attempt_id(attempt_id)


def _attempt_ids(events: list[dict]) -> list[str]:
    result = []
    seen = set()
    for event in events:
        attempt_id = event.get("attempt_id")
        if isinstance(attempt_id, str) and attempt_id not in seen:
            seen.add(attempt_id)
            result.append(attempt_id)
    return result


def _worker(root: Path, attempt_id: str) -> int:
    attempt = _attempt_dir(root, attempt_id)
    status_path = attempt / "status.json"
    transition_status(status_path, "running", worker_pid=os.getpid())
    append_jsonl(root / "experiments.jsonl", {"attempt_id": attempt_id, "event": "running", "timestamp": utc_now()})
    try:
        subprocess.run(
            ["sudo", "-n", "/task-tools/root_supervisor_entry.py", "--attempt-id", attempt_id],
            check=True,
        )
    except BaseException as exc:
        transition_status(status_path, "failed", error=f"{type(exc).__name__}: {exc}")
        append_jsonl(
            root / "experiments.jsonl",
            {"attempt_id": attempt_id, "event": "failed", "error": f"{type(exc).__name__}: {exc}", "timestamp": utc_now()},
        )
        return 1
    transition_status(status_path, "completed")
    append_jsonl(
        root / "experiments.jsonl",
        {"attempt_id": attempt_id, "event": "completed", "num_rollout": NUM_ROLLOUT, "timestamp": utc_now()},
    )
    return 0


def submit_attempt(root: Path, attempt_id: str, hypothesis: str) -> dict:
    attempt_id = validate_attempt_id(attempt_id)
    if not isinstance(hypothesis, str) or len(hypothesis.strip()) < 12:
        raise ValueError("hypothesis must state a falsifiable mechanism")
    attempt = _attempt_dir(root, attempt_id)
    if attempt.exists():
        raise ValueError("attempt id cannot be reused")
    for status_path in (root / "attempts").glob("*/status.json"):
        status = read_json(status_path).get("status")
        if status in {"queued", "running"}:
            raise ValueError("another attempt is queued or running")
    attempt.mkdir(parents=True)
    payload = {
        "attempt_id": attempt_id,
        "created_at": utc_now(),
        "hypothesis": hypothesis.strip(),
        "status": "queued",
        "updated_at": utc_now(),
    }
    atomic_write_json(attempt / "status.json", payload)
    append_jsonl(
        root / "experiments.jsonl",
        {"attempt_id": attempt_id, "event": "submitted", "hypothesis": hypothesis.strip(), "timestamp": utc_now()},
    )
    log = (attempt / "worker.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, __file__, "_worker", "--attempt-id", attempt_id, "--output-root", str(root)],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log.close()
    payload["worker_pid"] = process.pid
    current = read_json(attempt / "status.json")
    if current.get("status") == "queued":
        current["worker_pid"] = process.pid
        atomic_write_json(attempt / "status.json", current)
        payload = current
    else:
        payload = current
    return payload


def select_attempt(root: Path, attempt_id: str) -> dict:
    attempt = _attempt_dir(root, attempt_id)
    status = read_json(attempt / "status.json")
    if status.get("status") != "completed":
        raise ValueError("selected attempt must be completed")
    trusted_train = TRUSTED_RUN_ROOT / attempt_id / "train" if root == DEFAULT_OUTPUT_ROOT else attempt / "train"
    run_contract = read_json(trusted_train / "run-contract.json")
    if run_contract.get("num_rollout") != NUM_ROLLOUT or run_contract.get("world_size") != 8:
        raise ValueError("selected attempt is not a completed full-budget 8-GPU run")
    command_record = read_json(trusted_train / "training-command.json")
    if command_record.get("sha256") != run_contract.get("training_command_sha256"):
        raise ValueError("training command records disagree")
    source_model = trusted_train / "hf_model"
    if not source_model.is_dir():
        raise ValueError("selected attempt has no HF checkpoint")
    target_model = root / "hf_model"
    if target_model.exists() or (root / "selected-attempt.json").exists():
        raise ValueError("a final attempt is already selected")
    shutil.copytree(source_model, target_model, symlinks=False)
    attempts = _attempt_ids(read_events(root / "experiments.jsonl"))
    provenance = {
        "algorithm_sha256": run_contract.get("algorithm_sha256"),
        "base_model": MODEL_ID,
        "base_model_revision": MODEL_REVISION,
        "bm25_index_revision": BM25_INDEX_REVISION,
        "dataset_revision": DATASET_REVISION,
        "hypothesis": status.get("hypothesis"),
        "num_rollout": NUM_ROLLOUT,
        "optimization_attempts": attempts,
        "search_r1_commit": SEARCH_R1_COMMIT,
        "selected_attempt": attempt_id,
        "slime_commit": SLIME_COMMIT,
        "training_command": command_record.get("argv"),
        "training_command_sha256": command_record.get("sha256"),
        "web_search": "disabled",
        "world_size": 8,
    }
    atomic_write_json(root / "provenance.json", provenance)
    selection = {"num_rollout": NUM_ROLLOUT, "selected_attempt": attempt_id, "selected_at": utc_now()}
    atomic_write_json(root / "selected-attempt.json", selection)
    append_jsonl(root / "experiments.jsonl", {"attempt_id": attempt_id, "event": "selected", "timestamp": utc_now()})
    return selection


def complete_control(root: Path) -> dict:
    selection = read_json(root / "selected-attempt.json")
    for status_path in (root / "attempts").glob("*/status.json"):
        if read_json(status_path).get("status") in {"queued", "running"}:
            raise ValueError("cannot complete control while an attempt is active")
    payload = {"completed_at": utc_now(), "selected_attempt": selection["selected_attempt"], "status": "complete"}
    atomic_write_json(root / "control-complete.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("submit", "status", "select", "_worker"):
        command = subparsers.add_parser(name)
        command.add_argument("--attempt-id", required=True)
        command.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        if name == "submit":
            command.add_argument("--hypothesis", required=True)
    complete = subparsers.add_parser("complete-control")
    complete.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    if args.command == "submit":
        result = submit_attempt(args.output_root, args.attempt_id, args.hypothesis)
    elif args.command == "status":
        result = read_json(_attempt_dir(args.output_root, args.attempt_id) / "status.json")
    elif args.command == "select":
        result = select_attempt(args.output_root, args.attempt_id)
    elif args.command == "complete-control":
        result = complete_control(args.output_root)
    else:
        return _worker(args.output_root, args.attempt_id)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
