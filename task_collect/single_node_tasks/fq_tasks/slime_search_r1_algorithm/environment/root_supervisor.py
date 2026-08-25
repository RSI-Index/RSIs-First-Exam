#!/usr/bin/env python3
"""Privileged, fixed-path supervisor that attests a completed Slime attempt."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from async_contract import read_json, validate_attempt_id
from launch_train import build_slime_command, run_attempt
from policy_contract import tree_sha256
from task_contract import (
    ALGORITHM_ROOT,
    FINAL_ROLLOUT_ID,
    NUM_ROLLOUT,
    algorithm_source_sha256,
    canonical_command_sha256,
    load_algorithm_config,
)


OUTPUT_ROOT = Path("/app/output")
ATTESTATION_ROOT = Path("/var/lib/slime-task/attestations")
TRUSTED_RUN_ROOT = Path("/var/lib/slime-task/runs")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_attestation(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _make_tree_readonly(root: Path) -> None:
    for entry in sorted(root.rglob("*"), reverse=True):
        if entry.is_symlink():
            raise RuntimeError("supervised output contains a symlink")
        os.chmod(entry, 0o555 if entry.is_dir() else 0o444)
    os.chmod(root, 0o555)


def supervise(attempt_id: str) -> dict:
    attempt_id = validate_attempt_id(attempt_id)
    if os.geteuid() != 0:
        raise PermissionError("root supervisor must run with effective uid 0")
    if os.stat("/proc/self/ns/net").st_ino == os.stat("/proc/1/ns/net").st_ino:
        raise RuntimeError("root supervisor requires a private network namespace")
    subprocess.run(["ip", "link", "set", "lo", "up"], check=True)
    attempt = OUTPUT_ROOT / "attempts" / attempt_id
    status = read_json(attempt / "status.json")
    if status.get("attempt_id") != attempt_id or status.get("status") != "running":
        raise ValueError("supervised attempt must be in running state")
    trusted_attempt = TRUSTED_RUN_ROOT / attempt_id
    if trusted_attempt.exists() or trusted_attempt.is_symlink():
        raise ValueError("trusted attempt id cannot be reused")
    trusted_attempt.mkdir(mode=0o700)
    train_output = trusted_attempt / "train"
    if train_output.exists() or train_output.is_symlink():
        raise ValueError("supervised train output must not already exist")

    config_path = Path(ALGORITHM_ROOT) / "algorithm_config.toml"
    config = load_algorithm_config(config_path)
    algorithm_hash = algorithm_source_sha256(ALGORITHM_ROOT)
    command = build_slime_command(config, train_output)
    command_hash = canonical_command_sha256(command)
    started_at = utc_now()
    run_attempt(config_path, train_output)

    if algorithm_source_sha256(ALGORITHM_ROOT) != algorithm_hash:
        raise RuntimeError("algorithm config changed during supervised training")
    run_contract = read_json(train_output / "run-contract.json")
    command_record = read_json(train_output / "training-command.json")
    if run_contract.get("num_rollout") != NUM_ROLLOUT or run_contract.get("world_size") != 8:
        raise RuntimeError("supervised launcher did not complete the frozen budget")
    if command_record.get("argv") != command or command_record.get("sha256") != command_hash:
        raise RuntimeError("supervised launcher command record differs")
    hf_model = train_output / "hf_model"
    hf_hash = tree_sha256(hf_model)
    _make_tree_readonly(train_output)
    os.chmod(trusted_attempt, 0o555)
    payload = {
        "algorithm_sha256": algorithm_hash,
        "attempt_id": attempt_id,
        "command_sha256": command_hash,
        "completed_at": utc_now(),
        "final_rollout_id": FINAL_ROLLOUT_ID,
        "hf_model_sha256": hf_hash,
        "num_rollout": NUM_ROLLOUT,
        "started_at": started_at,
        "status": "completed",
        "world_size": 8,
    }
    destination = ATTESTATION_ROOT / f"{attempt_id}.json"
    if destination.exists() or destination.is_symlink():
        raise ValueError("attempt attestation already exists")
    _atomic_attestation(destination, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "offline",
            "PYTHONPATH": "/task-tools:/opt/slime:/opt/Megatron-LM",
        }
    )
    payload = supervise(args.attempt_id)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
