#!/usr/bin/env python3
"""Prepare, finish, summarize, and stage trusted TMAX attempts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tmax_contract import (
    FINAL_OPTIMIZER_UPDATES,
    FINAL_TRAINING_TRAJECTORIES,
    SCIENTIFIC_HASH_FIELDS,
    TRAJECTORIES_PER_UPDATE,
    append_jsonl,
    atomic_write_json,
    build_input_manifest,
    sha256_file,
    tree_digest,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _copy_scientific_tree(source: Path, destination: Path, expected_hash: str) -> None:
    if destination.exists():
        if tree_digest(destination) != expected_hash:
            raise ValueError(f"existing source snapshot hash mismatch: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp.{os.getpid()}"
    shutil.copytree(
        source,
        temporary,
        symlinks=False,
        ignore=shutil.ignore_patterns(
            ".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "wandb", "*.pyc"
        ),
    )
    if tree_digest(temporary) != expected_hash:
        shutil.rmtree(temporary)
        raise ValueError("candidate source changed while it was being snapshotted")
    os.replace(temporary, destination)


def _copy_scientific_input(source: Path, destination: Path, expected_hash: str) -> None:
    """Snapshot one file or tree and prove the copy matches its contract hash."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, symlinks=False)
    else:
        shutil.copy2(source, destination)
    if tree_digest(destination) != expected_hash:
        raise ValueError(f"scientific input changed while it was being snapshotted: {source}")


def prepare_attempt(
    *,
    output_root: Path,
    attempt_id: str,
    max_updates: int | str,
    hypothesis_file: Path,
    project: Path,
    training_data: Path,
    training_prompt: Path,
    training_reward: Path,
    rl_recipe: Path,
) -> Path:
    if not hypothesis_file.is_file() or not hypothesis_file.read_text().strip():
        raise ValueError("hypothesis_file must be a nonempty regular file")
    manifest = build_input_manifest(
        attempt_id=attempt_id,
        max_updates=max_updates,
        source=project,
        training_data=training_data,
        training_prompt=training_prompt,
        training_reward=training_reward,
        rl_recipe=rl_recipe,
    )
    attempt = output_root / "attempts" / manifest.attempt_id
    try:
        attempt.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ValueError(f"attempt already exists: {attempt}") from error

    source_snapshot = output_root / "source-worktrees" / manifest.source_sha256 / "project"
    try:
        _copy_scientific_tree(project, source_snapshot, manifest.source_sha256)
        training_snapshot = attempt / "training-inputs"
        data_snapshot = training_snapshot / "data"
        prompt_snapshot = training_snapshot / "prompt" / "system.txt"
        reward_snapshot = training_snapshot / "reward" / "reward.py"
        recipe_snapshot = training_snapshot / "recipe.json"
        _copy_scientific_input(training_data, data_snapshot, manifest.training_data_sha256)
        _copy_scientific_input(training_prompt, prompt_snapshot, manifest.training_prompt_sha256)
        _copy_scientific_input(training_reward, reward_snapshot, manifest.training_reward_sha256)
        _copy_scientific_input(rl_recipe, recipe_snapshot, manifest.rl_recipe_sha256)
        shutil.copy2(hypothesis_file, attempt / "HYPOTHESIS.md")
        contract = {
            **manifest.to_dict(),
            "prepared_at": utc_now(),
            "status": "prepared",
            "gpu_count": 64,
            "learner_gpus": 16,
            "inference_gpus": 48,
            "trajectories_per_update": TRAJECTORIES_PER_UPDATE,
            "requested_training_trajectories": manifest.max_updates * TRAJECTORIES_PER_UPDATE,
            "source_snapshot": str(source_snapshot),
            "training_artifact_snapshot": str(training_snapshot),
            "training_data_snapshot": str(data_snapshot),
            "training_prompt_snapshot": str(prompt_snapshot),
            "training_reward_snapshot": str(reward_snapshot),
            "rl_recipe_snapshot": str(recipe_snapshot),
        }
        atomic_write_json(attempt / "run_contract.json", contract)
        atomic_write_json(attempt / "status.json", {"attempt_id": manifest.attempt_id, "status": "prepared"})
    except Exception:
        shutil.rmtree(attempt)
        raise
    return attempt


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"missing or invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def finish_attempt(*, attempt: Path, exit_code: int, counters_file: Path) -> dict[str, Any]:
    contract = _read_json(attempt / "run_contract.json")
    counters = _read_json(counters_file)
    expected_updates = int(contract["max_updates"])
    expected_trajectories = expected_updates * TRAJECTORIES_PER_UPDATE
    completed = (
        exit_code == 0
        and counters.get("optimizer_updates") == expected_updates
        and counters.get("training_trajectories") == expected_trajectories
        and counters.get("gpu_count") == 64
    )
    status = "completed" if completed else "failed"
    row = {
        "attempt_id": contract["attempt_id"],
        "status": status,
        "exit_code": exit_code,
        "requested_optimizer_updates": expected_updates,
        "optimizer_updates": counters.get("optimizer_updates", 0),
        "scheduler_horizon_updates": contract["scheduler_horizon_updates"],
        "training_trajectories": counters.get("training_trajectories", 0),
        "gpu_count": counters.get("gpu_count", 0),
        "learner_gpus": contract["learner_gpus"],
        "inference_gpus": contract["inference_gpus"],
        "generated_tokens": counters.get("generated_tokens", 0),
        "sandbox_steps": counters.get("sandbox_steps", 0),
        "metrics": counters.get("metrics", {}),
        "finished_at": utc_now(),
        **{field: contract[field] for field in SCIENTIFIC_HASH_FIELDS},
    }
    atomic_write_json(attempt / "status.json", row)
    append_jsonl(attempt.parents[1] / "experiments.jsonl", row)
    return row


def summarize_attempt(attempt: Path) -> dict[str, Any]:
    status = _read_json(attempt / "status.json")
    contract = _read_json(attempt / "run_contract.json")
    return {**contract, **status}


def stop_attempt(attempt: Path, reason_file: Path) -> dict[str, Any]:
    reason = reason_file.read_text().strip() if reason_file.is_file() else ""
    if not reason:
        raise ValueError("early-stop reason file must be nonempty")
    control = _read_json(attempt / "control.json")
    pid = control.get("pid")
    if not isinstance(pid, int) or pid <= 1:
        raise ValueError("attempt control record has no valid PID")
    record = {"requested_at": utc_now(), "reason": reason, "pid": pid}
    atomic_write_json(attempt / "early_stop.json", record)
    os.killpg(pid, signal.SIGTERM)
    return record


def _model_files(model: Path) -> dict[str, str]:
    tree_digest(model)
    if not (model / "config.json").is_file():
        raise ValueError("model checkpoint is missing config.json")
    weights = list(model.glob("*.safetensors")) + list(model.glob("pytorch_model*.bin"))
    if not weights:
        raise ValueError("model checkpoint is missing weight files")
    return {
        path.relative_to(model).as_posix(): sha256_file(path)
        for path in sorted(model.rglob("*"))
        if path.is_file()
    }


def stage_attempt(attempt: Path, checkpoint: Path, output_root: Path) -> dict[str, Any]:
    contract = _read_json(attempt / "run_contract.json")
    if contract.get("max_updates") != FINAL_OPTIMIZER_UPDATES:
        raise ValueError("only an attempt requested for 200 updates may be staged")
    status = _read_json(attempt / "status.json")
    if (
        status.get("status") != "completed"
        or status.get("optimizer_updates") != FINAL_OPTIMIZER_UPDATES
        or status.get("training_trajectories") != FINAL_TRAINING_TRAJECTORIES
        or status.get("gpu_count") != 64
    ):
        raise ValueError("attempt did not complete the 200-update final contract")
    files = _model_files(checkpoint)
    submission = output_root / "submission"
    model_destination = submission / "model"
    if model_destination.exists():
        raise ValueError(f"submission model already exists: {model_destination}")
    submission.mkdir(parents=True, exist_ok=True)
    shutil.copytree(checkpoint, model_destination, symlinks=False)
    manifest = {
        "attempt_id": contract["attempt_id"],
        "created_at": utc_now(),
        "files": files,
    }
    atomic_write_json(submission / "model-manifest.json", manifest)
    selection = {
        "attempt_id": contract["attempt_id"],
        "status": "selected",
        "run_kind": "full_budget",
        "training_trajectories": FINAL_TRAINING_TRAJECTORIES,
        "optimizer_updates": FINAL_OPTIMIZER_UPDATES,
        "gpu_count": 64,
        "checkpoint": "/app/output/submission/model",
        "control_record_sha256": sha256_file(attempt / "run_contract.json"),
        **{field: contract[field] for field in SCIENTIFIC_HASH_FIELDS},
    }
    atomic_write_json(output_root / "submission-selection.json", selection)
    append_jsonl(output_root / "experiments.jsonl", {**status, **selection})
    return selection


def audit_output(output_root: Path) -> dict[str, Any]:
    selection = _read_json(output_root / "submission-selection.json")
    expected_selection = {
        "status": "selected",
        "run_kind": "full_budget",
        "training_trajectories": FINAL_TRAINING_TRAJECTORIES,
        "optimizer_updates": FINAL_OPTIMIZER_UPDATES,
        "gpu_count": 64,
        "checkpoint": "/app/output/submission/model",
    }
    mismatches = {
        name: {"expected": expected, "actual": selection.get(name)}
        for name, expected in expected_selection.items()
        if selection.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"selection contract mismatch: {mismatches}")

    attempt_id = selection.get("attempt_id")
    if not isinstance(attempt_id, str):
        raise ValueError("selection has no attempt_id")
    attempt = output_root / "attempts" / attempt_id
    contract = _read_json(attempt / "run_contract.json")
    status = _read_json(attempt / "status.json")
    if status.get("status") != "completed":
        raise ValueError("selected attempt is not completed")
    if selection.get("control_record_sha256") != sha256_file(attempt / "run_contract.json"):
        raise ValueError("selection control record hash mismatch")
    for field in SCIENTIFIC_HASH_FIELDS:
        if selection.get(field) != contract.get(field):
            raise ValueError(f"selection scientific input mismatch: {field}")

    model = output_root / "submission" / "model"
    manifest = _read_json(output_root / "submission" / "model-manifest.json")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("model manifest has no files")
    actual_files = {
        path.relative_to(model).as_posix(): sha256_file(path)
        for path in sorted(model.rglob("*"))
        if path.is_file()
    }
    if files != actual_files:
        raise ValueError("model manifest mismatch")

    try:
        ledger = [
            json.loads(line)
            for line in (output_root / "experiments.jsonl").read_text().splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("experiment ledger is missing or invalid") from error
    selected_rows = [row for row in ledger if row.get("status") == "selected"]
    if len(selected_rows) != 1 or selected_rows[0].get("attempt_id") != attempt_id:
        raise ValueError("experiment ledger must contain exactly one matching selected row")

    report = {
        "status": "passed",
        "attempt_id": attempt_id,
        "optimizer_updates": FINAL_OPTIMIZER_UPDATES,
        "training_trajectories": FINAL_TRAINING_TRAJECTORIES,
        "model_files": len(actual_files),
        "audited_at": utc_now(),
    }
    atomic_write_json(output_root / "audit.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output-root", type=Path, default=Path("/app/output"))
    prepare.add_argument("--attempt-id", required=True)
    prepare.add_argument("--max-updates", required=True)
    prepare.add_argument("--hypothesis-file", type=Path, required=True)
    prepare.add_argument("--project", type=Path, default=Path("/app/project"))
    prepare.add_argument("--training-data", type=Path, default=Path("/app/training/data"))
    prepare.add_argument("--training-prompt", type=Path, default=Path("/app/training/prompt/system.txt"))
    prepare.add_argument("--training-reward", type=Path, default=Path("/app/training/reward/reward.py"))
    prepare.add_argument("--rl-recipe", type=Path, default=Path("/app/training/recipe.json"))
    finish = subparsers.add_parser("finish")
    finish.add_argument("--attempt", type=Path, required=True)
    finish.add_argument("--exit-code", type=int, required=True)
    finish.add_argument("--counters-file", type=Path, required=True)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--attempt", type=Path, required=True)
    stop = subparsers.add_parser("stop")
    stop.add_argument("--attempt", type=Path, required=True)
    stop.add_argument("--reason-file", type=Path, required=True)
    stage = subparsers.add_parser("stage")
    stage.add_argument("--attempt", type=Path, required=True)
    stage.add_argument("--checkpoint", type=Path, required=True)
    stage.add_argument("--output-root", type=Path, default=Path("/app/output"))
    audit = subparsers.add_parser("audit")
    audit.add_argument("--output-root", type=Path, default=Path("/app/output"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        result: Any = prepare_attempt(
            output_root=args.output_root,
            attempt_id=args.attempt_id,
            max_updates=args.max_updates,
            hypothesis_file=args.hypothesis_file,
            project=args.project,
            training_data=args.training_data,
            training_prompt=args.training_prompt,
            training_reward=args.training_reward,
            rl_recipe=args.rl_recipe,
        )
        print(result)
    elif args.command == "finish":
        print(json.dumps(finish_attempt(attempt=args.attempt, exit_code=args.exit_code, counters_file=args.counters_file), indent=2, sort_keys=True))
    elif args.command == "summarize":
        print(json.dumps(summarize_attempt(args.attempt), indent=2, sort_keys=True))
    elif args.command == "stop":
        print(json.dumps(stop_attempt(args.attempt, args.reason_file), indent=2, sort_keys=True))
    elif args.command == "stage":
        print(json.dumps(stage_attempt(args.attempt, args.checkpoint, args.output_root), indent=2, sort_keys=True))
    elif args.command == "audit":
        print(json.dumps(audit_output(args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
