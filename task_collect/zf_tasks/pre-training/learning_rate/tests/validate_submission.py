#!/usr/bin/env python3
"""Validate a staged learning-rate ladder from raw trusted artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any


SCALES: dict[str, dict[str, int]] = {
    "E0": {"iterations": 44_317, "tokens": 2_904_358_912, "gbs": 16, "mbs": 2, "gpus": 8},
    "E1": {"iterations": 55_125, "tokens": 3_612_672_000, "gbs": 16, "mbs": 2, "gpus": 8},
    "E2": {"iterations": 38_014, "tokens": 4_982_571_008, "gbs": 32, "mbs": 4, "gpus": 8},
    "E3": {"iterations": 40_283, "tokens": 10_559_946_752, "gbs": 64, "mbs": 2, "gpus": 32},
    "E4": {"iterations": 56_477, "tokens": 14_805_106_688, "gbs": 64, "mbs": 2, "gpus": 32},
    "E5": {"iterations": 35_510, "tokens": 18_617_466_880, "gbs": 128, "mbs": 1, "gpus": 128},
}
ALLOWED_SOURCE = "examples/training/lr_schedule/runtime/lr_schedule_candidate.py"


def load_task_tools():
    candidates = [
        Path("/task-tools/lr_schedule_task.py"),
        Path(__file__).resolve().parents[1]
        / "environment"
        / "task-tools"
        / "lr_schedule_task.py",
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise ValueError("trusted lr_schedule_task.py is missing")
    spec = importlib.util.spec_from_file_location("trusted_lr_schedule_task", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load trusted task tool: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_artifact_path(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"{label} must be a non-empty relative path")
    root = root.resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"{label} escapes the artifact root")
    return path


def read_json_lines(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{label} line {line_number} is not an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{label} is empty")
    return rows


def validate_scale(
    root: Path,
    scale: str,
    item: dict[str, Any],
    source_hash: str,
    baseline: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    contract = SCALES[scale]
    checkpoint = safe_artifact_path(root, item.get("checkpoint_dir"), f"{scale} checkpoint")
    cost_path = safe_artifact_path(root, item.get("cost_trace"), f"{scale} cost trace")
    schedule_path = safe_artifact_path(
        root, item.get("schedule_trace"), f"{scale} schedule trace"
    )
    attempt = checkpoint.parent
    tracker = checkpoint / "latest_checkpointed_iteration.txt"
    if not tracker.is_file() or int(tracker.read_text().strip()) != contract["iterations"]:
        raise ValueError(f"{scale} checkpoint is not at its exact final update")

    run_contract = json.loads((attempt / "run_contract.json").read_text())
    status = json.loads((attempt / "status.json").read_text())
    if run_contract.get("scale") != scale or run_contract.get("source_inventory_sha256") != source_hash:
        raise ValueError(f"{scale} run contract does not match staged source")
    if status.get("status") != "completed" or status.get("run_valid") is not True:
        raise ValueError(f"{scale} is not a completed valid run")
    training = run_contract.get("contract", {}).get("training", {})
    expected_training = {
        "iterations": contract["iterations"],
        "tokens": contract["tokens"],
        "sequence_length": 4096,
        "global_batch_size": contract["gbs"],
        "micro_batch_size": contract["mbs"],
        "seed": 0,
        "gpus": contract["gpus"],
    }
    if training != expected_training:
        raise ValueError(f"{scale} training contract differs from the locked recipe")

    cost_rows = read_json_lines(cost_path, f"{scale} cost trace")
    if len(cost_rows) != contract["iterations"]:
        raise ValueError(f"{scale} cost trace is incomplete")
    previous_cost = 0.0
    for expected_iteration, row in enumerate(cost_rows, 1):
        cost = float(row.get("cumulative_gpu_seconds", float("nan")))
        if (
            int(row.get("iteration", -1)) != expected_iteration
            or not math.isfinite(cost)
            or cost <= previous_cost
            or int(row.get("skipped", 0) or 0) != 0
        ):
            raise ValueError(f"{scale} cost trace is invalid at update {expected_iteration}")
        previous_cost = cost

    schedule_rows = read_json_lines(schedule_path, f"{scale} schedule trace")
    if len(schedule_rows) != contract["iterations"]:
        raise ValueError(f"{scale} schedule trace is incomplete")
    for expected_iteration, row in enumerate(schedule_rows, 1):
        if int(row.get("iteration", -1)) != expected_iteration:
            raise ValueError(f"{scale} schedule trace update mismatch")
        if row.get("source_sha256") != source_hash:
            raise ValueError(f"{scale} schedule trace source mismatch")
        multiplier = float(row.get("multiplier", float("nan")))
        peak_lrs = [float(value) for value in row.get("peak_lrs", [])]
        effective_lrs = [float(value) for value in row.get("effective_lrs", [])]
        if (
            not math.isfinite(multiplier)
            or multiplier < 0.0
            or multiplier > 1.0
            or not peak_lrs
            or len(peak_lrs) != len(effective_lrs)
            or row.get("state") is not None
            or row.get("diagnostics") is not None
            or any(not math.isfinite(value) for value in (*peak_lrs, *effective_lrs))
            or any(
                not math.isclose(effective, peak * multiplier, rel_tol=1e-9, abs_tol=1e-15)
                for peak, effective in zip(peak_lrs, effective_lrs, strict=True)
            )
        ):
            raise ValueError(f"{scale} schedule trace has an invalid multiplier contract")

    expected_updates = [int(point["iteration"]) for point in baseline["paloma_trajectory"]]
    trajectory: list[dict[str, Any]] = []
    for iteration in expected_updates:
        result_path = attempt / "eval_harness" / f"step_{iteration:08d}" / "results.json"
        if not result_path.is_file():
            raise ValueError(f"{scale} is missing Paloma update {iteration}")
        paloma = json.loads(result_path.read_text()).get("paloma_aggregate", {})
        trajectory.append(
            {
                "iteration": iteration,
                "bits_per_byte": float(paloma["bits_per_byte"]),
                "macro_bits_per_byte": float(paloma["macro_bits_per_byte"]),
            }
        )
    measured = {
        "final_update": contract["iterations"],
        "scoring_gpu_seconds": float(cost_rows[-1]["cumulative_gpu_seconds"]),
        "source_inventory_sha256": source_hash,
        "paloma_trajectory": trajectory,
    }
    paths = {
        "attempt": str(attempt),
        "checkpoint": str(checkpoint),
        "cost_trace": str(cost_path),
        "schedule_trace": str(schedule_path),
    }
    return measured, paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("/app/output"))
    parser.add_argument("--baselines", type=Path, default=Path("/task-data/lr_schedule_baselines.json"))
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    root = args.output_root.resolve()
    task_tools = load_task_tools()
    baseline_payload = json.loads(args.baselines.read_text())
    if baseline_payload.get("version") != 3:
        raise ValueError("invalid learning-rate baseline manifest version")
    baselines = task_tools.validate_baseline_manifest(baseline_payload)

    submission = json.loads((root / "submission.json").read_text())
    if submission.get("version") != 3 or submission.get("scale_order") != list(SCALES):
        raise ValueError("unsupported or incomplete submission")
    if submission.get("submission_eligible") is not True:
        raise ValueError("submission is not marked eligible")
    source_hash = submission.get("source_inventory_sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise ValueError("invalid source inventory hash")
    changed_paths = submission.get("changed_paths")
    if not isinstance(changed_paths, list) or ALLOWED_SOURCE not in changed_paths:
        raise ValueError("submission does not change lr_schedule_candidate.py")
    if any(path != ALLOWED_SOURCE for path in changed_paths):
        raise ValueError("submission changes source outside lr_schedule_candidate.py")

    required_files = [
        safe_artifact_path(root, submission.get("hypothesis_file"), "hypothesis_file"),
        root / "LR_SCHEDULE.md",
        root / "experiments.jsonl",
        root / "COST_REPORT.json",
        root / "mechanism_ablation.json",
        root / "science_freeze.json",
        safe_artifact_path(root, submission.get("source_snapshot"), "source_snapshot"),
    ]
    if any(not path.is_file() or not path.read_text().strip() for path in required_files):
        raise ValueError("required schedule, hypothesis, ledger, cost, or source artifact is missing")
    budget = submission.get("budget_report", {})
    if budget.get("budget_multiplier") != 6.0 or budget.get("compliant") is not True:
        raise ValueError("submission does not carry a compliant six-control research budget")
    hypothesis_path = safe_artifact_path(
        root, submission.get("hypothesis_file"), "hypothesis_file"
    )
    task_tools.validate_hypothesis(hypothesis_path)
    hypothesis_sha256 = task_tools.source_sha256(hypothesis_path)
    if submission.get("hypothesis_sha256") != hypothesis_sha256:
        raise ValueError("staged hypothesis hash does not match its artifact")
    confirmation_freeze = json.loads((root / "science_freeze.json").read_text())
    if (
        confirmation_freeze.get("version") != 1
        or confirmation_freeze.get("development_scales")
        != list(task_tools.DEVELOPMENT_SCALES)
        or confirmation_freeze.get("confirmation_scales")
        != list(task_tools.CONFIRMATION_SCALES)
        or confirmation_freeze.get("source_inventory_sha256") != source_hash
        or confirmation_freeze.get("hypothesis_sha256") != hypothesis_sha256
        or submission.get("confirmation_freeze") != confirmation_freeze
    ):
        raise ValueError("confirmation_freeze does not bind the staged science claim")

    raw_scales = submission.get("scales")
    if not isinstance(raw_scales, dict) or set(raw_scales) != set(SCALES):
        raise ValueError("submission must contain exactly E0-E5")
    measured: dict[str, dict[str, Any]] = {}
    paths: dict[str, dict[str, str]] = {}
    for scale in SCALES:
        measured[scale], paths[scale] = validate_scale(
            root, scale, raw_scales[scale], source_hash, baselines[scale]
        )
    selected_attempts = {
        scale: Path(paths[scale]["attempt"]) for scale in SCALES
    }
    mechanism_ablation = task_tools.validate_mechanism_ablation(
        root, selected_attempts, baselines
    )
    if mechanism_ablation.get("passes") is not True:
        raise ValueError("mechanism_ablation does not support the staged mechanism")
    if submission.get("mechanism_ablation") != mechanism_ablation:
        raise ValueError("staged mechanism_ablation differs from verifier recomputation")
    scoring = task_tools.score_ladder(baselines, measured)
    if scoring.get("passes") is not True or float(scoring.get("reward", 0.0)) <= 1.0:
        raise ValueError(f"staged ladder fails scientific gates: {scoring.get('guards')}")
    if not math.isclose(
        float(submission.get("reward", 0.0)),
        float(scoring["reward"]),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("staged reward differs from verifier recomputation")

    payload = {
        "version": 3,
        "source_inventory_sha256": source_hash,
        "reward": scoring["reward"],
        "guards": scoring["guards"],
        "scales": paths,
    }
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
