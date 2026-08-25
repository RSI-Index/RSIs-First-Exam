#!/usr/bin/env python3
"""Validate a frozen visible parameterization without exposing held-out execution."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


SCALES: dict[str, dict[str, int | float]] = {
    "E0": {"iterations": 44_317, "tokens": 2_904_358_912, "gbs": 16, "mbs": 2, "gpus": 8, "flops": 9.0e18},
    "E1": {"iterations": 55_125, "tokens": 3_612_672_000, "gbs": 16, "mbs": 2, "gpus": 8, "flops": 1.8e19},
    "E2": {"iterations": 38_014, "tokens": 4_982_571_008, "gbs": 32, "mbs": 4, "gpus": 8, "flops": 3.0e19},
    "E3": {"iterations": 40_283, "tokens": 10_559_946_752, "gbs": 64, "mbs": 2, "gpus": 32, "flops": 9.0e19},
    "E4": {"iterations": 56_477, "tokens": 14_805_106_688, "gbs": 64, "mbs": 2, "gpus": 32, "flops": 1.8e20},
}
HASH = re.compile(r"[0-9a-f]{64}")


def safe_path(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"{label} must be a non-empty relative path")
    result = (root / relative).resolve()
    if root != result and root not in result.parents:
        raise ValueError(f"{label} escapes the artifact root")
    return result


def require_json_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("passes") is not True:
        raise ValueError(f"report does not pass: {path.name}")
    return payload


def validate_scale(root: Path, scale: str, item: dict[str, Any], source_hash: str) -> dict[str, str]:
    contract = SCALES[scale]
    checkpoint = safe_path(root, item.get("checkpoint_dir"), f"{scale} checkpoint")
    tracker = checkpoint / "latest_checkpointed_iteration.txt"
    if not tracker.is_file() or int(tracker.read_text().strip()) != contract["iterations"]:
        raise ValueError(f"{scale} checkpoint is not at its final update")
    attempt = checkpoint.parent
    run_contract = json.loads((attempt / "run_contract.json").read_text())
    status = json.loads((attempt / "status.json").read_text())
    if run_contract.get("scale") != scale or run_contract.get("source_inventory_sha256") != source_hash:
        raise ValueError(f"{scale} run contract differs from the frozen source")
    if status.get("status") != "completed" or status.get("run_valid") is not True:
        raise ValueError(f"{scale} is not a completed valid run")
    training = run_contract.get("contract", {}).get("training", {})
    expected = {
        "iterations": contract["iterations"],
        "tokens": contract["tokens"],
        "sequence_length": 4096,
        "global_batch_size": contract["gbs"],
        "micro_batch_size": contract["mbs"],
        "seed": 0,
        "gpus": contract["gpus"],
    }
    if training != expected:
        raise ValueError(f"{scale} training contract differs from the locked profile")
    trace = attempt / "parameterization_cost_trace.jsonl"
    rows = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    if len(rows) != contract["iterations"] or int(rows[-1]["iteration"]) != contract["iterations"]:
        raise ValueError(f"{scale} cost trace is incomplete")
    if not math.isfinite(float(rows[-1]["cumulative_gpu_seconds"])):
        raise ValueError(f"{scale} final cost is invalid")
    if item.get("full_run") is not True or float(item.get("training_flops", 0.0)) != contract["flops"]:
        raise ValueError(f"{scale} is not declared as the exact full visible run")
    return {"checkpoint": str(checkpoint), "attempt": str(attempt)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("/app/output"))
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    root = args.output_root.resolve()
    submission = json.loads((root / "submission.json").read_text())
    if submission.get("version") != 3 or submission.get("submission_type") != "frozen_visible_parameterization":
        raise ValueError("unsupported submission schema")
    if submission.get("submission_eligible") is not True or submission.get("quality_level") != "valid_visible_freeze":
        raise ValueError("submission is not an eligible visible freeze")
    if submission.get("candidate_claimed_reward") is not None:
        raise ValueError("candidate may not claim a held-out reward")
    held_out = submission.get("held_out_evaluation", {})
    if held_out != {"scale": "E5", "status": "pending_trusted_evaluator"}:
        raise ValueError("held-out result must remain pending at candidate freeze")
    source_hash = submission.get("source_inventory_sha256")
    if not isinstance(source_hash, str) or HASH.fullmatch(source_hash) is None:
        raise ValueError("invalid source inventory hash")
    if "parameterization_transfer.py" not in submission.get("changed_paths", []):
        raise ValueError("root parameterization recipe was not changed")

    required_text = ("HYPOTHESIS.md", "PARAMETERIZATION_RULE.md", "experiments.jsonl")
    if any(not (root / name).is_file() or not (root / name).read_text().strip() for name in required_text):
        raise ValueError("required hypothesis, rule, or experiment ledger is missing")
    required_json = (
        "PARAMETERIZATION_SPEC.json",
        "materialized_role_report.json",
        "coordinate_check_report.json",
        "POLICY_REPORT.json",
        "REPRODUCIBILITY_REPORT.json",
        "LEAKAGE_REPORT.json",
        "COST_REPORT.json",
        "FROZEN.json",
    )
    for name in required_json:
        if not (root / name).is_file():
            raise ValueError(f"required artifact is missing: {name}")
    for name in ("coordinate_check_report.json", "POLICY_REPORT.json", "REPRODUCIBILITY_REPORT.json", "LEAKAGE_REPORT.json"):
        require_json_report(root / name)
    spec = json.loads((root / "PARAMETERIZATION_SPEC.json").read_text())
    if not math.isfinite(float(spec.get("base_lr", 0.0))) or float(spec["base_lr"]) <= 0:
        raise ValueError("spec has no finite positive base LR")
    if float(spec["base_lr"]) != float(submission.get("base_lr", 0.0)):
        raise ValueError("submission and spec base LR differ")
    if spec.get("visible_scales") != list(SCALES) or spec.get("held_out_scale") != "E5":
        raise ValueError("spec scale boundary is invalid")
    if set(spec.get("materialized_recipe", {})) != {*SCALES, "E5"}:
        raise ValueError("recipe was not materialized across visible and sanitized target shapes")
    if any(float(row["base_lr"]) != float(spec["base_lr"]) for row in spec["materialized_recipe"].values()):
        raise ValueError("materialized recipe does not use one unchanged base LR")
    if any(float(row["auxiliary_adam_lr"]) != 0.000304311605183897 for row in spec["materialized_recipe"].values()):
        raise ValueError("materialized recipe changes the auxiliary Adam LR")

    budget = json.loads((root / "COST_REPORT.json").read_text())
    if budget.get("compliant") is not True or float(budget.get("charged_training_flops", float("inf"))) > 2.4e20:
        raise ValueError("visible FLOP budget is not compliant")
    if submission.get("budget_report") != budget:
        raise ValueError("submission budget report differs from durable report")
    selected = submission.get("selected_visible_scales")
    if not isinstance(selected, list) or selected != [scale for scale in SCALES if scale in selected]:
        raise ValueError("selected visible scales are invalid or unordered")
    if "E0" not in selected or not any(scale != "E0" for scale in selected):
        raise ValueError("freeze requires E0 and at least one larger visible confirmation")
    raw_scales = submission.get("scales")
    if not isinstance(raw_scales, dict) or set(raw_scales) != set(selected):
        raise ValueError("selected visible evidence differs from submission scales")
    paths = {scale: validate_scale(root, scale, raw_scales[scale], source_hash) for scale in selected}
    source_snapshot = safe_path(root, submission.get("source_snapshot"), "source snapshot")
    if not source_snapshot.is_file():
        raise ValueError("source snapshot manifest is missing")
    payload = {"version": 3, "source_inventory_sha256": source_hash, "scales": paths, "held_out_status": "pending"}
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
