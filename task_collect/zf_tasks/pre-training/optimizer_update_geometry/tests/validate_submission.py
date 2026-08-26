#!/usr/bin/env python3
"""Validate a staged optimizer ladder and emit trusted checkpoint paths."""

from __future__ import annotations

import argparse
import json
import math
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


def safe_artifact_path(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"{label} must be a non-empty relative path")
    root = root.resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"{label} escapes the artifact root")
    return path


def validate_scale(root: Path, scale: str, item: dict[str, Any], source_hash: str) -> dict[str, str]:
    contract = SCALES[scale]
    checkpoint = safe_artifact_path(root, item.get("checkpoint_dir"), f"{scale} checkpoint")
    harness = safe_artifact_path(root, item.get("training_eval_results"), f"{scale} Paloma")
    cost_trace = safe_artifact_path(root, item.get("cost_trace"), f"{scale} cost trace")
    tracker = checkpoint / "latest_checkpointed_iteration.txt"
    if not tracker.is_file() or int(tracker.read_text().strip()) != contract["iterations"]:
        raise ValueError(f"{scale} checkpoint is not at its exact final update")
    if not harness.is_file() or not cost_trace.is_file():
        raise ValueError(f"{scale} is missing final Paloma or cost evidence")
    attempt = checkpoint.parent
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
    last_trace = None
    count = 0
    for line in cost_trace.read_text().splitlines():
        if line.strip():
            last_trace = json.loads(line)
            count += 1
    if count != contract["iterations"] or int(last_trace["iteration"]) != contract["iterations"]:
        raise ValueError(f"{scale} cost trace is incomplete")
    if not math.isfinite(float(last_trace["cumulative_gpu_seconds"])):
        raise ValueError(f"{scale} final cost is invalid")
    paloma = json.loads(harness.read_text())["paloma_aggregate"]
    measured = {
        "paloma_bits_per_byte": float(paloma["bits_per_byte"]),
        "paloma_macro_bits_per_byte": float(paloma["macro_bits_per_byte"]),
    }
    staged = item.get("metrics", {})
    if any(float(staged.get(key, float("inf"))) != value for key, value in measured.items()):
        raise ValueError(f"{scale} staged final metrics differ from raw Paloma evidence")
    return {"checkpoint": str(checkpoint), "attempt": str(attempt)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("/app/output"))
    parser.add_argument("--baselines", type=Path, default=Path("/task-data/optimizer_adamh_baselines.json"))
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    root = args.output_root.resolve()
    baseline_payload = json.loads(args.baselines.read_text())
    if baseline_payload.get("version") != 2 or set(baseline_payload.get("scales", {})) != set(SCALES):
        raise ValueError("invalid six-scale optimizer baseline manifest")
    submission = json.loads((root / "submission.json").read_text())
    if submission.get("version") != 2 or submission.get("scale_order") != list(SCALES):
        raise ValueError("unsupported or incomplete submission")
    if submission.get("submission_eligible") is not True or float(submission.get("reward", 0.0)) < 1.002:
        raise ValueError("submission does not carry a passing aggregate result")
    if not all(bool(value) for value in submission.get("quality_guards", {}).values()):
        raise ValueError("submission contains a failed aggregate quality gate")
    source_hash = submission.get("source_inventory_sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise ValueError("invalid source inventory hash")
    required_files = [
        safe_artifact_path(root, submission.get("hypothesis_file"), "hypothesis_file"),
        root / "NOVELTY.md",
        root / "OPTIMIZER.md",
        root / "experiments.jsonl",
        root / "COST_REPORT.json",
        safe_artifact_path(
            root, submission.get("mechanism_ablation_file"), "mechanism_ablation_file"
        ),
        safe_artifact_path(root, submission.get("source_snapshot"), "source_snapshot"),
    ]
    if any(not path.is_file() or not path.read_text().strip() for path in required_files):
        raise ValueError("required hypothesis, novelty, optimizer, ledger, or source artifact is missing")
    changed_paths = submission.get("changed_paths")
    if not isinstance(changed_paths, list) or not any(
        isinstance(path, str) and (
            path.startswith("examples/training/optimizer/runtime/optimizer_")
            or path.startswith("3rdparty/Megatron-LM/megatron/core/optimizer/")
            or path.startswith("3rdparty/Megatron-LM/megatron/core/fusions/")
            or path.startswith("3rdparty/Megatron-LM/megatron/core/extensions/")
        )
        for path in changed_paths
    ):
        raise ValueError("submission does not contain a changed optimizer implementation")
    if submission.get("mechanism_ablation", {}).get("passes") is not True:
        raise ValueError("submission does not carry passing mechanism ablation evidence")
    if submission.get("budget_report", {}).get("compliant") is not True:
        raise ValueError("submission does not carry a compliant research budget report")
    raw_scales = submission.get("scales")
    if not isinstance(raw_scales, dict) or set(raw_scales) != set(SCALES):
        raise ValueError("submission must contain exactly E0-E5")
    paths = {
        scale: validate_scale(root, scale, raw_scales[scale], source_hash)
        for scale in SCALES
    }
    payload = {"version": 2, "source_inventory_sha256": source_hash, "scales": paths}
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
