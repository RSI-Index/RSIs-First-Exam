#!/usr/bin/env python3
"""Validate the complete staged six-rung artifact and emit checkpoint paths."""

from __future__ import annotations

import argparse
import json
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


def load_baselines(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text())
    if payload.get("version") != 1 or set(payload.get("scales", {})) != set(SCALES):
        raise ValueError("invalid six-scale baseline manifest")
    return payload["scales"]


def validate_scale(
    root: Path,
    scale: str,
    item: dict[str, Any],
    baseline: dict[str, float],
    source_hash: str,
) -> dict[str, str]:
    contract = SCALES[scale]
    checkpoint = safe_artifact_path(root, item.get("checkpoint_dir"), f"{scale} checkpoint")
    harness = safe_artifact_path(
        root, item.get("training_eval_results"), f"{scale} training_eval_results"
    )
    longppl = safe_artifact_path(root, item.get("longppl_results"), f"{scale} longppl_results")
    tracker = checkpoint / "latest_checkpointed_iteration.txt"
    if int(tracker.read_text().strip()) != contract["iterations"]:
        raise ValueError(f"{scale} checkpoint is not at its exact final update")
    if not harness.is_file() or not longppl.is_file():
        raise ValueError(f"{scale} is missing final evaluator evidence")
    attempt = checkpoint.parent
    run_contract = json.loads((attempt / "run_contract.json").read_text())
    if run_contract.get("scale") != scale or run_contract.get("source_inventory_sha256") != source_hash:
        raise ValueError(f"{scale} run contract does not match staged source")
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
    harness_metrics = json.loads(harness.read_text())["paloma_aggregate"]
    long_metrics = json.loads(longppl.read_text())["results"]
    measured = {
        "paloma_bits_per_byte": float(harness_metrics["bits_per_byte"]),
        "paloma_macro_bits_per_byte": float(harness_metrics["macro_bits_per_byte"]),
        "longppl": float(long_metrics["longppl"]),
    }
    staged_metrics = item.get("metrics", {})
    if any(float(staged_metrics.get(key, float("inf"))) != value for key, value in measured.items()):
        raise ValueError(f"{scale} staged metrics differ from raw evaluator evidence")
    if measured["longppl"] >= float(baseline["longppl"]):
        raise ValueError(f"{scale} does not strictly improve LongPPL")
    if measured["paloma_bits_per_byte"] > float(baseline["paloma_bits_per_byte"]) * 1.01:
        raise ValueError(f"{scale} fails the Paloma micro gate")
    if measured["paloma_macro_bits_per_byte"] > float(baseline["paloma_macro_bits_per_byte"]) * 1.01:
        raise ValueError(f"{scale} fails the Paloma macro gate")
    return {
        "checkpoint": str(checkpoint),
        "attempt": str(attempt),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("/app/output"))
    parser.add_argument(
        "--baselines",
        type=Path,
        default=Path("/task-data/positional_adamh_baselines.json"),
    )
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    root = args.output_root.resolve()
    submission = json.loads((root / "submission.json").read_text())
    if submission.get("version") != 2:
        raise ValueError("unsupported submission version")
    if submission.get("scale_order") != list(SCALES):
        raise ValueError("submission scale order is incomplete")
    source_hash = submission.get("source_inventory_sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise ValueError("invalid source inventory hash")
    hypothesis = safe_artifact_path(root, submission.get("hypothesis_file"), "hypothesis_file")
    source_snapshot = safe_artifact_path(root, submission.get("source_snapshot"), "source_snapshot")
    if not hypothesis.is_file() or not hypothesis.read_text().strip():
        raise ValueError("HYPOTHESIS.md is missing or empty")
    if not source_snapshot.is_file() or source_snapshot.is_symlink():
        raise ValueError("source snapshot is missing or unsafe")
    baselines = load_baselines(args.baselines)
    raw_scales = submission.get("scales")
    if not isinstance(raw_scales, dict) or set(raw_scales) != set(SCALES):
        raise ValueError("submission must contain exactly E0-E5")
    paths = {
        scale: validate_scale(root, scale, raw_scales[scale], baselines[scale], source_hash)
        for scale in SCALES
    }
    payload = {"version": 1, "source_inventory_sha256": source_hash, "scales": paths}
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
