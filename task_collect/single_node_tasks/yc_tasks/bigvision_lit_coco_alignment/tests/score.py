#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


class BaselineContractError(RuntimeError):
    pass


def compute_reward(metrics: dict, baseline: dict) -> dict:
    if baseline.get("status") != "complete":
        raise BaselineContractError("baseline_contract.json is not complete")
    if metrics.get("status") != "pass":
        raise ValueError("verifier metrics are not passing")
    required = ("i2t_r1", "t2i_r1", "imagenet_zero_shot_accuracy")
    values = {key: float(metrics[key]) for key in required}
    if not all(math.isfinite(value) and 0.0 <= value <= 100.0 for value in values.values()):
        raise ValueError("primary and retention metrics must be finite percentages in [0,100]")
    lower_bound = float(baseline["imagenet_retention_lower_bound"])
    if values["imagenet_zero_shot_accuracy"] < lower_bound:
        raise ValueError("ImageNet retention gate failed")
    primary = math.sqrt(values["i2t_r1"] * values["t2i_r1"])
    return {
        "reward": min(1.0, max(0.0, primary / 100.0)),
        "primary_metric": primary,
        "metric_name": "coco_bidirectional_r1_geometric_mean",
        "i2t_r1": values["i2t_r1"],
        "t2i_r1": values["t2i_r1"],
        "imagenet_zero_shot_accuracy": values["imagenet_zero_shot_accuracy"],
        "retention_lower_bound": lower_bound,
        "retention_gate": "pass",
        "reference_metric": float(baseline["reference_metric"]),
        "improvement": primary - float(baseline["reference_metric"]),
        "policy_gate": 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=Path("/logs/verifier/metrics.json"))
    parser.add_argument("--baseline", type=Path, default=Path("/tests/baseline_contract.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("/logs/verifier"))
    args = parser.parse_args()
    result = compute_reward(json.loads(args.metrics.read_text()), json.loads(args.baseline.read_text()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reward.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    (args.output_dir / "reward.txt").write_text(f"{result['reward']:.12f}\n")


if __name__ == "__main__":
    main()
