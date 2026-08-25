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
    components = metrics.get("component_metrics")
    if not isinstance(components, dict) or len(components) != 38:
        raise ValueError("exactly 38 component metrics are required")
    values = [float(value) for value in components.values()]
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise ValueError("component metrics must be finite values in [0,1]")
    primary = sum(values) / 38
    return {
        "reward": primary,
        "primary_metric": primary,
        "metric_name": "datacomp_38_task_unweighted_mean",
        "component_count": 38,
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
