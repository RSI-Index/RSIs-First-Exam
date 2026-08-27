#!/usr/bin/env python3
"""Recompute optimizer matched-cost reward and verify final checkpoint reloads."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any


SCALES = {
    "E0": {"iterations": 44_317, "parameters": 550_337_664},
    "E1": {"iterations": 55_125, "parameters": 837_007_744},
    "E2": {"iterations": 38_014, "parameters": 998_036_992},
    "E3": {"iterations": 40_283, "parameters": 1_384_584_448},
    "E4": {"iterations": 56_477, "parameters": 1_934_716_160},
    "E5": {"iterations": 35_510, "parameters": 2_544_614_912},
}
PARAMETER_PATTERN = re.compile(
    r"number of parameters on \(tensor, pipeline\) model parallel rank \(0, 0\):\s*([0-9]+)"
)


def load_task_module():
    path = Path("/task-tools/optimizer_scaling_task.py")
    if not path.is_file():
        path = Path(__file__).resolve().parents[1] / "environment/task-tools/optimizer_scaling_task.py"
    spec = importlib.util.spec_from_file_location("trusted_optimizer_scaling_task", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"invalid {label}: {result}")
    return result


def reload_metrics(scale: str, run_dir: Path) -> dict[str, float | int]:
    contract = SCALES[scale]
    step = f"step_{contract['iterations']:08d}"
    paloma = json.loads((run_dir / "eval_harness" / step / "results.json").read_text())[
        "paloma_aggregate"
    ]
    if int(paloma["subsets"]) != 16:
        raise ValueError(f"{scale} verifier Paloma did not evaluate all 16 subsets")
    matches = PARAMETER_PATTERN.findall((run_dir / "run.log").read_text(errors="replace"))
    if not matches or max(int(value) for value in matches) != contract["parameters"]:
        raise ValueError(f"{scale} verifier parameter count differs from the locked model")
    return {
        "paloma_bits_per_byte": finite(paloma["bits_per_byte"], f"{scale} micro BPB"),
        "paloma_macro_bits_per_byte": finite(paloma["macro_bits_per_byte"], f"{scale} macro BPB"),
        "parameters": contract["parameters"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/app/output"))
    parser.add_argument("--baselines", type=Path, default=Path("/task-data/optimizer_adamh_baselines.json"))
    parser.add_argument("--policy-report", type=Path, required=True)
    parser.add_argument("--logs-dir", type=Path, default=Path("/logs/verifier"))
    args = parser.parse_args()
    baseline_payload = json.loads(args.baselines.read_text())
    if baseline_payload.get("version") != 2 or set(baseline_payload.get("scales", {})) != set(SCALES):
        raise ValueError("invalid six-scale optimizer baseline manifest")
    policy = json.loads(args.policy_report.read_text())
    submission = json.loads((args.output_root / "submission.json").read_text())
    task = load_task_module()
    evaluations: dict[str, dict[str, Any]] = {}
    reloads: dict[str, dict[str, float | int]] = {}
    selected_attempts: dict[str, Path] = {}
    for scale in SCALES:
        item = submission["scales"][scale]
        attempt = (args.output_root / Path(item["checkpoint_dir"]).parent).resolve()
        selected_attempts[scale] = attempt
        evaluations[scale] = task.evaluate_matched_cost_scale(
            scale, attempt, baseline_payload["scales"][scale]
        )
        reloads[scale] = reload_metrics(scale, args.runs_root / scale)
        for key in ("paloma_bits_per_byte", "paloma_macro_bits_per_byte"):
            if not math.isclose(
                float(reloads[scale][key]), float(item["metrics"][key]), rel_tol=1e-8, abs_tol=1e-10
            ):
                raise ValueError(f"{scale} independently reloaded {key} differs from training evidence")

    scores = [float(evaluations[scale]["score"]) for scale in SCALES]
    reward = math.exp(sum(scores) / len(scores))
    macro_log_gains = [
        math.log(item["matched"][-1]["baseline_macro_bpb"] / item["matched"][-1]["candidate_macro_bpb"])
        for item in evaluations.values()
    ]
    loss_log_gains = [
        math.log(item["baseline_loss_auc"] / item["candidate_loss_auc"])
        for item in evaluations.values()
    ]
    macro_gain = math.exp(sum(macro_log_gains) / len(macro_log_gains))
    loss_auc_gain = math.exp(sum(loss_log_gains) / len(loss_log_gains))
    ablation = task.validate_mechanism_ablation(
        args.output_root.resolve(), selected_attempts, baseline_payload["scales"]
    )
    budget = task.research_budget_report(
        args.output_root.resolve(), baseline_payload["scales"]
    )
    changed_paths = submission.get("changed_paths", [])
    optimizer_source_changed = isinstance(changed_paths, list) and any(
        isinstance(path, str) and (
            path.startswith("examples/training/optimizer/runtime/optimizer_")
            or path.startswith("3rdparty/Megatron-LM/megatron/core/optimizer/")
            or path.startswith("3rdparty/Megatron-LM/megatron/core/fusions/")
            or path.startswith("3rdparty/Megatron-LM/megatron/core/extensions/")
        )
        for path in changed_paths
    )
    guards = {
        "aggregate_paloma_gain": reward >= 1.002,
        "improved_scale_count": sum(score > 0 for score in scores) >= 4,
        "large_scale_transfer": math.exp((scores[4] + scores[5]) / 2) > 1.0,
        "endpoint_micro_noninferiority": all(
            item["matched"][-1]["candidate_micro_bpb"] <= 1.005 * item["matched"][-1]["baseline_micro_bpb"]
            for item in evaluations.values()
        ),
        "intermediate_micro_noninferiority": all(
            point["candidate_micro_bpb"] <= 1.01 * point["baseline_micro_bpb"]
            for item in evaluations.values() for point in item["matched"][:2]
        ),
        "aggregate_macro_noninferiority": macro_gain >= 1.0,
        "per_scale_macro_noninferiority": all(
            item["matched"][-1]["candidate_macro_bpb"] <= 1.01 * item["matched"][-1]["baseline_macro_bpb"]
            for item in evaluations.values()
        ),
        "aggregate_loss_auc_improvement": loss_auc_gain > 1.0,
        "improved_loss_auc_scale_count": sum(
            item["candidate_loss_auc"] < item["baseline_loss_auc"] for item in evaluations.values()
        ) >= 4,
        "mechanism_ablation": bool(ablation["passes"]),
        "research_budget": bool(budget["compliant"]),
        "optimizer_source_changed": optimizer_source_changed,
    }
    policy_pass = policy.get("policy_gate") == 1
    passed = policy_pass and all(guards.values())
    trusted_reward = reward if passed else 0.0
    if passed and not math.isclose(reward, float(submission["reward"]), rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("trusted reward differs from staged reward")
    output = {
        "reward": trusted_reward,
        "policy_gate": policy_pass,
        "quality_guards": guards,
        "aggregate_macro_gain": macro_gain,
        "aggregate_loss_auc_gain": loss_auc_gain,
        "mechanism_ablation": ablation,
        "budget_report": budget,
        "scales": evaluations,
        "reloads": reloads,
    }
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    (args.logs_dir / "metrics.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    (args.logs_dir / "reward.json").write_text(json.dumps({
        "reward": round(trusted_reward, 6),
        "policy_gate": int(policy_pass),
        "all_quality_gates": int(all(guards.values())),
    }, indent=2, sort_keys=True) + "\n")
    (args.logs_dir / "reward.txt").write_text(f"{trusted_reward:.6f}\n")


if __name__ == "__main__":
    main()
