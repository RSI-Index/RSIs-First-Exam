#!/usr/bin/env python3
"""Recompute LR-schedule reward and verify exact-final checkpoint reloads."""

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
    path = Path("/task-tools/lr_schedule_task.py")
    if not path.is_file():
        path = Path(__file__).resolve().parents[1] / "environment/task-tools/lr_schedule_task.py"
    spec = importlib.util.spec_from_file_location("trusted_lr_schedule_task", path)
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
    aggregate = json.loads(
        (run_dir / "eval_harness" / step / "results.json").read_text()
    )["paloma_aggregate"]
    if int(aggregate["subsets"]) != 16:
        raise ValueError(f"{scale} verifier Paloma did not evaluate all 16 subsets")
    matches = PARAMETER_PATTERN.findall((run_dir / "run.log").read_text(errors="replace"))
    if not matches or max(int(value) for value in matches) != contract["parameters"]:
        raise ValueError(f"{scale} verifier parameter count differs from the locked model")
    return {
        "bits_per_byte": finite(aggregate["bits_per_byte"], f"{scale} micro BPB"),
        "macro_bits_per_byte": finite(
            aggregate["macro_bits_per_byte"], f"{scale} macro BPB"
        ),
        "parameters": contract["parameters"],
    }


def candidate_evidence(
    output_root: Path,
    submission: dict[str, Any],
    baselines: dict[str, dict[str, Any]],
    reloads: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for scale, contract in SCALES.items():
        item = submission["scales"][scale]
        attempt = (output_root / Path(item["checkpoint_dir"]).parent).resolve()
        points: list[dict[str, float | int]] = []
        for baseline_point in baselines[scale]["paloma_trajectory"]:
            iteration = int(baseline_point["iteration"])
            result = json.loads(
                (
                    attempt
                    / "eval_harness"
                    / f"step_{iteration:08d}"
                    / "results.json"
                ).read_text()
            )["paloma_aggregate"]
            point = {
                "iteration": iteration,
                "bits_per_byte": finite(result["bits_per_byte"], f"{scale} micro BPB"),
                "macro_bits_per_byte": finite(
                    result["macro_bits_per_byte"], f"{scale} macro BPB"
                ),
            }
            if iteration == contract["iterations"]:
                for key in ("bits_per_byte", "macro_bits_per_byte"):
                    if not math.isclose(
                        float(point[key]),
                        float(reloads[scale][key]),
                        rel_tol=1e-8,
                        abs_tol=1e-10,
                    ):
                        raise ValueError(
                            f"{scale} independently reloaded {key} differs from training evidence"
                        )
            points.append(point)
        evidence[scale] = {
            "final_update": contract["iterations"],
            "scoring_gpu_seconds": float(item["metrics"]["final_gpu_seconds"]),
            "source_inventory_sha256": submission["source_inventory_sha256"],
            "paloma_trajectory": points,
        }
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/app/output"))
    parser.add_argument("--baselines", type=Path, default=Path("/task-data/lr_schedule_baselines.json"))
    parser.add_argument("--policy-report", type=Path, required=True)
    parser.add_argument("--logs-dir", type=Path, default=Path("/logs/verifier"))
    args = parser.parse_args()

    task = load_task_module()
    baseline_payload = json.loads(args.baselines.read_text())
    baselines = task.validate_baseline_manifest(baseline_payload)
    policy = json.loads(args.policy_report.read_text())
    submission = json.loads((args.output_root / "submission.json").read_text())
    reloads = {
        scale: reload_metrics(scale, args.runs_root / scale) for scale in SCALES
    }
    candidates = candidate_evidence(args.output_root, submission, baselines, reloads)
    scoring = task.score_ladder(baselines, candidates)
    budget = task.research_budget_report(args.output_root.resolve(), baselines)
    policy_pass = policy.get("policy_gate") == 1
    source_changed = policy.get("changed_paths") == [
        "examples/training/lr_schedule/runtime/lr_schedule_candidate.py"
    ]
    guards = {
        **scoring["guards"],
        "research_budget": bool(budget["compliant"]),
        "policy_gate": policy_pass,
        "candidate_source_changed": source_changed,
    }
    passed = all(guards.values())
    reward = float(scoring["reward"]) if passed else 0.0
    if passed and not math.isclose(
        reward, float(submission["reward"]), rel_tol=1e-10, abs_tol=1e-12
    ):
        raise ValueError("trusted reward differs from staged reward")
    output = {
        "reward": reward,
        "policy_gate": policy_pass,
        "quality_guards": guards,
        "budget_report": budget,
        "scoring": scoring,
        "reloads": reloads,
    }
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    (args.logs_dir / "metrics.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    (args.logs_dir / "reward.json").write_text(
        json.dumps(
            {
                "reward": round(reward, 6),
                "policy_gate": int(policy_pass),
                "all_quality_gates": int(all(guards.values())),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (args.logs_dir / "reward.txt").write_text(f"{reward:.6f}\n")


if __name__ == "__main__":
    main()
