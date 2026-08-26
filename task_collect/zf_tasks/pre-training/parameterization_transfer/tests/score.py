#!/usr/bin/env python3
"""Score the candidate-visible validity layer without fabricating transfer reward."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def load_task_module():
    path = Path("/task-tools/parameterization_scaling_task.py")
    spec = importlib.util.spec_from_file_location("trusted_parameterization_scaling_task", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("/app/output"))
    parser.add_argument("--policy-report", type=Path, required=True)
    parser.add_argument("--logs-dir", type=Path, default=Path("/logs/verifier"))
    args = parser.parse_args()
    policy = json.loads(args.policy_report.read_text())
    submission = json.loads((args.output_root / "submission.json").read_text())
    task = load_task_module()
    materialized, checks = task.materialize_transfer_recipe(Path("/app/project"))
    staged = json.loads((args.output_root / "PARAMETERIZATION_SPEC.json").read_text())
    recipe_reproduces = materialized == staged and checks.get("passes") is True
    budget = task.research_budget_report(
        args.output_root.resolve(), task.load_baselines(Path("/task-data/scale_baselines.json"))
    )
    guards = {
        "source_policy": policy.get("policy_gate") == 1,
        "recipe_reproduces": recipe_reproduces,
        "visible_budget": budget.get("compliant") is True,
        "proxy_and_confirmation": (
            "E0" in submission.get("selected_visible_scales", [])
            and any(scale != "E0" for scale in submission.get("selected_visible_scales", []))
        ),
        "held_out_pending": submission.get("held_out_evaluation") == {
            "scale": "E5", "status": "pending_trusted_evaluator"
        },
    }
    passed = all(guards.values())
    reward = 1.0 if passed else 0.0
    output = {
        "reward": reward,
        "policy_gate": bool(guards["source_policy"]),
        "quality_level": "valid_visible_freeze" if passed else "invalid",
        "transfer_success_scored": False,
        "held_out_evaluation": "pending_trusted_evaluator",
        "guards": guards,
        "budget_report": budget,
    }
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    (args.logs_dir / "metrics.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    (args.logs_dir / "reward.json").write_text(json.dumps({
        "reward": reward,
        "policy_gate": int(guards["source_policy"]),
        "valid_visible_freeze": int(passed),
        "transfer_success_scored": 0,
    }, indent=2, sort_keys=True) + "\n")
    (args.logs_dir / "reward.txt").write_text(f"{reward:.6f}\n")
    if not passed:
        raise SystemExit("visible-freeze score guards failed")


if __name__ == "__main__":
    main()
