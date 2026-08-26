#!/usr/bin/env python3
"""Fast contract tests for the candidate-visible parameterization harness."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


TASK_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = TASK_ROOT / "environment/task-tools/parameterization_scaling_task.py"
SPEC = importlib.util.spec_from_file_location("parameterization_scaling_task_test", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
ladder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ladder
SPEC.loader.exec_module(ladder)


class ParameterizationTransferContractTest(unittest.TestCase):
    def test_resume_accepts_same_32_node_allocation_as_initial_run(self) -> None:
        validator = TASK_ROOT / "cluster/validate_outer_allocation.py"
        self.assertTrue(validator.is_file(), "allocation validator is missing")
        allocation = " ".join(f"node-{index:02d} 8" for index in range(32))
        result = subprocess.run(
            [sys.executable, str(validator), allocation],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "outer allocation: 32 nodes x 8 GPUs = 256 GPUs")

    def test_visible_boundary_and_budget(self) -> None:
        self.assertEqual(ladder.SCALE_ORDER, ("E0", "E1", "E2", "E3", "E4"))
        self.assertEqual(tuple(ladder.SCALE_CONTRACTS), ladder.SCALE_ORDER)
        self.assertNotIn("E5", ladder.SCALE_CONTRACTS)
        self.assertEqual(ladder.VISIBLE_FLOP_BUDGET, 2.4e20)
        self.assertEqual(ladder.SCALE_FULL_FLOPS["E0"], 9.0e18)
        self.assertEqual(ladder.SCALE_FULL_FLOPS["E4"], 1.8e20)

    def test_baseline_manifest_is_read_as_visible_subset(self) -> None:
        result = ladder.load_baselines(TASK_ROOT / "baselines/scale_baselines.json")
        self.assertEqual(tuple(result), ladder.SCALE_ORDER)
        self.assertNotIn("E5", result)

    def test_starter_recipe_materializes_all_known_shapes(self) -> None:
        project = TASK_ROOT / "environment/project-overlay"
        spec, checks = ladder.materialize_transfer_recipe(project)
        self.assertTrue(checks["passes"])
        self.assertEqual(spec["visible_scales"], list(ladder.SCALE_ORDER))
        self.assertEqual(spec["held_out_scale"], "E5")
        self.assertEqual(set(spec["materialized_recipe"]), {*ladder.SCALE_ORDER, "E5"})
        base_lrs = {row["base_lr"] for row in spec["materialized_recipe"].values()}
        adam_lrs = {row["auxiliary_adam_lr"] for row in spec["materialized_recipe"].values()}
        self.assertEqual(len(base_lrs), 1)
        self.assertEqual(adam_lrs, {0.000304311605183897})

    def test_exact_scale_lookup_is_rejected(self) -> None:
        overlay = TASK_ROOT / "environment/project-overlay"
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "hidden_target.json").write_bytes((overlay / "hidden_target.json").read_bytes())
            (project / "parameterization_transfer.py").write_text(
                "class R:\n"
                " base_lr=0.1\n"
                " def init_scale(self,*a): return {'std': 0.1}\n"
                " def forward_scale(self,*a): return 1.0\n"
                " def lr_scale(self,*a): return 2.0 if 'E5' else 1.0\n"
                "def build_recipe(): return R()\n"
            )
            with self.assertRaisesRegex(SystemExit, "scale-general boundary"):
                ladder.materialize_transfer_recipe(project)

    def test_partial_attempts_charge_task_wide_flops(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            attempt = output / "attempts" / "pt-001"
            attempt.mkdir(parents=True)
            (attempt / "run_contract.json").write_text(json.dumps({
                "attempt_id": "pt-001", "scale": "E0", "source_inventory_sha256": "0" * 64,
            }))
            (attempt / "status.json").write_text(json.dumps({"status": "early_stopped"}))
            (attempt / "parameterization_cost_trace.jsonl").write_text(json.dumps({
                "iteration": 1, "cumulative_gpu_seconds": 8.0, "lm_loss": 5.0, "skipped": 0,
            }) + "\n")
            baselines = ladder.load_baselines(TASK_ROOT / "baselines/scale_baselines.json")
            report = ladder.research_budget_report(output, baselines)
            expected = 9.0e18 / 44_317
            self.assertAlmostEqual(report["charged_training_flops"], expected)
            self.assertTrue(report["compliant"])

    def test_declared_task_tools_exist(self) -> None:
        root = TASK_ROOT / "environment/task-tools"
        for name in (
            "run_parameterization_experiment",
            "summarize_experiment",
            "freeze_parameterization",
            "stage",
        ):
            self.assertTrue((root / name).is_file(), name)
        runner = (root / "run_parameterization_scaling_ladder.sh").read_text()
        self.assertNotIn("--eval-only", runner)
        self.assertIn("^E[0-4]$", runner)

    def test_runtime_uses_root_recipe_for_all_three_hook_families(self) -> None:
        runtime = (
            TASK_ROOT
            / "environment/project-overlay/examples/training/parameterization/runtime/parameterization_candidate.py"
        ).read_text()
        self.assertIn("initializer_methods", runtime)
        self.assertIn("recipe.forward_scale", runtime)
        self.assertIn("recipe.lr_scale", runtime)
        self.assertIn("marin_adamh.ADAMH_LR", runtime)


if __name__ == "__main__":
    unittest.main()
