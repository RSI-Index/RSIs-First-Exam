#!/usr/bin/env python3
"""Regression tests for portable RSI-Harness Blue Vela preflight planning."""

from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
SCRIPT = Path(__file__).with_name("preflight_task.py")
HARNESS = REPO / "RSI-Harness"
REPRESENTATIVE_TASK = (
    Path(__file__).resolve().parent
    / "fixtures/representative-task"
)
PROFILE = HARNESS / "src/rsi_harness/cluster/bluevela/profile.toml"
SKILL = Path(__file__).resolve().parents[1] / "SKILL.md"
FALLBACK_EVIDENCE = json.dumps(
    {
        "normal_job_id": "normal-123",
        "declared_wait_seconds": 600,
        "observations": [
            {
                "observed_at": "2026-08-30T12:00:00Z",
                "status": "PEND",
                "reason": "insufficient full-size allocation",
            },
            {
                "observed_at": "2026-08-30T12:10:00Z",
                "status": "PEND",
                "reason": "insufficient full-size allocation",
            },
        ],
    }
)


def load_preflight():
    spec = importlib.util.spec_from_file_location("bluevela_preflight", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BlueVelaPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.preflight = load_preflight()

    def test_plans_generic_work_and_judge_topology_from_task_and_profile(self) -> None:
        plan = self.preflight.build_plan(
            REPRESENTATIVE_TASK,
            harness_root=HARNESS,
            cluster=PROFILE,
        )

        self.assertEqual(plan["task_contract"]["schema_version"], "1.4")
        self.assertEqual(plan["task_contract"]["work_gpus"], 32)
        self.assertEqual(plan["task_contract"]["judge_gpus"], 8)
        self.assertEqual(plan["resource_plan"]["mode"], "multi_node")
        self.assertEqual(plan["resource_plan"]["work_nodes"], 4)
        self.assertEqual(plan["resource_plan"]["judge_nodes"], 1)
        self.assertEqual(plan["resource_plan"]["total_nodes"], 5)
        self.assertTrue(plan["conditional_baseline_authority"]["required"])
        self.assertTrue(plan["profile"]["exclusive"])
        self.assertEqual(
            plan["commands"]["run"],
            [
                str(HARNESS / ".venv/bin/rsi-harness"),
                "run",
                str(REPRESENTATIVE_TASK.resolve()),
                "--cluster",
                str(PROFILE.resolve()),
            ],
        )
        self.assertNotIn("--gpus", plan["commands"]["run"])

    def test_rejects_legacy_task_owned_cluster_control_plane(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            task = Path(raw) / REPRESENTATIVE_TASK.name
            shutil.copytree(REPRESENTATIVE_TASK, task)
            (task / "cluster").mkdir()
            (task / "cluster/bluevela.toml").write_text("schema_version = 1\n")

            with self.assertRaisesRegex(ValueError, "task-owned cluster control plane"):
                self.preflight.build_plan(
                    task,
                    harness_root=HARNESS,
                    cluster=PROFILE,
                )

    def test_profile_not_task_determines_whole_node_arithmetic(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            task = Path(raw) / REPRESENTATIVE_TASK.name
            shutil.copytree(REPRESENTATIVE_TASK, task)
            task_toml = task / "task.toml"
            task_toml.write_text(task_toml.read_text().replace("gpus = 32", "gpus = 31"))

            with self.assertRaisesRegex(ValueError, "whole .*GPU nodes"):
                self.preflight.build_plan(
                    task,
                    harness_root=HARNESS,
                    cluster=PROFILE,
                )

    def test_same_entrypoint_preserves_single_node_branch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            task = Path(raw) / REPRESENTATIVE_TASK.name
            shutil.copytree(REPRESENTATIVE_TASK, task)
            task_toml = task / "task.toml"
            task_toml.write_text(task_toml.read_text().replace("gpus = 32", "gpus = 8"))

            plan = self.preflight.build_plan(
                task,
                harness_root=HARNESS,
                cluster=PROFILE,
            )

        self.assertEqual(plan["resource_plan"]["mode"], "single_node")
        self.assertEqual(plan["resource_plan"]["total_nodes"], 1)
        self.assertEqual(
            plan["resource_plan"]["pool_policy"],
            "legacy single-node Harness behavior",
        )

    def test_profile_gpu_density_is_not_hardcoded_by_task(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            profile = Path(raw) / "profile.toml"
            profile.write_text(
                PROFILE.read_text().replace("gpus_per_node = 8", "gpus_per_node = 4")
            )

            plan = self.preflight.build_plan(
                REPRESENTATIVE_TASK,
                harness_root=HARNESS,
                cluster=profile,
            )

        self.assertEqual(plan["resource_plan"]["work_nodes"], 8)
        self.assertEqual(plan["resource_plan"]["judge_nodes"], 2)

    def test_rejects_task_runtime_lsf_logic(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            task = Path(raw) / REPRESENTATIVE_TASK.name
            shutil.copytree(REPRESENTATIVE_TASK, task)
            solution = task / "solution/solve.sh"
            solution.write_text(solution.read_text() + "\nbsub dangerous-command\n")

            with self.assertRaisesRegex(
                ValueError, "TASK_CLUSTER_RUNTIME_FORBIDDEN"
            ):
                self.preflight.build_plan(
                    task,
                    harness_root=HARNESS,
                    cluster=PROFILE,
                )

    def test_rejects_generated_bytecode_from_image_context(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            task = Path(raw) / REPRESENTATIVE_TASK.name
            shutil.copytree(REPRESENTATIVE_TASK, task)
            cache = task / "solution/__pycache__"
            cache.mkdir()
            (cache / "solve.cpython-312.pyc").write_bytes(b"generated")

            with self.assertRaisesRegex(ValueError, "BYTECODE_CACHE_FORBIDDEN"):
                self.preflight.build_plan(
                    task,
                    harness_root=HARNESS,
                    cluster=PROFILE,
                )

    def test_end_to_end_rejects_nonexclusive_profile(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            profile = Path(raw) / "profile.toml"
            profile.write_text(
                PROFILE.read_text().replace("exclusive = true", "exclusive = false")
            )

            with self.assertRaisesRegex(ValueError, "exclusive profile"):
                self.preflight.build_plan(
                    REPRESENTATIVE_TASK,
                    harness_root=HARNESS,
                    cluster=profile,
                )

    def test_allocation_fallback_accepts_priority_nonexclusive_at_full_size(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            profile = Path(raw) / "allocation-fallback.toml"
            profile.write_text(
                PROFILE.read_text()
                .replace('queue = "normal"', 'queue = "priority"')
                .replace("exclusive = true", "exclusive = false")
            )

            plan = self.preflight.build_plan(
                REPRESENTATIVE_TASK,
                harness_root=HARNESS,
                cluster=profile,
                scheduler_policy="allocation-fallback",
                allocation_evidence=FALLBACK_EVIDENCE,
            )

        self.assertEqual(plan["profile"]["queue"], "priority")
        self.assertFalse(plan["profile"]["exclusive"])
        self.assertEqual(plan["resource_plan"]["work_gpus"], 32)
        self.assertEqual(plan["resource_plan"]["judge_gpus"], 8)
        self.assertTrue(plan["scheduler_policy"]["completed_run_is_final"])
        self.assertFalse(plan["scheduler_policy"]["normal_rerun_required"])
        self.assertTrue(
            plan["scheduler_policy"]["cancel_normal_before_fallback"]
        )
        self.assertIn(
            "two timestamped observations",
            plan["scheduler_policy"]["allocation_evidence_requirements"],
        )

    def test_allocation_fallback_requires_observed_scheduler_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            profile = Path(raw) / "allocation-fallback.toml"
            profile.write_text(
                PROFILE.read_text()
                .replace('queue = "normal"', 'queue = "priority"')
                .replace("exclusive = true", "exclusive = false")
            )

            with self.assertRaisesRegex(ValueError, "allocation evidence"):
                self.preflight.build_plan(
                    REPRESENTATIVE_TASK,
                    harness_root=HARNESS,
                    cluster=profile,
                    scheduler_policy="allocation-fallback",
                )

    def test_allocation_fallback_rejects_unstructured_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            profile = Path(raw) / "allocation-fallback.toml"
            profile.write_text(
                PROFILE.read_text()
                .replace('queue = "normal"', 'queue = "priority"')
                .replace("exclusive = true", "exclusive = false")
            )

            with self.assertRaisesRegex(ValueError, "JSON object"):
                self.preflight.build_plan(
                    REPRESENTATIVE_TASK,
                    harness_root=HARNESS,
                    cluster=profile,
                    scheduler_policy="allocation-fallback",
                    allocation_evidence="it was pending for a while",
                )

    def test_allocation_fallback_accepts_terminal_allocation_evidence(self) -> None:
        evidence = json.dumps(
            {
                "normal_job_id": "normal-456",
                "terminal": {
                    "observed_at": "2026-08-30T12:00:00Z",
                    "status": "EXIT",
                    "reason": "scheduler could not place the full allocation",
                    "allocation_specific": True,
                },
            }
        )
        with tempfile.TemporaryDirectory() as raw:
            profile = Path(raw) / "allocation-fallback.toml"
            profile.write_text(
                PROFILE.read_text()
                .replace('queue = "normal"', 'queue = "priority"')
                .replace("exclusive = true", "exclusive = false")
            )

            plan = self.preflight.build_plan(
                REPRESENTATIVE_TASK,
                harness_root=HARNESS,
                cluster=profile,
                scheduler_policy="allocation-fallback",
                allocation_evidence=evidence,
            )

        self.assertEqual(
            plan["scheduler_policy"]["allocation_evidence"]["terminal"][
                "status"
            ],
            "EXIT",
        )

    def test_default_policy_requires_normal_exclusive_profile(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            profile = Path(raw) / "wrong-default.toml"
            profile.write_text(
                PROFILE.read_text().replace(
                    'queue = "normal"', 'queue = "priority"'
                )
            )

            with self.assertRaisesRegex(ValueError, "normal.*exclusive"):
                self.preflight.build_plan(
                    REPRESENTATIVE_TASK,
                    harness_root=HARNESS,
                    cluster=profile,
                )

    def test_skill_exposes_one_proposal_to_reward_workflow(self) -> None:
        text = SKILL.read_text()
        self.assertNotIn("generate-only", text)
        self.assertNotIn("validate-existing", text)
        self.assertNotIn("Choose one mode", text)
        self.assertIn("approved proposal", text)
        self.assertIn("END_TO_END_VALIDATED", text)
        self.assertNotIn("STATIC_READY", text)


if __name__ == "__main__":
    unittest.main()
