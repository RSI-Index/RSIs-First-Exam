#!/usr/bin/env python3
"""Behavioral contract tests for the portable search-RL Layer 1 package."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


TASK_ROOT = Path(__file__).resolve().parents[1]
TOOLS = TASK_ROOT / "environment" / "task-tools"
POLICY_CHECK = TASK_ROOT / "tests" / "policy_check.py"


def run_tool(name: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOLS / name), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )


class SearchRlLayer1Contract(unittest.TestCase):
    def test_task_tools_expose_help(self) -> None:
        for name in ("preflight.py", "run_candidate.py", "evaluate.py", "task_state.py"):
            result = run_tool(name, "--help")
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("usage:", result.stdout.lower())

    def test_preflight_rejects_unsafe_attempt_id(self) -> None:
        result = run_tool("preflight.py", "--attempt-id", "../escape")
        self.assertNotEqual(result.returncode, 0)

    def test_smoke_run_evaluates_and_records_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            hypothesis = output / "hypotheses" / "smoke-001.json"
            hypothesis.parent.mkdir(parents=True)
            hypothesis.write_text(
                json.dumps({"attempt_id": "smoke-001", "hypothesis": "exercise artifact wiring"})
                + "\n",
                encoding="utf-8",
            )
            run_dir = output / "attempts" / "smoke-001"

            preflight = run_tool(
                "preflight.py",
                "--attempt-id",
                "smoke-001",
                "--hypothesis-file",
                str(hypothesis),
                "--output-root",
                str(output),
            )
            self.assertEqual(preflight.returncode, 0, msg=preflight.stderr)

            candidate = run_tool(
                "run_candidate.py",
                "--attempt-id",
                "smoke-001",
                "--hypothesis-file",
                str(hypothesis),
                "--output-root",
                str(output),
                "--dry-run",
            )
            self.assertEqual(candidate.returncode, 0, msg=candidate.stderr)

            evaluation = run_tool("evaluate.py", "--run-dir", str(run_dir))
            self.assertEqual(evaluation.returncode, 0, msg=evaluation.stderr)
            result = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
            self.assertEqual(result["validation_mode"], "layer1-smoke")
            self.assertEqual(result["scientific_validation"], "not-run")

            record = run_tool(
                "task_state.py",
                "record",
                "--run-dir",
                str(run_dir),
                "--output-root",
                str(output),
            )
            self.assertEqual(record.returncode, 0, msg=record.stderr)
            stage = run_tool(
                "task_state.py",
                "stage",
                "--run-dir",
                str(run_dir),
                "--output-root",
                str(output),
            )
            self.assertEqual(stage.returncode, 0, msg=stage.stderr)
            staged = json.loads((output / "staged_candidate.json").read_text(encoding="utf-8"))
            self.assertEqual(staged["attempt_id"], "smoke-001")

    def test_evaluator_rejects_missing_candidate_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "missing"
            run_dir.mkdir()
            result = run_tool("evaluate.py", "--run-dir", str(run_dir))
            self.assertNotEqual(result.returncode, 0)

    def test_full_run_cannot_be_mislabeled_as_available(self) -> None:
        result = run_tool(
            "run_candidate.py",
            "--attempt-id",
            "full-001",
            "--hypothesis-file",
            "/app/output/hypotheses/full-001.json",
        )
        self.assertEqual(result.returncode, 3)
        self.assertIn("FULL-RUN-UNVERIFIED", result.stderr)

    def test_policy_gate_rejects_exposed_evaluation_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            launcher = project / "training_scripts/rl/recipe/deepresearch"
            launcher.mkdir(parents=True)
            (launcher / "run_deepresearch_fully_async_megatron.sh").write_text(
                "#!/usr/bin/env bash\n", encoding="utf-8"
            )
            (project / "evaluation").mkdir()
            result = subprocess.run(
                [
                    sys.executable,
                    str(POLICY_CHECK),
                    "--project",
                    str(project),
                    "--policy",
                    str(TASK_ROOT / "policy.yaml"),
                    "--task",
                    "search-rl",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)

    def test_materialized_project_is_redacted(self) -> None:
        project_raw = os.environ.get("TASK_PROJECT")
        if not project_raw:
            self.skipTest("TASK_PROJECT is set by isolated-copy validation")
        project = Path(project_raw)
        self.assertFalse((project / "evaluation").exists())
        self.assertTrue(
            (project / "training_scripts/rl/recipe/deepresearch/run_deepresearch_fully_async_megatron.sh").is_file()
        )


if __name__ == "__main__":
    unittest.main()
