from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "extract_logs.py"
SPEC = importlib.util.spec_from_file_location("extract_logs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
extract_logs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extract_logs)


class ExtractLogsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.run_root = root / "runs"
        self.logs_root = root / "logs"
        self.destination_root = root / "published"
        self.run_id = "20260901T111154Z-depth-width-allocation-566a0eea"
        self.task_id = "depth-width-allocation"
        self.source = self._make_completed_run()

    def _write_json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))

    def _make_completed_run(self) -> Path:
        self._write_json(
            self.run_root / self.run_id / "RUN_INFO.json",
            {
                "run_id": self.run_id,
                "state": "completed",
                "purpose": f"RSI Harness task {self.task_id}",
            },
        )
        leaf = self.logs_root / "runs" / self.run_id / self.task_id
        self._write_json(leaf / "final_result.json", {"status": "completed"})
        self._write_json(
            leaf / "run-plan.json",
            {
                "task": {
                    "agent": {
                        "name": "codex",
                        "model": "gpt-5.6-sol",
                        "reasoning_effort": "xhigh",
                    }
                }
            },
        )
        (leaf / "agent_output.txt").write_text("trajectory\n")
        (leaf / "run_agent.log").write_text("agent log\n")
        self._write_json(
            leaf / "submissions" / "agent-1" / "report.json",
            {"status": "completed", "metrics": {"reward": 1.0}},
        )
        feedback = leaf / "feedback" / "agent-1.log"
        feedback.parent.mkdir(parents=True, exist_ok=True)
        feedback.write_text("scored\n")
        return leaf

    def _run(self) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = extract_logs.main(
                [
                    self.task_id,
                    "--run-root",
                    str(self.run_root),
                    "--logs-root",
                    str(self.logs_root),
                    "--destination-root",
                    str(self.destination_root),
                ]
            )
        return result, stdout.getvalue(), stderr.getvalue()

    @property
    def standard_destination(self) -> Path:
        return (
            self.destination_root
            / self.task_id
            / "codex-gpt-5.6-sol-xhigh"
        )

    def test_publishes_task_agent_model_reasoning_layout(self) -> None:
        result, stdout, stderr = self._run()

        self.assertEqual(0, result, stderr)
        self.assertTrue(self.standard_destination.is_dir())
        self.assertFalse((self.destination_root / self.run_id).exists())
        self.assertIn(str(self.standard_destination), stdout)
        self.assertTrue(extract_logs._same_tree(self.source, self.standard_destination))

    def test_identical_standard_destination_is_idempotent(self) -> None:
        self.assertEqual(0, self._run()[0])

        result, stdout, stderr = self._run()

        self.assertEqual(0, result, stderr)
        self.assertIn("Destination already matches source", stdout)

    def test_conflicting_standard_destination_is_not_overwritten(self) -> None:
        self.standard_destination.mkdir(parents=True)
        marker = self.standard_destination / "existing.txt"
        marker.write_text("keep me\n")

        result, _stdout, stderr = self._run()

        self.assertEqual(1, result)
        self.assertIn("destination already exists with different contents", stderr)
        self.assertEqual("keep me\n", marker.read_text())


if __name__ == "__main__":
    unittest.main()
