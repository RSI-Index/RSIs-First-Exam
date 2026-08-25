from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


TASK_TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_TOOLS))

from tmax_async_run import run_process  # noqa: E402
from tmax_contract import tree_digest  # noqa: E402


class AsyncRunTest(unittest.TestCase):
    def test_trusted_runner_rehashes_every_scientific_snapshot_before_gpu_launch(self) -> None:
        source = (TASK_TOOLS / "run_tmax_attempt.sh").read_text()

        for field in (
            "source_sha256",
            "training_data_sha256",
            "training_prompt_sha256",
            "training_reward_sha256",
            "rl_recipe_sha256",
        ):
            self.assertIn(field, source)
        self.assertIn("tree_digest", source)

    def test_trusted_runner_maps_container_snapshots_into_bound_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "candidate-output"
            attempt = run_root / "attempts" / "rl-path-map"
            source = run_root / "source-worktrees" / "digest" / "project"
            training = attempt / "training-inputs"
            data = training / "data"
            prompt = training / "prompt" / "system.txt"
            reward = training / "reward" / "reward.py"
            recipe = training / "recipe.json"
            for directory in (source, data, prompt.parent, reward.parent):
                directory.mkdir(parents=True, exist_ok=True)
            (source / "candidate.py").write_text("VALUE = 1\n")
            (data / "train.jsonl").write_text('{"prompt":"x"}\n')
            prompt.write_text("training prompt\n")
            reward.write_text("def reward(value):\n    return value\n")
            recipe.write_text('{"algorithm":"dppo"}\n')

            def container_path(path: Path) -> str:
                return "/app/output/" + path.relative_to(run_root).as_posix()

            contract = {
                "attempt_id": "rl-path-map",
                "max_updates": 1,
                "source_snapshot": container_path(source),
                "training_artifact_snapshot": container_path(training),
                "training_data_snapshot": container_path(data),
                "training_prompt_snapshot": container_path(prompt),
                "training_reward_snapshot": container_path(reward),
                "rl_recipe_snapshot": container_path(recipe),
                "source_sha256": tree_digest(source),
                "training_data_sha256": tree_digest(data),
                "training_prompt_sha256": tree_digest(prompt),
                "training_reward_sha256": tree_digest(reward),
                "rl_recipe_sha256": tree_digest(recipe),
            }
            contract_path = attempt / "run_contract.json"
            contract_path.write_text(json.dumps(contract))
            captured = attempt / "captured.json"
            fake_runner = Path(temporary) / "fake-runner.sh"
            fake_runner.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "python3 - <<'PY'\n"
                "import json, os\n"
                "from pathlib import Path\n"
                "keys = ['TMAX_PROJECT_ROOT', 'TMAX_TRAINING_DATA_DIR', "
                "'TMAX_TRAINING_PROMPT_FILE', 'TMAX_TRAINING_REWARD_FILE', "
                "'TMAX_TRAINING_ARTIFACT_ROOT', 'TMAX_RL_RECIPE_FILE']\n"
                "Path(os.environ['CAPTURED']).write_text(json.dumps({k: os.environ[k] for k in keys}))\n"
                "PY\n"
            )
            fake_runner.chmod(0o755)
            env = {
                **os.environ,
                "TMAX_RUN_ROOT": str(run_root),
                "TMAX_64GPU_RUNNER": str(fake_runner),
                "CAPTURED": str(captured),
            }

            completed = subprocess.run(
                [str(TASK_TOOLS / "run_tmax_attempt.sh"), str(contract_path)],
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            resolved = json.loads(captured.read_text())
            self.assertEqual(resolved["TMAX_PROJECT_ROOT"], str(source))
            self.assertEqual(resolved["TMAX_TRAINING_DATA_DIR"], str(data))
            self.assertEqual(resolved["TMAX_TRAINING_PROMPT_FILE"], str(prompt))
            self.assertEqual(resolved["TMAX_TRAINING_REWARD_FILE"], str(reward))
            self.assertEqual(resolved["TMAX_TRAINING_ARTIFACT_ROOT"], str(training))
            self.assertEqual(resolved["TMAX_RL_RECIPE_FILE"], str(recipe))

    def test_process_is_polled_and_terminal_state_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            attempt = Path(temporary) / "attempt"
            attempt.mkdir()

            state = run_process(
                attempt=attempt,
                command=[sys.executable, "-c", "import time; time.sleep(0.03)"],
                poll_seconds=0.005,
            )

            self.assertEqual(state["status"], "completed")
            self.assertEqual(state["exit_code"], 0)
            self.assertGreaterEqual(state["poll_count"], 1)
            control = json.loads((attempt / "control.json").read_text())
            self.assertEqual(control["pid"], control["process_group_id"])
            self.assertEqual(json.loads((attempt / "async_status.json").read_text()), state)

    def test_second_active_process_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            attempt = Path(temporary) / "attempt"
            attempt.mkdir()
            (attempt / "control.json").write_text(
                json.dumps({"pid": 2, "process_group_id": 2, "status": "running"})
            )

            with self.assertRaisesRegex(ValueError, "active attempt"):
                run_process(
                    attempt=attempt,
                    command=[sys.executable, "-c", "pass"],
                    poll_seconds=0.01,
                )


if __name__ == "__main__":
    unittest.main()
