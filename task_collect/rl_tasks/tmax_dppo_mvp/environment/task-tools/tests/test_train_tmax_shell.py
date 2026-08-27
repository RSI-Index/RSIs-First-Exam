from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


TRAIN_SCRIPT = Path(__file__).resolve().parents[2] / "train_tmax.sh"
DOCKERFILE = Path(__file__).resolve().parents[2] / "Dockerfile"


def render(max_updates: object) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "TMAX_MAX_UPDATES": str(max_updates),
            "TMAX_ATTEMPT_ID": "render-test",
            "TMAX_TRAINING_DATA_DIR": "/app/training/data",
            "TMAX_TRAINING_PROMPT_FILE": "/app/training/prompt/system.txt",
            "TMAX_TRAINING_REWARD_FILE": "/app/training/reward/reward.py",
            "TMAX_RL_RECIPE_FILE": "/app/training/recipe.json",
        }
    )
    return subprocess.run(
        ["bash", str(TRAIN_SCRIPT), "--print-command"],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


class TrainTMaxShellTest(unittest.TestCase):
    def test_environment_image_installs_trusted_tools_and_default_training_artifacts(self) -> None:
        dockerfile = DOCKERFILE.read_text()

        self.assertIn("COPY --chmod=755 task-tools /task-tools", dockerfile)
        self.assertIn("COPY training-defaults /opt/training-defaults", dockerfile)
        self.assertIn('git -C /opt/project merge-base --is-ancestor "$TMAX_COMMIT" HEAD', dockerfile)
        self.assertIn("/app/training/prompt/system.txt", dockerfile)
        self.assertIn("/app/training/reward/reward.py", dockerfile)

    def test_arbitrary_attempt_length_preserves_official_topology_and_scheduler_horizon(self) -> None:
        for updates in (1, 7, 137, 200):
            with self.subTest(updates=updates):
                result = render(updates)
                self.assertEqual(result.returncode, 0, result.stderr)
                command = result.stdout
                self.assertIn(f"--total_episodes {updates * 256}", command)
                self.assertIn("--scheduler_horizon_steps 200", command)
                self.assertIn("--num_learners_per_node 8 8", command)
                self.assertIn("--vllm_num_engines 48", command)

    def test_training_artifact_paths_are_passed_only_to_training(self) -> None:
        result = render(7)
        self.assertEqual(result.returncode, 0, result.stderr)
        command = result.stdout

        self.assertIn("--dataset_mixer_list /app/training/data 1.0", command)
        self.assertIn("--system_prompt_override_file /app/training/prompt/system.txt", command)
        self.assertIn("--training_reward_file /app/training/reward/reward.py", command)
        self.assertIn("--training_artifact_root /app/training", command)

    def test_invalid_attempt_lengths_are_rejected_before_runtime_setup(self) -> None:
        for value in (0, 201, "5.0", "screen"):
            with self.subTest(value=value):
                result = render(value)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("TMAX_MAX_UPDATES", result.stderr)


if __name__ == "__main__":
    unittest.main()
