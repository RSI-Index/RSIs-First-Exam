from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

import yaml


TASK_ROOT = Path(__file__).resolve().parents[1]


class TMaxTaskContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.task = tomllib.loads((TASK_ROOT / "task.toml").read_text())
        cls.policy = yaml.safe_load((TASK_ROOT / "policy.yaml").read_text())
        cls.instruction = (TASK_ROOT / "instruction.md").read_text()

    def test_final_budget_is_two_hundred_updates_on_fixed_9b_model(self) -> None:
        contract = self.task["metadata"]["run_contract"]
        constraints = self.policy["constraints"]

        self.assertEqual(contract.get("final_optimizer_updates"), 200)
        self.assertEqual(contract.get("final_training_trajectories"), 51_200)
        self.assertEqual(contract.get("attempt_updates"), "agent_selected_1_to_200")
        self.assertEqual(constraints.get("required_optimizer_updates"), 200)
        self.assertEqual(constraints.get("required_training_trajectories"), 51_200)
        self.assertEqual(constraints.get("fixed_trajectories_per_update"), 256)
        self.assertEqual(constraints.get("starting_model"), "hamishivi/Qwen3.5-9B at the staged immutable revision")

    def test_training_data_prompt_and_reward_are_in_action_space(self) -> None:
        allowed = "\n".join(self.policy["scope"]["allowed"]).lower()

        self.assertIn("training data selection", allowed)
        self.assertIn("training prompts", allowed)
        self.assertIn("training reward", allowed)
        self.assertIn("bounded synthetic augmentation", allowed)

    def test_final_prompt_reward_and_evaluation_assets_are_frozen(self) -> None:
        constraints = self.policy["constraints"]
        protected = set(self.policy["scope"]["protected_paths"])

        self.assertEqual(constraints.get("final_prompt"), "frozen")
        self.assertEqual(constraints.get("final_reward"), "original programmatic verifier")
        self.assertIn("evaluation_assets/**", protected)
        self.assertIn("Vanillux2Agent/**", protected)

    def test_instruction_assigns_run_length_decisions_to_agent(self) -> None:
        lower = self.instruction.lower()

        self.assertIn("you choose how many optimizer updates", lower)
        self.assertIn("1 through 200", lower)
        self.assertNotIn("publish screening anchors at", lower)
        self.assertNotIn("only a completed 500-update run", lower)

    def test_official_step_200_is_baseline_artifact_not_retrained_run(self) -> None:
        baseline = self.task["metadata"]["baseline"]

        self.assertEqual(baseline.get("method"), "official released allenai/tmax-9b step_200 checkpoint")
        self.assertEqual(baseline.get("training_reproduction"), "not_required")
        self.assertEqual(baseline.get("optimizer_updates"), 200)


if __name__ == "__main__":
    unittest.main()
