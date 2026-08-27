from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


EVALUATE = Path(__file__).resolve().parent / "evaluate.py"
spec = importlib.util.spec_from_file_location("tmax_evaluate", EVALUATE)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class EvaluateContractTest(unittest.TestCase):
    def test_comparison_requires_one_frozen_protocol_for_both_models(self) -> None:
        baseline = {"mean_reward": 0.25, "protocol_sha256": "a" * 64}
        candidate = {"mean_reward": 0.375, "protocol_sha256": "a" * 64}

        comparison = module.compare_results(baseline, candidate)

        self.assertEqual(comparison["absolute_gain"], 0.125)
        self.assertEqual(comparison["relative_gain"], 0.5)
        with self.assertRaisesRegex(ValueError, "protocol"):
            module.compare_results(baseline, {**candidate, "protocol_sha256": "b" * 64})

    def test_relative_gain_is_null_for_zero_baseline(self) -> None:
        comparison = module.compare_results(
            {"mean_reward": 0.0, "protocol_sha256": "a" * 64},
            {"mean_reward": 0.1, "protocol_sha256": "a" * 64},
        )

        self.assertIsNone(comparison["relative_gain"])

    def test_final_protocol_does_not_import_editable_training_prompt_or_reward(self) -> None:
        source = EVALUATE.read_text()

        self.assertNotIn("TMAX_TRAINING_PROMPT", source)
        self.assertNotIn("TMAX_TRAINING_REWARD", source)
        self.assertIn("TMAX_FINAL_PROMPT", source)
        self.assertIn("TMAX_FINAL_REWARD", source)

    def test_official_baseline_staging_is_pinned_to_step_200_commit(self) -> None:
        stage = (EVALUATE.parents[1] / "cluster" / "stage_official_baseline.sh").read_text()

        self.assertIn("allenai/tmax-9b", stage)
        self.assertIn("ecb24e8c608870ee5fc776d85f84500375768f15", stage)
        self.assertNotIn("step_500", stage)


if __name__ == "__main__":
    unittest.main()
