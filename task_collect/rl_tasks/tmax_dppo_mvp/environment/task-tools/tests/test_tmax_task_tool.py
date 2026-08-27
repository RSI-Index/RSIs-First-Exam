from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


TASK_TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_TOOLS))

from tmax_task_tool import (  # noqa: E402
    audit_output,
    finish_attempt,
    prepare_attempt,
    stage_attempt,
    stop_attempt,
    summarize_attempt,
)


def scientific_inputs(root: Path) -> dict[str, Path]:
    project = root / "project"
    data = root / "training" / "data"
    prompt = root / "training" / "prompt" / "system.txt"
    reward = root / "training" / "reward" / "reward.py"
    recipe = root / "training" / "recipe.json"
    project.mkdir(parents=True)
    data.mkdir(parents=True)
    prompt.parent.mkdir(parents=True)
    reward.parent.mkdir(parents=True)
    (project / "method.py").write_text("METHOD = 'dppo'\n")
    (data / "mix.json").write_text('{"public": 1.0}\n')
    prompt.write_text("training prompt\n")
    reward.write_text("def reward(row): return row['public_reward']\n")
    recipe.write_text('{"algorithm": "dppo", "unique_prompts": 8, "group_size": 32}\n')
    return {
        "project": project,
        "training_data": data,
        "training_prompt": prompt,
        "training_reward": reward,
        "rl_recipe": recipe,
    }


class TMaxTaskToolTest(unittest.TestCase):
    def test_prepare_freezes_contract_and_rejects_duplicate_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            hypothesis = root / "hypothesis.md"
            hypothesis.write_text("Seven updates should reveal whether reward variance survives.\n")
            paths = scientific_inputs(root)

            attempt = prepare_attempt(
                output_root=output,
                attempt_id="rl-001",
                max_updates=7,
                hypothesis_file=hypothesis,
                **paths,
            )

            contract = json.loads((attempt / "run_contract.json").read_text())
            self.assertEqual(contract["max_updates"], 7)
            self.assertEqual(contract["scheduler_horizon_updates"], 200)
            self.assertEqual(contract["gpu_count"], 64)
            self.assertEqual(contract["learner_gpus"], 16)
            self.assertEqual(contract["inference_gpus"], 48)
            self.assertTrue((output / "source-worktrees" / contract["source_sha256"] / "project").is_dir())
            self.assertEqual(Path(contract["training_artifact_snapshot"]).parent, attempt)
            self.assertTrue(Path(contract["training_data_snapshot"]).is_dir())
            self.assertEqual(Path(contract["training_prompt_snapshot"]).read_text(), "training prompt\n")
            self.assertIn("def reward", Path(contract["training_reward_snapshot"]).read_text())
            self.assertIn("dppo", Path(contract["rl_recipe_snapshot"]).read_text())

            paths["training_prompt"].write_text("mutated after prepare\n")
            self.assertEqual(Path(contract["training_prompt_snapshot"]).read_text(), "training prompt\n")

            with self.assertRaisesRegex(ValueError, "already exists"):
                prepare_attempt(
                    output_root=output,
                    attempt_id="rl-001",
                    max_updates=7,
                    hypothesis_file=hypothesis,
                    **paths,
                )

    def test_finish_appends_trusted_completed_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            hypothesis = root / "hypothesis.md"
            hypothesis.write_text("Complete seven updates.\n")
            attempt = prepare_attempt(
                output_root=output,
                attempt_id="rl-002",
                max_updates=7,
                hypothesis_file=hypothesis,
                **scientific_inputs(root),
            )
            counters = attempt / "trusted_counters.json"
            counters.write_text(json.dumps({
                "optimizer_updates": 7,
                "training_trajectories": 1792,
                "generated_tokens": 123,
                "sandbox_steps": 45,
                "gpu_count": 64,
                "metrics": {"mean_public_reward": 0.25},
            }))

            row = finish_attempt(attempt=attempt, exit_code=0, counters_file=counters)

            self.assertEqual(row["status"], "completed")
            self.assertEqual(row["optimizer_updates"], 7)
            ledger_rows = (output / "experiments.jsonl").read_text().splitlines()
            self.assertEqual(len(ledger_rows), 1)

    def test_summarize_combines_contract_and_terminal_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            hypothesis = root / "hypothesis.md"
            hypothesis.write_text("Inspect a short run.\n")
            attempt = prepare_attempt(
                output_root=output,
                attempt_id="summary",
                max_updates=3,
                hypothesis_file=hypothesis,
                **scientific_inputs(root),
            )
            counters = attempt / "trusted_counters.json"
            counters.write_text(json.dumps({
                "optimizer_updates": 3,
                "training_trajectories": 768,
                "generated_tokens": 12,
                "sandbox_steps": 4,
                "gpu_count": 64,
                "metrics": {"mean_public_reward": 0.5},
            }))
            finish_attempt(attempt=attempt, exit_code=0, counters_file=counters)

            summary = summarize_attempt(attempt)

            self.assertEqual(summary["attempt_id"], "summary")
            self.assertEqual(summary["max_updates"], 3)
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["metrics"]["mean_public_reward"], 0.5)

    def test_stop_records_reason_and_signals_only_controlled_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            hypothesis = root / "hypothesis.md"
            reason = root / "reason.md"
            hypothesis.write_text("Stop if reward becomes non-finite.\n")
            reason.write_text("Non-finite public training reward at update 2.\n")
            attempt = prepare_attempt(
                output_root=output,
                attempt_id="stopped",
                max_updates=9,
                hypothesis_file=hypothesis,
                **scientific_inputs(root),
            )
            (attempt / "control.json").write_text('{"pid": 43210}\n')

            with patch("tmax_task_tool.os.killpg") as killpg:
                record = stop_attempt(attempt, reason)

            killpg.assert_called_once_with(43210, 15)
            self.assertIn("Non-finite", record["reason"])
            self.assertTrue((attempt / "early_stop.json").is_file())

    def test_stage_rejects_short_attempt_and_accepts_complete_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            hypothesis = root / "hypothesis.md"
            hypothesis.write_text("Full candidate.\n")
            paths = scientific_inputs(root)

            short = prepare_attempt(
                output_root=output,
                attempt_id="short",
                max_updates=7,
                hypothesis_file=hypothesis,
                **paths,
            )
            with self.assertRaisesRegex(ValueError, "200 updates"):
                stage_attempt(short, short / "model", output)

            full = prepare_attempt(
                output_root=output,
                attempt_id="full",
                max_updates=200,
                hypothesis_file=hypothesis,
                **paths,
            )
            counters = full / "trusted_counters.json"
            counters.write_text(json.dumps({
                "optimizer_updates": 200,
                "training_trajectories": 51200,
                "generated_tokens": 1000,
                "sandbox_steps": 500,
                "gpu_count": 64,
                "metrics": {},
            }))
            finish_attempt(attempt=full, exit_code=0, counters_file=counters)
            model = full / "model"
            model.mkdir()
            (model / "config.json").write_text('{"model_type": "qwen3_5"}\n')
            (model / "model-00001-of-00001.safetensors").write_bytes(b"weights")

            selection = stage_attempt(full, model, output)

            self.assertEqual(selection["optimizer_updates"], 200)
            self.assertEqual(selection["training_trajectories"], 51200)
            self.assertTrue((output / "submission" / "model" / "config.json").is_file())
            self.assertTrue((output / "submission" / "model-manifest.json").is_file())

            report = audit_output(output)
            self.assertEqual(report["status"], "passed")

            (output / "submission" / "model" / "config.json").write_text("tampered\n")
            with self.assertRaisesRegex(ValueError, "model manifest mismatch"):
                audit_output(output)


if __name__ == "__main__":
    unittest.main()
