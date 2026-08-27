from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


TASK_TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_TOOLS))

from tmax_contract import (  # noqa: E402
    SCIENTIFIC_HASH_FIELDS,
    append_jsonl,
    assert_resume_compatible,
    build_input_manifest,
    tree_digest,
    validate_attempt_id,
    validate_max_updates,
)


class TMaxContractTest(unittest.TestCase):
    def test_any_integer_between_one_and_two_hundred_is_valid(self) -> None:
        for value in (1, "7", 137, "200"):
            self.assertEqual(validate_max_updates(value), int(value))

        for value in (0, "0", 201, "201", "5.0", "screen", True, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_max_updates(value)

    def test_attempt_id_accepts_safe_names_only(self) -> None:
        self.assertEqual(validate_attempt_id("rl-007_prompt-v2"), "rl-007_prompt-v2")
        for value in ("", ".hidden", "../escape", "has space", "x" * 97):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_attempt_id(value)

    def test_resume_requires_all_scientific_hashes_to_match(self) -> None:
        prior = {name: "a" * 64 for name in SCIENTIFIC_HASH_FIELDS}
        assert_resume_compatible(prior, dict(prior))

        for field in SCIENTIFIC_HASH_FIELDS:
            changed = dict(prior)
            changed[field] = "b" * 64
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                assert_resume_compatible(prior, changed)

    def test_tree_digest_is_stable_and_rejects_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "b.txt").write_text("two\n")
            (root / "a.txt").write_text("one\n")
            first = tree_digest(root)
            second = tree_digest(root)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 64)

            (root / "link").symlink_to(root / "a.txt")
            with self.assertRaisesRegex(ValueError, "symlink"):
                tree_digest(root)

    def test_input_manifest_hashes_all_five_scientific_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            data = root / "data"
            source.mkdir()
            data.mkdir()
            (source / "method.py").write_text("METHOD = 'dppo'\n")
            (data / "mix.json").write_text('{"public": 1.0}\n')
            prompt = root / "prompt.txt"
            reward = root / "reward.py"
            recipe = root / "recipe.json"
            prompt.write_text("training prompt\n")
            reward.write_text("def reward(row): return row['public_reward']\n")
            recipe.write_text('{"algorithm": "dppo"}\n')

            manifest = build_input_manifest(
                attempt_id="rl-001",
                max_updates=7,
                source=source,
                training_data=data,
                training_prompt=prompt,
                training_reward=reward,
                rl_recipe=recipe,
            ).to_dict()

            self.assertEqual(manifest["attempt_id"], "rl-001")
            self.assertEqual(manifest["max_updates"], 7)
            self.assertEqual(manifest["scheduler_horizon_updates"], 200)
            self.assertEqual(set(SCIENTIFIC_HASH_FIELDS), set(manifest) & set(SCIENTIFIC_HASH_FIELDS))
            self.assertTrue(all(len(manifest[field]) == 64 for field in SCIENTIFIC_HASH_FIELDS))

    def test_append_jsonl_preserves_existing_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "experiments.jsonl"
            append_jsonl(ledger, {"attempt_id": "one"})
            append_jsonl(ledger, {"attempt_id": "two"})
            rows = [json.loads(line) for line in ledger.read_text().splitlines()]
            self.assertEqual([row["attempt_id"] for row in rows], ["one", "two"])


if __name__ == "__main__":
    unittest.main()
