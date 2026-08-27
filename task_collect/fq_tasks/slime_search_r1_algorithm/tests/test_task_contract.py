from __future__ import annotations

import importlib.util
import math
import json
import tempfile
import unittest
from pathlib import Path


TASK_DIR = Path(__file__).resolve().parents[1]
ENVIRONMENT = TASK_DIR / "environment"
SHARED = ENVIRONMENT


def load_module(name: str, path: Path):
    if not path.is_file():
        raise AssertionError(f"required module is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def flag_value(command: list[str], flag: str) -> str:
    index = command.index(flag)
    if index + 1 >= len(command) or command[index + 1].startswith("--"):
        raise AssertionError(f"{flag} has no value")
    return command[index + 1]


class ScientificReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_module("slime_task_contract", SHARED / "task_contract.py")

    def test_command_preserves_the_1000_round_eight_gpu_receipt(self):
        command = self.contract.build_training_arguments(
            self.contract.BASELINE_ALGORITHM,
            "/app/output/attempts/baseline-1/train",
        )
        self.assertEqual(flag_value(command, "--num-rollout"), "1000")
        self.assertEqual(flag_value(command, "--global-batch-size"), "256")
        self.assertEqual(flag_value(command, "--rollout-batch-size"), "32")
        self.assertEqual(flag_value(command, "--n-samples-per-prompt"), "8")
        self.assertEqual(flag_value(command, "--actor-num-gpus-per-node"), "4")
        self.assertEqual(flag_value(command, "--rollout-num-gpus"), "4")
        self.assertEqual(flag_value(command, "--tensor-model-parallel-size"), "2")
        self.assertEqual(flag_value(command, "--rollout-temperature"), "1.0")
        self.assertEqual(flag_value(command, "--rollout-top-p"), "1.0")
        self.assertEqual(flag_value(command, "--rollout-top-k"), "-1")
        self.assertEqual(flag_value(command, "--rollout-max-response-len"), "512")
        self.assertEqual(flag_value(command, "--rollout-seed"), "123")
        self.assertEqual(flag_value(command, "--clip-grad"), "1.0")
        self.assertIn("--sglang-enable-deterministic-inference", command)
        self.assertNotIn("--max-grad-norm", command)

    def test_command_uses_frozen_search_r1_data_reward_and_generation(self):
        command = self.contract.build_training_arguments(
            self.contract.BASELINE_ALGORITHM,
            "/app/output/attempts/run-7/train",
        )
        self.assertEqual(flag_value(command, "--hf-checkpoint"), "/models/Qwen2.5-3B")
        self.assertEqual(flag_value(command, "--load"), "/models/Qwen2.5-3B")
        self.assertEqual(flag_value(command, "--ref-load"), "/models/Qwen2.5-3B")
        self.assertEqual(flag_value(command, "--prompt-data"), "/datasets/nq_hotpotqa_train/train.parquet")
        self.assertEqual(flag_value(command, "--custom-generate-function-path"), "generate_with_search.generate")
        self.assertEqual(flag_value(command, "--custom-rm-path"), "generate_with_search.reward_func")
        self.assertEqual(flag_value(command, "--seed"), "123")
        self.assertEqual(flag_value(command, "--save"), "/app/output/attempts/run-7/train/torch_dist")

    def test_scientific_overrides_are_not_editable_algorithm_keys(self):
        forbidden = {
            "num_rollout": 5,
            "rollout_temperature": 0,
            "global_batch_size": 16,
            "prompt_data": "/tmp/other.parquet",
        }
        for key, value in forbidden.items():
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                self.contract.validate_algorithm_config({key: value})

    def test_attempt_output_must_be_one_direct_child_with_train_suffix(self):
        invalid = [
            "/app/output/train",
            "/app/output/attempts/a",
            "/app/output/attempts/a/train/nested",
            "/tmp/a/train",
        ]
        for path in invalid:
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "attempt output"):
                self.contract.build_training_arguments(self.contract.BASELINE_ALGORITHM, path)
        trusted = self.contract.build_training_arguments(
            self.contract.BASELINE_ALGORITHM,
            "/var/lib/slime-task/runs/run-1/train",
        )
        self.assertEqual(flag_value(trusted, "--save"), "/var/lib/slime-task/runs/run-1/train/torch_dist")

    def test_algorithm_config_loader_accepts_only_the_algorithm_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "algorithm_config.toml"
            path.write_text(
                "[algorithm]\n"
                "advantage_estimator = \"grpo\"\n"
                "learning_rate = 0.000001\n"
                "eps_clip = 0.2\n",
                encoding="utf-8",
            )
            self.assertEqual(
                self.contract.load_algorithm_config(path),
                {"advantage_estimator": "grpo", "learning_rate": 0.000001, "eps_clip": 0.2},
            )

    def test_shipped_algorithm_config_is_a_valid_native_grpo_baseline(self):
        loaded = self.contract.load_algorithm_config(ENVIRONMENT / "algorithm_config.toml")
        self.assertEqual(loaded["advantage_estimator"], "grpo")
        self.assertTrue(loaded["use_kl_loss"])

    def test_kl_loss_choices_match_the_pinned_slime_parser(self):
        for value in ("k1", "k2", "k3", "low_var_kl"):
            with self.subTest(value=value):
                self.contract.validate_algorithm_config({"kl_loss_type": value})
        with self.assertRaisesRegex(ValueError, "kl_loss_type"):
            self.contract.validate_algorithm_config({"kl_loss_type": "kl"})

    def test_estimators_must_be_valid_for_the_frozen_no_critic_topology(self):
        for value in ("grpo", "gspo", "cispo"):
            self.contract.validate_algorithm_config({"advantage_estimator": value})
        self.contract.validate_algorithm_config(
            {"advantage_estimator": "reinforce_plus_plus", "normalize_advantages": True}
        )
        with self.assertRaisesRegex(ValueError, "critic topology"):
            self.contract.validate_algorithm_config({"advantage_estimator": "ppo"})
        with self.assertRaisesRegex(ValueError, "normalize_advantages"):
            self.contract.validate_algorithm_config({"advantage_estimator": "reinforce_plus_plus"})

    def test_all_external_assets_are_full_commit_pins(self):
        for revision in self.contract.ALL_ASSET_REVISIONS:
            with self.subTest(revision=revision):
                self.assertRegex(revision, r"^[0-9a-f]{40}$")


class AlgorithmBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_module("slime_algorithm_boundary", SHARED / "task_contract.py")

    def test_training_uses_native_slime_rl_without_participant_python_hooks(self):
        command = self.contract.build_training_arguments(
            self.contract.BASELINE_ALGORITHM,
            "/app/output/attempts/native-1/train",
        )
        self.assertFalse(any(flag.startswith("--custom-") and flag not in {
            "--custom-generate-function-path", "--custom-rm-path"
        } for flag in command))
        self.assertNotIn("rl_research.algorithm", " ".join(command))
        launcher_text = (ENVIRONMENT / "launch_train.py").read_text(encoding="utf-8")
        docker_text = (ENVIRONMENT / "Dockerfile").read_text(encoding="utf-8")
        self.assertNotIn('examples/search-r1:{ALGORITHM_ROOT', launcher_text)
        self.assertNotIn("PYTHONPATH=/task-tools:/app/project", docker_text)

    def test_algorithm_tree_is_exactly_one_declarative_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "algorithm_config.toml").write_text("[algorithm]\nlearning_rate = 1e-6\n", encoding="utf-8")
            self.assertRegex(self.contract.algorithm_source_sha256(root), r"^[0-9a-f]{64}$")
            (root / "extra.py").write_text("raise SystemExit\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "only algorithm_config.toml"):
                self.contract.algorithm_source_sha256(root)


class FrozenLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.launcher = load_module("slime_launch_train", ENVIRONMENT / "launch_train.py")

    def test_executable_command_wraps_only_the_frozen_argument_builder(self):
        command = self.launcher.build_slime_command(
            {"learning_rate": 2e-6},
            "/app/output/attempts/lr-2/train",
        )
        self.assertEqual(command[:2], ["python3", "/opt/slime/train.py"])
        self.assertEqual(flag_value(command, "--lr"), "2e-06")
        self.assertEqual(flag_value(command, "--num-rollout"), "1000")
        self.assertEqual(flag_value(command, "--save"), "/app/output/attempts/lr-2/train/torch_dist")

    def test_visible_gpu_parser_requires_exactly_eight_unique_devices(self):
        self.assertEqual(self.launcher.parse_visible_devices("0,1,2,3,4,5,6,7"), tuple(str(i) for i in range(8)))
        for value in ("", "0,1,2,3", "0,1,2,3,4,5,6,6", "all"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "eight"):
                self.launcher.parse_visible_devices(value)

    def test_attempt_rejects_algorithm_source_changed_during_training(self):
        self.assertEqual(self.launcher.require_stable_algorithm("a" * 64, "a" * 64), "a" * 64)
        with self.assertRaisesRegex(RuntimeError, "changed during training"):
            self.launcher.require_stable_algorithm("a" * 64, "b" * 64)

    def test_only_the_exact_final_checkpoint_after_1000_zero_based_rounds_can_be_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            partial = root / "iter_0000998"
            partial.mkdir()
            (partial / "common.pt").write_bytes(b"partial")
            (partial / ".metadata").write_bytes(b"partial")
            with self.assertRaisesRegex(RuntimeError, "iter_0000999"):
                self.launcher._resolve_dist_checkpoint(root)
            complete = root / "iter_0000999"
            complete.mkdir()
            (complete / "common.pt").write_bytes(b"complete")
            (complete / ".metadata").write_bytes(b"complete")
            self.assertEqual(self.launcher._resolve_dist_checkpoint(root), complete)


class AsyncContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_module("slime_async_contract", SHARED / "async_contract.py")

    def test_attempt_ids_are_stable_lowercase_slugs(self):
        self.assertEqual(self.contract.validate_attempt_id("grpo-001"), "grpo-001")
        for value in ("", "UPPER", "../escape", "a/b", "-leading", "a" * 65):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.contract.validate_attempt_id(value)

    def test_status_machine_cannot_regress_or_skip_terminal_states(self):
        legal = [("queued", "running"), ("running", "completed"), ("running", "failed")]
        for old, new in legal:
            with self.subTest(old=old, new=new):
                self.assertEqual(self.contract.validate_transition(old, new), new)
        for old, new in (("queued", "completed"), ("completed", "running"), ("failed", "queued")):
            with self.subTest(old=old, new=new), self.assertRaisesRegex(ValueError, "transition"):
                self.contract.validate_transition(old, new)


class AttemptSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_module("slime_async_runner", ENVIRONMENT / "slime_async_run.py")

    def _write_attempt(self, root: Path, num_rollout: int) -> None:
        attempt = root / "attempts" / "run-1"
        model = attempt / "train" / "hf_model"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}\n", encoding="utf-8")
        (model / "tokenizer.json").write_text("{}\n", encoding="utf-8")
        (model / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
        (attempt / "status.json").write_text(
            json.dumps({"attempt_id": "run-1", "hypothesis": "clipping reduces variance", "status": "completed"}),
            encoding="utf-8",
        )
        (attempt / "train" / "run-contract.json").write_text(
            json.dumps(
                {
                    "algorithm_sha256": "a" * 64,
                    "num_rollout": num_rollout,
                    "training_command_sha256": "6ea2e456ab651e7bebe564f86acf2e5cb47eeb57a2af4ccd937293a5513c14a4",
                    "world_size": 8,
                }
            ),
            encoding="utf-8",
        )
        (attempt / "train" / "training-command.json").write_text(
            json.dumps(
                {
                    "argv": ["python3", "/opt/slime/train.py"],
                    "sha256": "6ea2e456ab651e7bebe564f86acf2e5cb47eeb57a2af4ccd937293a5513c14a4",
                }
            ),
            encoding="utf-8",
        )
        (root / "experiments.jsonl").write_text(
            json.dumps({"attempt_id": "run-1", "event": "completed", "num_rollout": num_rollout}) + "\n",
            encoding="utf-8",
        )

    def test_selection_materializes_only_a_completed_full_budget_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_attempt(root, 1000)
            payload = self.runner.select_attempt(root, "run-1")
            self.assertEqual(payload["selected_attempt"], "run-1")
            self.assertEqual(payload["num_rollout"], 1000)
            self.assertTrue((root / "hf_model" / "config.json").is_file())
            self.assertEqual(
                json.loads((root / "provenance.json").read_text())["training_command_sha256"],
                "6ea2e456ab651e7bebe564f86acf2e5cb47eeb57a2af4ccd937293a5513c14a4",
            )

    def test_selection_rejects_a_partial_budget_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_attempt(root, 10)
            with self.assertRaisesRegex(ValueError, "full-budget"):
                self.runner.select_attempt(root, "run-1")


class RootAttestationTests(unittest.TestCase):
    def test_attempt_worker_can_only_use_the_root_owned_fixed_supervisor(self):
        runner = (ENVIRONMENT / "slime_async_run.py").read_text(encoding="utf-8")
        supervisor = (ENVIRONMENT / "root_supervisor.py").read_text(encoding="utf-8")
        dockerfile = (ENVIRONMENT / "Dockerfile").read_text(encoding="utf-8")
        policy_check = (TASK_DIR / "tests" / "policy_check.py").read_text(encoding="utf-8")
        entry = (ENVIRONMENT / "root_supervisor_entry.py").read_text(encoding="utf-8")
        self.assertIn('["sudo", "-n", "/task-tools/root_supervisor_entry.py", "--attempt-id", attempt_id]', runner)
        self.assertNotIn('["bash", "/task-tools/train.sh"', runner)
        self.assertIn("run_attempt(config_path, train_output)", supervisor)
        self.assertIn('"final_rollout_id": FINAL_ROLLOUT_ID', supervisor)
        self.assertIn('"hf_model_sha256": hf_hash', supervisor)
        self.assertIn('os.stat("/proc/self/ns/net").st_ino == os.stat("/proc/1/ns/net").st_ino', supervisor)
        self.assertIn('"/usr/bin/unshare"', entry)
        self.assertIn('"--net"', entry)
        self.assertIn("/var/lib/slime-task/attestations", dockerfile)
        self.assertIn("NOPASSWD: /task-tools/root_supervisor_entry.py --attempt-id *", dockerfile)
        self.assertNotIn("NOPASSWD: /task-tools/root_supervisor.py", dockerfile)
        self.assertIn("SYS_ADMIN", (ENVIRONMENT / "docker-compose.yaml").read_text(encoding="utf-8"))
        self.assertIn("NET_ADMIN", (ENVIRONMENT / "docker-compose.yaml").read_text(encoding="utf-8"))
        self.assertIn("/var/lib/slime-task/attestations", policy_check)
        self.assertIn("selected HF checkpoint differs from root-owned training attestation", policy_check)


class PolicyContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = load_module("slime_policy_contract", SHARED / "policy_contract.py")

    def test_missing_provenance_fields_fail_closed(self):
        errors = self.policy.validate_provenance({"selected_attempt": "run-1"})
        self.assertTrue(errors)
        self.assertIn("base_model", " ".join(errors))

    def test_complete_provenance_with_matching_algorithm_hash_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            algorithm = Path(tmp)
            (algorithm / "algorithm_config.toml").write_text("[algorithm]\n", encoding="utf-8")
            digest = load_module("policy_task_contract", SHARED / "task_contract.py").algorithm_source_sha256(algorithm)
            payload = {
                "algorithm_sha256": digest,
                "base_model": "Qwen/Qwen2.5-3B",
                "base_model_revision": "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b",
                "bm25_index_revision": "2c7554f25f425038c4bcb155735a0f831851fd78",
                "dataset_revision": "b7d80abfee334a7a91cb377544f09180d58b34f6",
                "num_rollout": 1000,
                "optimization_attempts": ["run-1"],
                "search_r1_commit": "598e61bd1d36895726d28a8d06b3a15bed19f5d3",
                "selected_attempt": "run-1",
                "slime_commit": "06ffdbe22be068b52f9ed0fc318c473f7030197e",
                "training_command": ["python3", "/opt/slime/train.py"],
                "training_command_sha256": "6ea2e456ab651e7bebe564f86acf2e5cb47eeb57a2af4ccd937293a5513c14a4",
                "web_search": "disabled",
                "world_size": 8,
            }
            self.assertEqual(self.policy.validate_provenance(payload, algorithm), [])


class ScoreContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluation = load_module("slime_evaluation_contract", SHARED / "evaluation_contract.py")

    def test_halfway_candidate_receives_half_dynamic_reward(self):
        self.assertAlmostEqual(self.evaluation.normalized_reward(0.4, 0.2, 0.6), 0.5)
        self.assertEqual(self.evaluation.normalized_reward(0.1, 0.2, 0.6), 0.0)
        self.assertEqual(self.evaluation.normalized_reward(0.8, 0.2, 0.6), 1.0)

    def test_nonimproving_or_nonfinite_anchor_fails_closed(self):
        for candidate, base, reference in ((0.3, 0.4, 0.4), (math.nan, 0.2, 0.6), (0.3, 0.2, math.inf)):
            with self.subTest(values=(candidate, base, reference)), self.assertRaises(ValueError):
                self.evaluation.normalized_reward(candidate, base, reference)

    def test_verifier_commands_start_cpu_bm25_and_freeze_base_tokenizer(self):
        bm25 = self.evaluation.build_bm25_command()
        self.assertEqual(bm25[:2], ["python3", "/opt/Search-R1/search_r1/search/retrieval_server.py"])
        self.assertEqual(flag_value(bm25, "--index_path"), "/retriever/bm25")
        self.assertEqual(flag_value(bm25, "--corpus_path"), "/retriever/wiki-18.jsonl")
        self.assertEqual(flag_value(bm25, "--topk"), "3")
        self.assertEqual(flag_value(bm25, "--retriever_name"), "bm25")
        self.assertNotIn("--faiss_gpu", bm25)

        server = self.evaluation.build_sglang_command("/app/output/hf_model", 30000)
        self.assertEqual(flag_value(server, "--model-path"), "/app/output/hf_model")
        self.assertEqual(flag_value(server, "--tokenizer-path"), "/models/Qwen2.5-3B")
        self.assertIn("--enable-deterministic-inference", server)
        self.assertNotIn("--trust-remote-code", server)

    def test_reward_label_shape_matches_pinned_search_r1(self):
        record = {"reward_model": {"ground_truth": {"target": ["answer"]}}}
        self.assertEqual(
            self.evaluation.reward_label(record),
            {"ground_truth": {"target": ["answer"]}},
        )
        with self.assertRaisesRegex(ValueError, "ground_truth"):
            self.evaluation.reward_label({"reward_model": {}})

    def test_candidate_is_staged_with_frozen_config_before_root_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            candidate = root / "candidate"
            staged = root / "staged"
            base.mkdir()
            candidate.mkdir()
            frozen = {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "eos_token_id": 151645}
            (base / "config.json").write_text(json.dumps(frozen), encoding="utf-8")
            (base / "generation_config.json").write_text('{"eos_token_id":151645}', encoding="utf-8")
            (base / "tokenizer.json").write_text('{"frozen":true}', encoding="utf-8")
            changed = dict(frozen, eos_token_id=0)
            (candidate / "config.json").write_text(json.dumps(changed), encoding="utf-8")
            (candidate / "tokenizer.json").write_text('{"frozen":true}', encoding="utf-8")
            (candidate / "model-00001-of-00001.safetensors").write_bytes(b"weights")
            (candidate / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"model.embed_tokens.weight": "model-00001-of-00001.safetensors"}}),
                encoding="utf-8",
            )
            self.evaluation.stage_candidate_checkpoint(candidate, staged, base, frozen)
            self.assertEqual(json.loads((staged / "config.json").read_text()), frozen)
            self.assertEqual(
                {path.name for path in staged.iterdir()},
                {"config.json", "generation_config.json", "model.safetensors.index.json", "model-00001-of-00001.safetensors"},
            )
            self.assertEqual((staged / "config.json").stat().st_mode & 0o222, 0)

    def test_aggregate_requires_exactly_512_bounded_finite_row_scores(self):
        self.assertEqual(self.evaluation.aggregate_scores([0.5] * 512), 0.5)
        for values in ([0.5], [0.5] * 511 + [math.nan], [0.5] * 511 + [1.1]):
            with self.subTest(length=len(values)), self.assertRaises(ValueError):
                self.evaluation.aggregate_scores(values)

    def test_checkpoint_tree_rejects_symlinks_and_unindexed_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text('{"architectures":["Qwen2ForCausalLM"],"model_type":"qwen2"}', encoding="utf-8")
            (root / "tokenizer.json").write_text("{}", encoding="utf-8")
            (root / "model-00001-of-00001.safetensors").write_bytes(b"weights")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"model.embed_tokens.weight": "model-00001-of-00001.safetensors"}}),
                encoding="utf-8",
            )
            self.evaluation.validate_checkpoint_tree(root, {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2"})
            (root / "extra.safetensors").write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "unindexed"):
                self.evaluation.validate_checkpoint_tree(root, {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2"})
            (root / "extra.safetensors").unlink()
            (root / "tokenizer.json").unlink()
            (root / "tokenizer.json").symlink_to("config.json")
            with self.assertRaisesRegex(ValueError, "regular"):
                self.evaluation.validate_checkpoint_tree(root, {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2"})

    def test_checkpoint_rejects_architecture_drift_remote_code_and_tokenizer_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            candidate = Path(tmp) / "candidate"
            base.mkdir()
            candidate.mkdir()
            expected = {
                "architectures": ["Qwen2ForCausalLM"],
                "model_type": "qwen2",
                "hidden_size": 2048,
                "num_hidden_layers": 36,
                "vocab_size": 151936,
            }
            (base / "tokenizer.json").write_text('{"frozen":true}', encoding="utf-8")
            (base / "tokenizer_config.json").write_text('{"chat_template":"base"}', encoding="utf-8")
            (candidate / "tokenizer.json").write_text('{"frozen":true}', encoding="utf-8")
            (candidate / "tokenizer_config.json").write_text('{"chat_template":"base"}', encoding="utf-8")
            (candidate / "model-00001-of-00001.safetensors").write_bytes(b"weights")
            (candidate / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"model.embed_tokens.weight": "model-00001-of-00001.safetensors"}}),
                encoding="utf-8",
            )
            (candidate / "config.json").write_text(json.dumps(expected), encoding="utf-8")
            self.evaluation.validate_checkpoint_tree(candidate, expected, tokenizer_reference=base)

            changed = dict(expected, hidden_size=4096)
            (candidate / "config.json").write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hidden_size"):
                self.evaluation.validate_checkpoint_tree(candidate, expected, tokenizer_reference=base)

            malicious = dict(expected, auto_map={"AutoModel": "modeling_bad.BadModel"})
            (candidate / "config.json").write_text(json.dumps(malicious), encoding="utf-8")
            (candidate / "modeling_bad.py").write_text("raise RuntimeError('executed')\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "remote|executable"):
                self.evaluation.validate_checkpoint_tree(candidate, expected, tokenizer_reference=base)
            (candidate / "modeling_bad.py").unlink()

            (candidate / "config.json").write_text(json.dumps(expected), encoding="utf-8")
            (candidate / "tokenizer_config.json").write_text('{"chat_template":"changed"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tokenizer"):
                self.evaluation.validate_checkpoint_tree(candidate, expected, tokenizer_reference=base)


class HarborContainerTests(unittest.TestCase):
    def test_container_pins_runtime_code_and_every_huggingface_asset(self):
        dockerfile = ENVIRONMENT / "Dockerfile"
        if not dockerfile.is_file():
            self.fail("environment/Dockerfile is missing")
        text = dockerfile.read_text(encoding="utf-8")
        self.assertIn(
            "FROM slimerl/slime@sha256:2a6e9702dcb31657d8f498eba54cf161f15515eb500cf4d7591402441f7f6557",
            text,
        )
        contract = load_module("container_task_contract", SHARED / "task_contract.py")
        for revision in contract.ALL_ASSET_REVISIONS:
            with self.subTest(revision=revision):
                self.assertIn(revision, text)

    def test_agent_gets_only_algorithm_and_output_write_access(self):
        text = (ENVIRONMENT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("chown agent:agent /app/project/rl_research/algorithm_config.toml", text)
        self.assertIn("test ! -w /app/project/rl_research", text)
        self.assertIn("chmod -R a-w /opt/slime /opt/Search-R1 /opt/Megatron-LM /task-tools /datasets /retriever /models /app/project", text)
        self.assertIn("chmod 0700 /eval /models/reference", text)
        self.assertIn("USER agent", text)

    def test_docker_build_inputs_are_all_inside_harbor_environment_context(self):
        text = (ENVIRONMENT / "Dockerfile").read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("COPY ") and "--from=" not in line:
                source = line.split()[1 if not line.split()[1].startswith("--") else 2]
                self.assertNotIn("/", source, line)
                self.assertTrue((ENVIRONMENT / source).exists(), line)
        self.assertNotIn("COPY tests/", text)
        self.assertNotIn("COPY shared/", text)

    def test_hidden_eval_preparation_selects_256_nq_and_256_hotpot_rows_in_source_order(self):
        preparer = load_module("prepare_hidden_eval", ENVIRONMENT / "prepare_eval_data.py")
        rows = (
            [{"id": f"nq-{index}", "data_source": "nq"} for index in range(300)]
            + [{"id": f"other-{index}", "data_source": "triviaqa"} for index in range(20)]
            + [{"id": f"hotpot-{index}", "data_source": "hotpotqa"} for index in range(300)]
        )
        selected = preparer.select_eval_rows(rows)
        self.assertEqual([row["id"] for row in selected[:2]], ["nq-0", "nq-1"])
        self.assertEqual([row["id"] for row in selected[254:258]], ["nq-254", "nq-255", "hotpot-0", "hotpot-1"])
        self.assertEqual([row["id"] for row in selected[-2:]], ["hotpot-254", "hotpot-255"])
        with self.assertRaisesRegex(ValueError, "512"):
            preparer.select_eval_rows([{"id": index, "data_source": "nq"} for index in range(512)])

    def test_root_evaluator_uses_pinned_search_r1_and_owns_bm25_lifecycle(self):
        text = (TASK_DIR / "tests" / "evaluate.py").read_text(encoding="utf-8")
        self.assertIn("from generate_with_search import generate as search_r1_generate", text)
        self.assertIn("from generate_with_search import reward_func as search_r1_reward", text)
        self.assertIn("init_http_client(args)", text)
        self.assertIn("await close_http_client()", text)
        self.assertIn("build_bm25_command()", text)
        self.assertIn('"sampling_seed": SEED', text)
        self.assertIn("stage_candidate_checkpoint(", text)
        self.assertNotIn("trust_remote_code=True", text)


if __name__ == "__main__":
    unittest.main()
