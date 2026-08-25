import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from ast import literal_eval
from argparse import Namespace
from pathlib import Path
from unittest import mock

from autolab.tasks.magpie_data_pipeline.tests import task_contract
from autolab.tasks.magpie_data_pipeline.tests import dataset_contract
from autolab.tasks.magpie_data_pipeline.tests import evaluation_contract


FROZEN_LLAMA3_CHAT_TEMPLATE = (
    "{% set loop_messages = messages %}{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'"
    "+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}"
    "{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}"
)


def _parse_simple_toml(path):
    """Parse the scalar/table subset used by task metadata on Python 3.10."""

    document = {}
    section = document
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = document
            for part in line[1:-1].split("."):
                section = section.setdefault(part, {})
            continue
        key, value = (piece.strip() for piece in line.split("=", 1))
        section[key] = literal_eval(value.replace("true", "True").replace("false", "False"))
    return document


def _expected_source_manifest(root):
    files = []
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and not path.is_symlink():
            files.append(
                {
                    "bytes": path.stat().st_size,
                    "path": path.relative_to(root).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    manifest = {"files": files, "version": 1}
    digest = hashlib.sha256(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return manifest, digest


class AsyncLifecycleTests(unittest.TestCase):
    """The durable state machine must reject reuse and invalid finalization."""

    def setUp(self):
        from autolab.tasks.magpie_data_pipeline.environment import magpie_async_run

        self.runner = magpie_async_run
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.previous_root = self.runner.OUTPUT_ROOT
        self.runner.OUTPUT_ROOT = self.root

    def tearDown(self):
        self.runner.OUTPUT_ROOT = self.previous_root
        self.directory.cleanup()

    def test_attempt_ids_are_lowercase_slug_ids_and_cannot_be_reused(self):
        for value in ("attempt-1", "a" * 64, "a0-9"):
            self.runner.attempt_paths(value)
        for value in ("", "Attempt", "has_dot", "-leading", "a" * 65):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "attempt id"):
                self.runner.attempt_paths(value)

        args = Namespace(attempt_id="baseline-1", hypothesis="baseline")
        with mock.patch.object(self.runner.subprocess, "Popen", return_value=Namespace(pid=4567)):
            self.runner.submit(args)
        with self.assertRaisesRegex(RuntimeError, "immutable"):
            self.runner.submit(args)
        payload = json.loads((self.root / "attempts" / "baseline-1" / "status.json").read_text())
        self.assertEqual(payload["status"], "queued")
        self.assertNotIn("pid", payload)

    def test_submit_never_regresses_a_worker_running_status(self):
        args = Namespace(attempt_id="race-1", hypothesis="race")

        def worker_starts(*_args, **_kwargs):
            status_path = self.root / "attempts" / "race-1" / "status.json"
            self.runner.transition_status(status_path, "running", pid=9876)
            return Namespace(pid=9876)

        with mock.patch.object(self.runner.subprocess, "Popen", side_effect=worker_starts):
            self.runner.submit(args)
        payload = json.loads((self.root / "attempts" / "race-1" / "status.json").read_text())
        self.assertEqual(payload["status"], "running")
        self.assertEqual(payload["pid"], 9876)

    def test_atomic_status_reaches_only_legal_terminal_states(self):
        status_path = self.root / "attempts" / "run-1" / "status.json"
        self.runner.write_status(status_path, {"attempt_id": "run-1", "status": "queued"})
        self.runner.transition_status(status_path, "running")
        self.runner.transition_status(status_path, "completed")
        self.assertEqual(json.loads(status_path.read_text())["status"], "completed")
        self.assertFalse(list(status_path.parent.glob(".status.json.*.tmp")))
        with self.assertRaisesRegex(RuntimeError, "terminal"):
            self.runner.transition_status(status_path, "failed")

    def test_selection_requires_completed_attempt_and_control_requires_selection(self):
        with self.assertRaisesRegex(RuntimeError, "select a completed"):
            self.runner.complete_control(Namespace())
        root, status_path, _ = self.runner.attempt_paths("run-2")
        root.mkdir(parents=True)
        self.runner.write_status(status_path, {"attempt_id": "run-2", "status": "running"})
        with self.assertRaisesRegex(RuntimeError, "not completed"):
            self.runner.select(Namespace(attempt_id="run-2"))

    def test_selection_copies_dataset_manifest_and_appends_selected_ledger_event(self):
        root, status_path, _ = self.runner.attempt_paths("run-3")
        root.mkdir(parents=True)
        dataset = b'{"conversations":[{"from":"human","value":"question"},{"from":"gpt","value":"answer"}],"id":"row"}\n'
        (root / "dataset.jsonl").write_bytes(dataset)
        (root / "pipeline_config.toml").write_text("[pipeline]\n", encoding="utf-8")
        (root / "provenance.json").write_text(json.dumps({"hypothesis": "select me", "pipeline_config": {"seed": 42}}), encoding="utf-8")
        (root / "dataset_manifest.json").write_text(
            json.dumps({"records": 1, "tokens": 2, "dataset_sha256": hashlib.sha256(dataset).hexdigest()}), encoding="utf-8"
        )
        self.runner.write_status(status_path, {"attempt_id": "run-3", "status": "completed"})
        self.runner.select(Namespace(attempt_id="run-3"))
        self.assertTrue((self.root / "dataset_manifest.json").is_file())
        event = json.loads((self.root / "experiments.jsonl").read_text().splitlines()[-1])
        self.assertEqual(event["outcome"], "selected")
        self.assertTrue(event["selected"])


class PipelineAdapterTests(unittest.TestCase):
    """The editable hook is deterministic and its artifacts are self-contained."""

    def setUp(self):
        from autolab.tasks.magpie_data_pipeline.environment import generate_dataset, pipeline

        self.generate_dataset = generate_dataset
        self.pipeline = pipeline
        self.rows = [
            row(f"row-{index}", ("human", f"question {index}"), ("gpt", f"answer {index}"))
            for index in range(5)
        ]

    def test_starter_selects_ranked_baseline_rows_without_calling_generator(self):
        generator = _FakeGenerator()
        selected = list(
            self.pipeline.build_dataset(
                {"strategy": "baseline_prefix", "max_records": 3, "include_baseline": True, "generate_new": False, "seed": 42},
                self.rows,
                generator,
            )
        )
        expected_ids = [
            item["id"]
            for item in sorted(self.rows, key=lambda item: hashlib.sha256(item["id"].encode("utf-8")).hexdigest())[:3]
        ]
        self.assertEqual([item["id"] for item in selected], expected_ids)
        self.assertEqual(generator.calls, [])

    def test_local_70b_generator_uses_pinned_eight_way_vllm_completion_adapter(self):
        constructor_calls, sampling_calls = [], []

        class FakeSamplingParams:
            def __init__(self, **kwargs):
                sampling_calls.append(kwargs)

        class FakeEngine:
            def generate(self, prompts, parameters, **kwargs):
                self.call = (prompts, parameters, kwargs)
                return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=f"completion-{index}")]) for index, _ in enumerate(prompts)]

        engine = FakeEngine()

        class FakeLLM:
            def __new__(cls, **kwargs):
                constructor_calls.append(kwargs)
                return engine

        fake_vllm = types.SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams)
        with mock.patch.dict(sys.modules, {"vllm": fake_vllm}):
            adapter = self.generate_dataset.load_local_generator()
            answers = adapter.generate(["PROMPT-ONE", "PROMPT-TWO"], {"seed": 17, "max_tokens": 23})
        self.assertEqual(answers, ["completion-0", "completion-1"])
        self.assertNotIn("PROMPT", "".join(answers))
        self.assertEqual(
            constructor_calls,
            [{
                "model": task_contract.RUNTIME_PATHS["generator_model"],
                "tokenizer": task_contract.RUNTIME_PATHS["generator_model"],
                "tensor_parallel_size": 8,
                "trust_remote_code": False,
            }],
        )
        self.assertEqual(sampling_calls, [{"seed": 17, "max_tokens": 23}])
        self.assertEqual(engine.call[0], ["PROMPT-ONE", "PROMPT-TWO"])
        self.assertEqual(engine.call[2], {"use_tqdm": False})
        with self.assertRaisesRegex(ValueError, "seed|max_tokens"):
            adapter.generate(["prompt"], {"seed": 17, "max_new_tokens": 23})

    def test_baseline_loader_uses_the_frozen_runtime_path_without_an_override(self):
        """Changing the contract path must redirect the default baseline reader."""

        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.jsonl"
            expected = {"id": "baseline-row", "conversations": []}
            baseline.write_text(json.dumps(expected) + "\n", encoding="utf-8")
            previous_paths = self.generate_dataset.RUNTIME_PATHS
            self.generate_dataset.RUNTIME_PATHS = {**previous_paths, "baseline_dataset": str(baseline)}
            try:
                with mock.patch.dict(os.environ, {}, clear=True):
                    self.assertEqual(self.generate_dataset._load_baseline_rows(), [expected])
            finally:
                self.generate_dataset.RUNTIME_PATHS = previous_paths

    def test_adapter_writes_jsonl_and_companion_artifacts_with_generator_provenance(self):
        generator = _FakeGenerator(["generated answer one", "generated answer two"])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = self.generate_dataset.materialize_attempt(
                attempt_id="attempt-1",
                hypothesis="test the generator",
                output_dir=output,
                config={"strategy": "baseline_prefix", "max_records": 2, "include_baseline": True, "generate_new": True, "seed": 42},
                baseline_rows=self.rows,
                generator=generator,
                tokenizer=_FixedTokenizer(2),
                baseline_token_cap=100,
                chat_template=FROZEN_LLAMA3_CHAT_TEMPLATE,
            )
            expected = {"dataset.jsonl", "pipeline_config.toml", "provenance.json", "dataset_manifest.json", "status.json"}
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            self.assertEqual(result["status"], "completed")
            self.assertTrue(all(line.endswith(b"\n") for line in (output / "dataset.jsonl").read_bytes().splitlines(keepends=True)))
            provenance = json.loads((output / "provenance.json").read_text())
            self.assertEqual(provenance["generator_calls"], generator.calls)
            self.assertIn("pipeline_source_sha256", provenance)
            manifest = json.loads((output / "dataset_manifest.json").read_text())
            self.assertEqual(manifest["records"], 2)
            self.assertEqual(manifest["tokens"], 4)
            self.assertEqual(manifest["dataset_sha256"], provenance["dataset_sha256"])

    def test_adapter_enforces_injected_canonical_token_budget_before_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(Exception, "token budget"):
                self.generate_dataset.materialize_attempt(
                    attempt_id="attempt-3",
                    hypothesis="enforce canonical budget",
                    output_dir=Path(directory),
                    config={"strategy": "baseline_prefix", "max_records": 1, "include_baseline": True, "generate_new": False, "seed": 42},
                    baseline_rows=self.rows,
                    generator=_FakeGenerator(),
                    tokenizer=_FixedTokenizer(5),
                    baseline_token_cap=4,
                    chat_template=FROZEN_LLAMA3_CHAT_TEMPLATE,
                )

    def test_run_attempt_loads_editable_hook_and_config_from_injected_project_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data_research"
            root.mkdir()
            (root / "pipeline_config.toml").write_text(
                "[pipeline]\nstrategy = 'baseline_prefix'\nmax_records = 1\ninclude_baseline = true\ngenerate_new = false\nseed = 42\n",
                encoding="utf-8",
            )
            (root / "pipeline.py").write_text(
                "def build_dataset(config, baseline_rows, generator):\n    return list(baseline_rows)[:config['max_records']]\n",
                encoding="utf-8",
            )
            output = Path(directory) / "attempt"
            with mock.patch.object(self.generate_dataset, "_load_baseline_rows", return_value=self.rows), mock.patch.object(
                self.generate_dataset, "load_pinned_tokenizer", return_value=_FixedTokenizer(1)
            ), mock.patch.object(self.generate_dataset, "load_baseline_token_cap", return_value=100):
                with mock.patch.object(
                    self.generate_dataset,
                    "load_llama3_chat_template",
                    return_value=FROZEN_LLAMA3_CHAT_TEMPLATE,
                ):
                    self.generate_dataset.run_attempt("attempt-4", "editable hook", output, editable_root=root)
            provenance = json.loads((output / "provenance.json").read_text())
            self.assertEqual(provenance["editable_root"], str(root))
            self.assertEqual(provenance["pipeline_config"]["max_records"], 1)

    def test_run_attempt_preserves_config_bytes_and_hashes_every_editable_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            editable = root / "data_research"
            editable.mkdir()
            config_bytes = (
                b"# preserve this comment and ordering\n"
                b"[pipeline]\nstrategy = 'baseline_prefix'\nmax_records = 1\n"
                b"include_baseline = true\ngenerate_new = false\nseed = 42\n\n"
                b"[ranking]\nmetric = 'quality'\n"
            )
            (editable / "pipeline_config.toml").write_bytes(config_bytes)
            (editable / "helper.py").write_text("def choose(rows):\n    return list(rows)[:1]\n", encoding="utf-8")
            notes = editable / "nested"
            notes.mkdir()
            (notes / "rules.txt").write_text("keep exact bytes\n", encoding="utf-8")
            (editable / "pipeline.py").write_text(
                "from helper import choose\n"
                "def build_dataset(config, baseline_rows, generator):\n"
                "    return choose(baseline_rows)\n",
                encoding="utf-8",
            )
            expected_manifest, expected_tree_sha = _expected_source_manifest(editable)
            output = root / "attempt"
            with mock.patch.object(self.generate_dataset, "_load_baseline_rows", return_value=self.rows), mock.patch.object(
                self.generate_dataset, "load_pinned_tokenizer", return_value=_FixedTokenizer(1)
            ), mock.patch.object(self.generate_dataset, "load_baseline_token_cap", return_value=100), mock.patch.object(
                self.generate_dataset, "load_llama3_chat_template", return_value=FROZEN_LLAMA3_CHAT_TEMPLATE
            ):
                self.generate_dataset.run_attempt("attempt-tree", "bind every file", output, editable_root=editable)
            provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
            manifest = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual((output / "pipeline_config.toml").read_bytes(), config_bytes)
            self.assertEqual(provenance["data_research_manifest"], expected_manifest)
            self.assertEqual(provenance["data_research_sha256"], expected_tree_sha)
            self.assertEqual(manifest["data_research_sha256"], expected_tree_sha)
            self.assertEqual(provenance["pipeline_config_sha256"], hashlib.sha256(config_bytes).hexdigest())
            self.assertFalse(list(editable.rglob("*.pyc")))

    def test_editable_loader_rejects_support_config_and_ancestor_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write_editable(path):
                path.mkdir()
                (path / "pipeline.py").write_text("def build_dataset(*args):\n    return []\n", encoding="utf-8")
                (path / "pipeline_config.toml").write_text("[pipeline]\n", encoding="utf-8")

            support_root = root / "support-root"
            write_editable(support_root)
            outside_support = root / "outside.py"
            outside_support.write_text("SECRET = True\n", encoding="utf-8")
            (support_root / "support.py").symlink_to(outside_support)
            with self.assertRaisesRegex(ValueError, "symlink|ordinary"):
                self.generate_dataset._load_editable_pipeline(support_root)

            config_root = root / "config-root"
            write_editable(config_root)
            (config_root / "pipeline_config.toml").unlink()
            outside_config = root / "outside.toml"
            outside_config.write_text("[pipeline]\n", encoding="utf-8")
            (config_root / "pipeline_config.toml").symlink_to(outside_config)
            with self.assertRaisesRegex(ValueError, "symlink|ordinary"):
                self.generate_dataset._load_editable_pipeline(config_root)

            actual = root / "actual"
            write_editable(actual)
            linked = root / "linked"
            linked.symlink_to(actual, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink|ordinary"):
                self.generate_dataset._load_editable_pipeline(linked)

    def test_invalid_rows_fail_before_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaisesRegex(Exception, "conversations|dataset"):
                self.generate_dataset.materialize_attempt(
                    attempt_id="attempt-2",
                    hypothesis="reject invalid row",
                    output_dir=output,
                    config={"strategy": "baseline_prefix", "max_records": 5, "include_baseline": True, "generate_new": False, "seed": 42},
                    baseline_rows=[{"id": "bad", "conversations": []}],
                    generator=_FakeGenerator(),
                    tokenizer=_FixedTokenizer(1),
                    baseline_token_cap=100,
                    chat_template=FROZEN_LLAMA3_CHAT_TEMPLATE,
                )
            self.assertFalse((output / "dataset.jsonl").exists())


class _FakeGenerator:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    def generate(self, prompts, sampling):
        self.calls.append({"prompts": list(prompts), "sampling": dict(sampling)})
        return list(self.responses)


class _FixedTokenizer:
    def __init__(self, count):
        self.count = count

    def apply_chat_template(self, _messages, **_kwargs):
        return [0] * self.count


class ContractConstantsTests(unittest.TestCase):
    def test_frozen_assets_and_scientific_limits(self):
        self.assertEqual(
            dict(task_contract.ASSET_REVISIONS),
            {
                "magpie-align/magpie": "b734a36818e4ba7ef7f0f582fc55f7e860407d26",
                "meta-llama/Meta-Llama-3-70B-Instruct": "50fd307e57011801c7833c87efa1984ddf2db42f",
                "meta-llama/Meta-Llama-3-8B": "8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920",
                "Magpie-Align/Magpie-Pro-MT-300K-v0.1": "3c3045600d578b8f37413924b1b3bf117929755e",
                "Magpie-Align/Llama-3-8B-Magpie-Align-SFT-v0.1": "1ed587f54f70334f495efb9c027acb03e96fe24f",
                "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
                "Axolotl": "7c2bf3091f5e73c787afe839dfdcc8220b770a1a",
                "vLLM inference runtime": "643c125fab66d5ed5ec3143b7e764a77e7ae8ac7",
                "Transformers for Qwen 3.5": "943628458a1691f8af09c47ea9fc6e314734722f",
                "AlpacaEval code and prompts": "tatsu-lab/alpaca_eval@cd543a149df89434d8a54582c0151c0b945c3d20",
                "AlpacaEval 2 data": "tatsu-lab/alpaca_eval@2edc6fad8be6b14ea7230aabfd08188da6b8b814",
                "Arena-Hard code": "lmarena/arena-hard-auto@196f6b826783b3da7310e361a805fa36f0be83f3",
                "Arena-Hard data": "lmarena-ai/arena-hard-auto@15f3746e21432264ce9b453999bde4f3c946d2e6",
                "WildBench code": "allenai/WildBench@d6b8dcaf377d173d031980f97c16e1a82618c03d",
                "WildBench data": "allenai/WildBench@26c49eb39d7d5ce2099b0bbafed5a88dcce954ec",
            },
        )
        self.assertEqual(task_contract.MAX_RECORDS, 300_000)
        self.assertEqual(task_contract.SEQUENCE_LENGTH, 8_192)

    def test_eight_gpu_batch_contract(self):
        self.assertEqual(task_contract.accumulation_for_world_size(8), 4)
        self.assertEqual(8 * 1 * 4, 32)

    def test_non_dividing_world_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "global batch 32"):
            task_contract.accumulation_for_world_size(3)

    def test_frozen_training_and_judge_settings(self):
        contract = task_contract.CONTRACT
        self.assertEqual(contract.validation_fraction, 0.001)
        self.assertEqual(contract.optimizer, "paged_adamw_8bit")
        self.assertEqual(contract.learning_rate, 2e-5)
        self.assertEqual(contract.lr_scheduler, "cosine")
        self.assertEqual(contract.warmup_steps, 100)
        self.assertEqual(contract.weight_decay, 0.0)
        self.assertTrue(contract.bf16)
        self.assertTrue(contract.flash_attention)
        self.assertTrue(contract.assistant_response_only_loss)
        self.assertEqual(
            (
                contract.judge_temperature,
                contract.judge_top_p,
                contract.judge_top_k,
                contract.judge_min_p,
                contract.judge_presence_penalty,
                contract.judge_repetition_penalty,
                contract.judge_max_output_tokens,
            ),
            (1.0, 0.95, 20, 0.0, 1.5, 1.0, 32_768),
        )

    def test_flat_verifier_contracts_are_byte_for_byte_mirrors(self):
        task_root = Path(__file__).parents[1]
        for name in ("task_contract.py", "dataset_contract.py", "evaluation_contract.py"):
            with self.subTest(name=name):
                shared_hash = hashlib.sha256((task_root / "shared" / name).read_bytes()).hexdigest()
                verifier_hash = hashlib.sha256((task_root / "tests" / name).read_bytes()).hexdigest()
                self.assertEqual(verifier_hash, shared_hash)

    def test_frozen_llama3_template_source_hash_and_runtime_copies(self):
        task_root = Path(__file__).parents[1]
        source = task_root / "shared" / "llama3_chat_template.jinja"
        self.assertEqual(source.read_text(encoding="utf-8"), FROZEN_LLAMA3_CHAT_TEMPLATE + "\n")
        self.assertEqual(
            task_contract.load_llama3_chat_template(source),
            FROZEN_LLAMA3_CHAT_TEMPLATE,
        )
        self.assertEqual(
            task_contract.LLAMA3_CHAT_TEMPLATE_SHA256,
            hashlib.sha256(FROZEN_LLAMA3_CHAT_TEMPLATE.encode("utf-8")).hexdigest(),
        )
        dockerfile = (task_root / "environment" / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("COPY shared/llama3_chat_template.jinja /task-tools/llama3_chat_template.jinja", dockerfile)
        self.assertIn("COPY shared/llama3_chat_template.jinja /tests/llama3_chat_template.jinja", dockerfile)


class TinyTokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        return [0 for message in messages for _ in message["content"].split()]


def row(record_id, *messages):
    return {
        "id": record_id,
        "conversations": [
            {"from": role, "value": value}
            for role, value in messages
        ],
    }


def write_safetensors(path, tensors=None):
    """Write a stand-in file whose schema is supplied by the patched official reader."""

    tensors = tensors or {"model.embed_tokens.weight": ("F32", [1])}
    dtype = next(iter(tensors.values()))[0]
    Path(path).write_bytes(dtype.encode("ascii"))


class _FakeTensorSlice:
    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = shape

    def get_dtype(self):
        return self.dtype

    def get_shape(self):
        return self.shape


class _FakeSafeOpen:
    def __init__(self, schema):
        self.schema = schema

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def keys(self):
        return list(self.schema)

    def get_slice(self, name):
        dtype, shape = self.schema[name]
        return _FakeTensorSlice(dtype, shape)


class DatasetContractTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TinyTokenizer()

    def write_rows(self, directory, rows):
        path = Path(directory) / "dataset.jsonl"
        path.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) for item in rows) + "\n",
            encoding="utf-8",
        )
        return path

    def test_canonicalizes_valid_single_and_multi_turn_rows(self):
        raw = row(
            "e\u0301\r\nrecord",
            ("human", "hello\r\nthere"),
            ("gpt", "answer"),
            ("human", "follow up"),
            ("gpt", "final"),
        )
        self.assertEqual(
            dataset_contract.canonicalize_record(raw),
            row("é\nrecord", ("human", "hello\nthere"), ("gpt", "answer"), ("human", "follow up"), ("gpt", "final")),
        )

    def test_token_accounting_uses_frozen_template_when_base_tokenizer_has_none(self):
        class BaseTokenizerWithoutTemplate:
            chat_template = None

            def __init__(self):
                self.calls = []

            def apply_chat_template(self, messages, **kwargs):
                self.calls.append((messages, kwargs))
                return [1, 2, 3]

        tokenizer = BaseTokenizerWithoutTemplate()
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_rows(directory, [row("one", ("human", "hello"), ("gpt", "answer"))])
            summary = dataset_contract.validate_dataset(
                path,
                tokenizer,
                10,
                [],
                chat_template=FROZEN_LLAMA3_CHAT_TEMPLATE,
            )
        self.assertEqual(summary.tokens, 3)
        self.assertIsNone(tokenizer.chat_template)
        self.assertEqual(tokenizer.calls[0][1]["chat_template"], FROZEN_LLAMA3_CHAT_TEMPLATE)

    def test_canonicalization_rejects_unknown_fields_empty_values_bad_roles_and_nul(self):
        cases = [
            ({**row("one", ("human", "hello"), ("gpt", "answer")), "extra": True}, "unknown"),
            (row("one", ("human", ""), ("gpt", "answer")), "non-empty"),
            (row("one", ("gpt", "answer"), ("human", "hello")), "alternate"),
            (row("one", ("human", "unanswered")), "final gpt"),
            (row("one", ("human", "bad\0text"), ("gpt", "answer")), "NUL"),
        ]
        for raw, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(dataset_contract.DatasetValidationError, message):
                dataset_contract.canonicalize_record(raw)

    def test_validator_rejects_duplicate_ids_and_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            duplicate_path = self.write_rows(
                directory,
                [row("one", ("human", "hello"), ("gpt", "answer")), row("one", ("human", "again"), ("gpt", "again"))],
            )
            with self.assertRaisesRegex(dataset_contract.DatasetValidationError, "duplicate"):
                dataset_contract.validate_dataset(duplicate_path, self.tokenizer, 10, [], chat_template=FROZEN_LLAMA3_CHAT_TEMPLATE)
            link = Path(directory) / "link.jsonl"
            link.symlink_to(duplicate_path)
            with self.assertRaisesRegex(dataset_contract.DatasetValidationError, "ordinary file"):
                dataset_contract.validate_dataset(link, self.tokenizer, 10, [], chat_template=FROZEN_LLAMA3_CHAT_TEMPLATE)

    def test_validator_rejects_a_path_through_a_symlinked_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual = root / "actual"
            actual.mkdir()
            self.write_rows(actual, [row("one", ("human", "hello"), ("gpt", "answer"))])
            linked_directory = root / "linked"
            linked_directory.symlink_to(actual, target_is_directory=True)
            with self.assertRaisesRegex(dataset_contract.DatasetValidationError, "ordinary file"):
                dataset_contract.validate_dataset(linked_directory / "dataset.jsonl", self.tokenizer, 10, [], chat_template=FROZEN_LLAMA3_CHAT_TEMPLATE)

    def test_validator_rejects_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate-keys.jsonl"
            path.write_text(
                '{"id":"first","id":"second","conversations":[{"from":"human","value":"hello"},{"from":"gpt","value":"answer"}]}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(dataset_contract.DatasetValidationError, "duplicate JSON key"):
                dataset_contract.validate_dataset(path, self.tokenizer, 10, [], chat_template=FROZEN_LLAMA3_CHAT_TEMPLATE)

    def test_validator_enforces_record_and_token_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            overlong = self.write_rows(
                directory,
                [row("one", ("human", "x " * 8_192), ("gpt", "answer"))],
            )
            with self.assertRaisesRegex(dataset_contract.DatasetValidationError, "8192"):
                dataset_contract.validate_dataset(overlong, self.tokenizer, 20_000, [], chat_template=FROZEN_LLAMA3_CHAT_TEMPLATE)
            total_over_cap = self.write_rows(
                directory,
                [row("one", ("human", "one two"), ("gpt", "three")), row("two", ("human", "four five"), ("gpt", "six seven"))],
            )
            with self.assertRaisesRegex(dataset_contract.DatasetValidationError, "token budget"):
                dataset_contract.validate_dataset(total_over_cap, self.tokenizer, 6, [], chat_template=FROZEN_LLAMA3_CHAT_TEMPLATE)

    def test_validator_rejects_300001_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "too-many.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for index in range(300_001):
                    handle.write(
                        json.dumps(
                            row(str(index), ("human", "question"), ("gpt", "answer")),
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
            with self.assertRaisesRegex(dataset_contract.DatasetValidationError, "300000"):
                dataset_contract.validate_dataset(path, self.tokenizer, 1_000_000, [], chat_template=FROZEN_LLAMA3_CHAT_TEMPLATE)

    def test_validator_rejects_hidden_prompt_matches_and_high_13_gram_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            exact = self.write_rows(directory, [row("one", ("human", "Hidden\r\nPrompt"), ("gpt", "answer"))])
            with self.assertRaisesRegex(dataset_contract.DatasetValidationError, "hidden prompt"):
                dataset_contract.validate_dataset(exact, self.tokenizer, 10, ["Hidden\nPrompt"], chat_template=FROZEN_LLAMA3_CHAT_TEMPLATE)
            hidden = " ".join(f"token{i}" for i in range(20))
            overlap = " ".join([*(f"token{i}" for i in range(19)), "replacement"])
            contaminated = self.write_rows(directory, [row("two", ("human", overlap), ("gpt", "answer"))])
            with self.assertRaisesRegex(dataset_contract.DatasetValidationError, "13-gram"):
                dataset_contract.validate_dataset(contaminated, self.tokenizer, 100, [hidden], chat_template=FROZEN_LLAMA3_CHAT_TEMPLATE)

    def test_writer_sorts_rows_and_hashes_compact_utf8_json_lines(self):
        rows = [row("z", ("human", "hello"), ("gpt", "answer")), row("a", ("human", "hi"), ("gpt", "there"))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "canonical.jsonl"
            summary = dataset_contract.write_canonical_jsonl(path, rows, self.tokenizer, 10, [], chat_template=FROZEN_LLAMA3_CHAT_TEMPLATE)
            expected = (
                '{"conversations":[{"from":"human","value":"hi"},{"from":"gpt","value":"there"}],"id":"a"}\n'
                '{"conversations":[{"from":"human","value":"hello"},{"from":"gpt","value":"answer"}],"id":"z"}\n'
            ).encode("utf-8")
            self.assertEqual(path.read_bytes(), expected)
            self.assertEqual(summary.records, 2)
            self.assertEqual(summary.tokens, 4)
            self.assertEqual(summary.sha256, hashlib.sha256(expected).hexdigest())
            self.assertEqual(summary.duplicate_records, 0)


class EvaluationContractTests(unittest.TestCase):
    def test_prompt_selection_is_order_independent_and_exact(self):
        rows = [{"id": f"prompt-{index}", "source": "alpaca", "language": "en"} for index in range(260)]
        expected = [
            item["id"]
            for item in sorted(
                rows,
                key=lambda item: hashlib.sha256(f"alpaca\0{item['id']}".encode()).hexdigest(),
            )[:256]
        ]
        self.assertEqual(evaluation_contract.select_prompt_ids(list(reversed(rows)), "alpaca"), expected)

    def test_prompt_selection_requires_source_and_english_eligibility(self):
        with self.assertRaisesRegex(ValueError, "source exhausted"):
            evaluation_contract.select_prompt_ids([{"id": "missing-source", "language": "en"}], "alpaca", count=1)
        with self.assertRaisesRegex(ValueError, "source exhausted"):
            evaluation_contract.select_prompt_ids([{"id": "missing-language", "source": "alpaca"}], "alpaca", count=1)

    def test_candidate_label_normalizes_both_judge_orientations(self):
        self.assertEqual(evaluation_contract.normalize_verdict({"winner": "A"}, "A"), 1.0)
        self.assertEqual(evaluation_contract.normalize_verdict({"winner": "B"}, "A"), 0.0)
        self.assertEqual(evaluation_contract.normalize_verdict({"winner": "A"}, "B"), 0.0)
        self.assertEqual(evaluation_contract.normalize_verdict({"winner": "B"}, "B"), 1.0)
        self.assertEqual(evaluation_contract.normalize_verdict({"winner": "tie"}, "A"), 0.5)

    def test_orientation_combination_only_keeps_agreement(self):
        self.assertEqual(evaluation_contract.combine_orientations(1, 1), 1.0)
        self.assertEqual(evaluation_contract.combine_orientations(0, 0), 0.0)
        self.assertEqual(evaluation_contract.combine_orientations(0.5, 0.5), 0.5)
        for first, second in ((1, 0), (1, 0.5), (0, 1), (0, 0.5), (0.5, 1), (0.5, 0)):
            with self.subTest(first=first, second=second):
                self.assertEqual(evaluation_contract.combine_orientations(first, second), 0.5)

    def test_aggregate_reward_requires_exactly_768_finite_scores(self):
        self.assertEqual(evaluation_contract.aggregate_reward([0.5] * 768), 0.5)
        for scores in ([], [0.5] * 767, [0.5] * 767 + [float("nan")], [0.5] * 767 + [2.0]):
            with self.subTest(scores=len(scores)), self.assertRaisesRegex(ValueError, "768|finite"):
                evaluation_contract.aggregate_reward(scores)


class EvaluationPipelineTests(unittest.TestCase):
    """The root-only evaluator must be deterministic and fail closed on CPU fakes."""

    def test_generation_formats_standard_and_wildbench_prompts_and_rejects_coverage_gaps(self):
        from autolab.tasks.magpie_data_pipeline.tests import generate_responses

        tokenizer = _ChatTokenizer()
        normal = {"id": "alpaca-1", "source": "alpaca_eval_2", "prompt": "normal prompt"}
        wild = {
            "id": "wild-1",
            "source": "wildbench",
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "follow-up"},
            ],
        }
        self.assertEqual(
            generate_responses.format_prompt(tokenizer, normal, chat_template=FROZEN_LLAMA3_CHAT_TEMPLATE),
            "user:normal prompt|assistant:",
        )
        self.assertEqual(
            generate_responses.format_prompt(tokenizer, wild, chat_template=FROZEN_LLAMA3_CHAT_TEMPLATE),
            "user:first|assistant:answer|user:follow-up|assistant:",
        )
        self.assertIsNone(tokenizer.chat_template)
        self.assertTrue(all(call["add_generation_prompt"] for call in tokenizer.calls))
        self.assertTrue(all(call["chat_template"] == FROZEN_LLAMA3_CHAT_TEMPLATE for call in tokenizer.calls))
        implementation = Path(generate_responses.__file__).read_text(encoding="utf-8")
        self.assertIn("load_llama3_chat_template()", implementation)
        self.assertNotIn('getattr(tokenizer, "chat_template"', implementation)
        with self.assertRaisesRegex(ValueError, "missing|extra|768"):
            generate_responses.validate_response_coverage([{"id": "only-one", "response": "x"}], ["only-one", "missing"])

    def test_judge_uses_fixed_sampling_json_verdicts_and_same_seed_retries(self):
        from autolab.tasks.magpie_data_pipeline.tests import judge

        self.assertEqual(
            judge.JUDGE_SAMPLING,
            {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0, "max_tokens": 32768},
        )
        self.assertEqual(judge.parse_verdict('{"winner":"A"}'), (None, "A"))
        for raw in ('{"winner":"a"}', '{"winner":"A","why":"no"}', 'prefix {"winner":"A"}', '{"winner":"tie"} trailing'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                judge.parse_verdict(raw)
        seen = []

        def invalid_then_valid(_prompt, sampling, seed):
            seen.append((dict(sampling), seed))
            return "invalid" if len(seen) < 3 else '{"winner":"B"}'

        verdict = judge.request_verdict(invalid_then_valid, "judge prompt", "stable-id", "A")
        self.assertEqual(verdict, "B")
        self.assertEqual(len(seen), 3)
        self.assertEqual({seed for _sampling, seed in seen}, {judge.derive_seed("stable-id", "A")})

    def test_judge_extracts_thinking_before_a_strict_final_json_verdict(self):
        from autolab.tasks.magpie_data_pipeline.tests import judge

        raw = 'Compare factuality and relevance. The answer says {"winner":"A"} in its reasoning.</think>\n {"winner":"B"} '
        reasoning, winner = judge.parse_verdict(raw)
        self.assertEqual(reasoning, 'Compare factuality and relevance. The answer says {"winner":"A"} in its reasoning.')
        self.assertEqual(winner, "B")
        self.assertEqual(judge.parse_verdict('{"winner":"tie"}'), (None, "tie"))
        for malformed in (
            'missing close {"winner":"A"}',
            '</think>{"winner":"A"}',
            ' \n\t </think>{"winner":"A"}',
            'reason</think>{"winner":"A"} trailing',
            'one</think>two</think>{"winner":"A"}',
        ):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                judge.parse_verdict(malformed)

    def test_qwen_batch_adapter_enables_thinking_and_preserves_raw_final_output(self):
        from autolab.tasks.magpie_data_pipeline.tests import judge

        chat_calls, sampling_calls = [], []

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                chat_calls.append((messages, kwargs))
                return "rendered"

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(*_args, **_kwargs):
                return FakeTokenizer()

        class FakeSamplingParams:
            def __init__(self, **kwargs):
                sampling_calls.append(kwargs)

        class FakeEngine:
            def generate(self, prompts, parameters, **kwargs):
                self.prompts, self.parameters, self.kwargs = prompts, parameters, kwargs
                return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text='why</think>\n{"winner":"A"}')]) for _ in prompts]

        engine = FakeEngine()
        class FakeLLM:
            def __new__(cls, **_kwargs):
                return engine

        with mock.patch.dict(sys.modules, {"transformers": types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer), "vllm": types.SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams)}):
            adapter = judge._QwenBatchGenerator()
            raw = adapter.complete([("first", judge.JUDGE_SAMPLING, 101), ("second", judge.JUDGE_SAMPLING, 202)])
        self.assertEqual(raw, ['why</think>\n{"winner":"A"}'] * 2)
        self.assertEqual(judge.parse_verdict(raw[0]), ("why", "A"))
        self.assertTrue(all(call[1]["enable_thinking"] is True for call in chat_calls))
        self.assertEqual(sampling_calls, [{**judge.JUDGE_SAMPLING, "seed": 101}, {**judge.JUDGE_SAMPLING, "seed": 202}])
        self.assertEqual(engine.kwargs, {"use_tqdm": False})

    def test_score_writes_aggregate_only_artifacts_and_rejects_wrong_cardinality(self):
        from autolab.tasks.magpie_data_pipeline.tests import score

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reward_output = root / "logs" / "verifier"
            artifact_output = root / "output" / "verifier"
            reward = score.write_score(
                [{"id": str(index), "score": 1.0} for index in range(768)],
                reward_output,
                artifact_output,
            )
            self.assertEqual(reward, 1.0)
            self.assertEqual((reward_output / "reward.txt").read_text(), "1.0\n")
            self.assertEqual(json.loads((reward_output / "reward.json").read_text()), {"reward": 1.0})
            self.assertFalse((root / "output" / "reward.txt").exists())
            self.assertFalse((root / "output" / "reward.json").exists())
            details = json.loads((artifact_output / "score_details.json").read_text())
            self.assertEqual(details, {"count": 768, "losses": 0, "ties": 0, "wins": 768})
            with self.assertRaisesRegex(ValueError, "768"):
                score.write_score(
                    [{"id": str(index), "score": 0.5} for index in range(767)],
                    reward_output,
                    artifact_output,
                )

    def test_score_accepts_each_complete_outcome_and_rejects_non_numeric_scores(self):
        from autolab.tasks.magpie_data_pipeline.tests import score

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reward_output = root / "logs" / "verifier"
            artifact_output = root / "output" / "verifier"
            for value, expected in ((0.0, 0.0), (0.5, 0.5), (1.0, 1.0)):
                rows = [{"id": str(index), "score": value} for index in range(768)]
                with self.subTest(value=value):
                    self.assertEqual(score.write_score(rows, reward_output, artifact_output), expected)
            for value in (float("nan"), float("inf"), True):
                rows = [{"id": str(index), "score": value} for index in range(768)]
                with self.subTest(value=value), self.assertRaises(ValueError):
                    score.write_score(rows, reward_output, artifact_output)
            with self.assertRaisesRegex(ValueError, "768"):
                score.write_score(
                    [{"id": str(index), "score": 0.5} for index in range(769)],
                    reward_output,
                    artifact_output,
                )


class _ChatTokenizer:
    def __init__(self):
        self.calls = []
        self.chat_template = None

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(dict(kwargs))
        text = "|".join(f"{message['role']}:{message['content']}" for message in messages)
        return text + ("|assistant:" if kwargs.get("add_generation_prompt") else "")


class ShellFailClosedTests(unittest.TestCase):
    """The real verifier shell must zero rewards for stage and signal failures."""

    _STAGES = ("policy_check.py", "dataset_check.py", "train_candidate.py", "generate_responses.py", "judge.py", "score.py")

    def _fixture(self, root, *, broken_reward=False):
        root = Path(root)
        output, verifier, tests = root / "output", root / "output" / "verifier", root / "tests"
        output.mkdir()
        tests.mkdir()
        fake = root / "infer"
        log, started = root / "stages.log", root / "started"
        reward = root / "reward-is-a-file" if broken_reward else root / "logs" / "verifier"
        if broken_reward:
            reward.write_text("not a directory\n", encoding="utf-8")
        fake.write_text(
            "#!/usr/bin/env bash\n"
            "set -eu\n"
            "stage=$(basename \"$1\")\n"
            "printf '%s\\n' \"$stage\" >>\"$FAKE_LOG\"\n"
            "if [[ \"${BLOCK_STAGE:-}\" == \"$stage\" ]]; then touch \"$FAKE_STARTED\"; while :; do sleep 1; done; fi\n"
            "if [[ \"${FAIL_STAGE:-}\" == \"$stage\" ]]; then exit 17; fi\n"
            "if [[ \"$stage\" == score.py ]]; then printf '0.75\\n' >\"$FAKE_REWARD/reward.txt\"; printf '{\\\"reward\\\":0.75}\\n' >\"$FAKE_REWARD/reward.json\"; fi\n",
            encoding="utf-8",
        )
        os.chmod(fake, 0o755)
        script = (Path(__file__).parent / "test.sh").read_text(encoding="utf-8")
        replacements = {
            "/opt/infer-venv/bin/python": str(fake),
            "/app/output/verifier": str(verifier),
            "/app/output": str(output),
            "/app/project": str(root / "project"),
            "/opt/project": str(root / "clean"),
            "/logs/verifier": str(reward),
            "/tests/": str(tests) + "/",
        }
        for old, new in replacements.items():
            script = script.replace(old, new)
        runner = root / "test.sh"
        runner.write_text(script, encoding="utf-8")
        os.chmod(runner, 0o755)
        return runner, output, reward, log, started

    @staticmethod
    def _assert_zero(reward):
        assert (reward / "reward.txt").read_text() == "0.0\n"
        assert json.loads((reward / "reward.json").read_text()) == {"reward": 0.0}

    def test_only_harbor_reward_files_use_logs_and_diagnostics_stay_in_output(self):
        script = (Path(__file__).parent / "test.sh").read_text(encoding="utf-8")
        self.assertIn("REWARD_ROOT=/logs/verifier", script)
        self.assertIn('--reward-output "${REWARD_ROOT}"', script)
        self.assertIn('--artifact-output "${VERIFY}"', script)
        self.assertIn('--report "${VERIFY}/policy.json"', script)

    def test_each_stage_failure_writes_zero_reward(self):
        for stage in self._STAGES:
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                runner, output, reward, log, started = self._fixture(directory)
                reward.mkdir(parents=True)
                (reward / "reward.txt").write_text("0.9\n")
                (reward / "reward.json").write_text('{"reward":0.9}\n')
                environment = {**os.environ, "FAIL_STAGE": stage, "FAKE_REWARD": str(reward), "FAKE_LOG": str(log), "FAKE_STARTED": str(started)}
                result = subprocess.run([str(runner)], env=environment, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self._assert_zero(reward)
                self.assertFalse((output / "reward.txt").exists())
                self.assertFalse((output / "reward.json").exists())
                self.assertEqual(log.read_text().splitlines(), list(self._STAGES[: self._STAGES.index(stage) + 1]))

    def test_termination_writes_zero_and_success_does_not_overwrite_score(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, output, reward, log, started = self._fixture(directory)
            reward.mkdir(parents=True)
            (reward / "reward.txt").write_text("0.9\n")
            (reward / "reward.json").write_text('{"reward":0.9}\n')
            environment = {**os.environ, "BLOCK_STAGE": "policy_check.py", "FAKE_REWARD": str(reward), "FAKE_LOG": str(log), "FAKE_STARTED": str(started)}
            process = subprocess.Popen([str(runner)], env=environment, start_new_session=True)
            for _ in range(100):
                if started.exists():
                    break
                time.sleep(0.01)
            self.assertTrue(started.exists())
            os.killpg(process.pid, signal.SIGTERM)
            self.assertEqual(process.wait(timeout=5), 0)
            self._assert_zero(reward)

        with tempfile.TemporaryDirectory() as directory:
            runner, output, reward, log, started = self._fixture(directory)
            environment = {**os.environ, "FAKE_REWARD": str(reward), "FAKE_LOG": str(log), "FAKE_STARTED": str(started)}
            self.assertEqual(subprocess.run([str(runner)], env=environment, capture_output=True).returncode, 0)
            self.assertEqual((reward / "reward.txt").read_text(), "0.75\n")
            self.assertEqual(json.loads((reward / "reward.json").read_text()), {"reward": 0.75})
            self.assertFalse((output / "reward.txt").exists())
            self.assertFalse((output / "reward.json").exists())

    def test_reward_write_failure_is_the_only_nonzero_verifier_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, _output, reward, log, started = self._fixture(directory, broken_reward=True)
            environment = {**os.environ, "FAKE_REWARD": str(reward), "FAKE_LOG": str(log), "FAKE_STARTED": str(started)}
            result = subprocess.run([str(runner)], env=environment, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)


class HarborMetadataAndPolicyTests(unittest.TestCase):
    """Task packaging must expose the fixed agent/verifier boundary."""

    def _write_valid_output(self, output, policy_contract, project):
        dataset_path = output / "dataset.jsonl"
        summary = dataset_contract.write_canonical_jsonl(
            dataset_path,
            [row("candidate-1", ("human", "question"), ("gpt", "answer"))],
            _FixedTokenizer(2),
            100,
            [],
            chat_template=FROZEN_LLAMA3_CHAT_TEMPLATE,
        )
        config_sha = hashlib.sha256((project / "data_research" / "pipeline_config.toml").read_bytes()).hexdigest()
        source_sha = hashlib.sha256((project / "data_research" / "pipeline.py").read_bytes()).hexdigest()
        source_manifest, source_tree_sha = _expected_source_manifest(project / "data_research")
        (output / "dataset_manifest.json").write_text(
            json.dumps({"records": summary.records, "tokens": summary.tokens, "dataset_sha256": summary.sha256, "pipeline_source_sha256": source_sha, "pipeline_config_sha256": config_sha, "data_research_manifest": source_manifest, "data_research_sha256": source_tree_sha}),
            encoding="utf-8",
        )
        (output / "provenance.json").write_text(
            json.dumps({"attempt_id": "candidate-1", "asset_revisions": dict(task_contract.ASSET_REVISIONS), "pipeline_source_sha256": source_sha, "pipeline_config_sha256": config_sha, "data_research_manifest": source_manifest, "data_research_sha256": source_tree_sha, "command": ["/task-tools/magpie_async_run", "submit"], "generator_calls": [], "dataset_sha256": summary.sha256, "records": summary.records, "tokens": summary.tokens}),
            encoding="utf-8",
        )
        (output / "pipeline_config.toml").write_bytes((project / "data_research" / "pipeline_config.toml").read_bytes())
        (output / "experiments.jsonl").write_text(
            json.dumps({"attempt_id": "candidate-1", "outcome": "completed", "selected": False, "dataset_sha256": summary.sha256}) + "\n"
            + json.dumps({"attempt_id": "candidate-1", "outcome": "selected", "selected": True, "dataset_sha256": summary.sha256}) + "\n",
            encoding="utf-8",
        )
        (output / "submission-selection.json").write_text(json.dumps({"selected_attempt": "candidate-1", "status": "selected", "dataset_sha256": summary.sha256}), encoding="utf-8")
        (output / "control-complete.json").write_text(json.dumps({"status": "complete", "selected_attempt": "candidate-1", "dataset_sha256": summary.sha256}), encoding="utf-8")
        status = output / "attempts" / "candidate-1"
        status.mkdir(parents=True)
        (status / "status.json").write_text(json.dumps({"attempt_id": "candidate-1", "status": "completed", "dataset_sha256": summary.sha256}), encoding="utf-8")
        for name in ("dataset.jsonl", "dataset_manifest.json", "provenance.json", "pipeline_config.toml"):
            (status / name).write_bytes((output / name).read_bytes())

    def test_task_metadata_declares_the_frozen_harbor_resources(self):
        task_root = Path(__file__).parents[1]
        document = _parse_simple_toml(task_root / "task.toml")
        self.assertEqual(document["schema_version"], "1.1")
        self.assertEqual(document["task"]["name"], "autolab/magpie-data-pipeline")
        self.assertEqual(document["agent"]["timeout_sec"], 28_800)
        self.assertEqual(document["environment"]["build_timeout_sec"], 21_600)
        self.assertFalse(document["environment"]["allow_internet"])
        self.assertEqual(
            document["environment"].get("cpus"),
            64,
        )
        self.assertEqual(document["environment"]["memory_mb"], 524_288)
        self.assertEqual(document["environment"]["storage_mb"], 400_000)
        self.assertEqual(document["environment"]["gpus"], 8)
        self.assertEqual(document["environment"]["gpu_types"], ["H100"])
        self.assertEqual(document["verifier"]["timeout_sec"], 43_200)
        self.assertEqual(document["verifier"]["user"], "root")

    def test_metadata_and_policy_declare_every_pinned_asset_and_identical_boundaries(self):
        import yaml

        task_root = Path(__file__).parents[1]
        metadata = _parse_simple_toml(task_root / "task.toml")
        policy = yaml.safe_load((task_root / "policy.yaml").read_text(encoding="utf-8"))
        expected_pins = {
            "alpaca_eval_code": task_contract.ASSET_REVISIONS["AlpacaEval code and prompts"],
            "alpaca_eval_data": task_contract.ASSET_REVISIONS["AlpacaEval 2 data"],
            "arena_hard_code": task_contract.ASSET_REVISIONS["Arena-Hard code"],
            "arena_hard_data": task_contract.ASSET_REVISIONS["Arena-Hard data"],
            "wildbench_code": task_contract.ASSET_REVISIONS["WildBench code"],
            "wildbench_data": task_contract.ASSET_REVISIONS["WildBench data"],
        }
        for name, revision in expected_pins.items():
            self.assertIn(revision, metadata["metadata"]["asset_pins"][name])
        self.assertEqual(tuple(policy["scope"]["protected_paths"]), tuple(__import__("autolab.tasks.magpie_data_pipeline.shared.policy_contract", fromlist=["PROTECTED_PATHS"]).PROTECTED_PATHS))
        self.assertEqual(
            tuple(path.rsplit("/", 1)[-1] for path in policy["required_outputs"]),
            __import__("autolab.tasks.magpie_data_pipeline.shared.policy_contract", fromlist=["REQUIRED_OUTPUT_FILES"]).REQUIRED_OUTPUT_FILES,
        )
        self.assertIn(
            "pipeline_config.toml",
            __import__("autolab.tasks.magpie_data_pipeline.shared.policy_contract", fromlist=["REQUIRED_OUTPUT_FILES"]).REQUIRED_OUTPUT_FILES,
        )

    def test_policy_audit_allows_only_data_research_and_complete_regular_outputs(self):
        from autolab.tasks.magpie_data_pipeline.shared import policy_contract

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pristine = root / "pristine"
            candidate = root / "candidate"
            output = root / "output"
            for project in (pristine, candidate):
                (project / "data_research").mkdir(parents=True)
                (project / "data_research" / "pipeline.py").write_text("pass\n", encoding="utf-8")
                (project / "data_research" / "pipeline_config.toml").write_text("[pipeline]\n", encoding="utf-8")
                (project / "data_research" / "pipeline.py").write_text("frozen\n", encoding="utf-8")
                (project / "data_research" / "pipeline_config.toml").write_text("[pipeline]\n", encoding="utf-8")
                (project / "frozen.py").write_text("frozen\n", encoding="utf-8")
                (project / "data_research" / "support.py").write_text("VALUE = 1\n", encoding="utf-8")
            (candidate / "data_research" / "pipeline.py").write_text("changed\n", encoding="utf-8")
            output.mkdir()
            self._write_valid_output(output, policy_contract, candidate)
            self.assertEqual(policy_contract.audit_candidate(pristine, candidate, output, tokenizer=_FixedTokenizer(2), baseline_token_cap=100, chat_template=FROZEN_LLAMA3_CHAT_TEMPLATE), [])

            (candidate / "frozen.py").write_text("tampered\n", encoding="utf-8")
            errors = policy_contract.audit_candidate(pristine, candidate, output, tokenizer=_FixedTokenizer(2), baseline_token_cap=100, chat_template=FROZEN_LLAMA3_CHAT_TEMPLATE)
            self.assertIn("forbidden_modified:frozen.py", errors)

    def test_policy_binds_current_support_tree_not_only_pipeline_and_config(self):
        from autolab.tasks.magpie_data_pipeline.shared import policy_contract

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pristine, candidate, output = root / "pristine", root / "candidate", root / "output"
            for project in (pristine, candidate):
                (project / "data_research").mkdir(parents=True)
                (project / "data_research" / "pipeline.py").write_text("pass\n", encoding="utf-8")
                (project / "data_research" / "pipeline_config.toml").write_text("# exact\n[pipeline]\n", encoding="utf-8")
                (project / "data_research" / "support.py").write_text("VALUE = 1\n", encoding="utf-8")
            output.mkdir()
            self._write_valid_output(output, policy_contract, candidate)
            self.assertEqual(
                policy_contract.audit_candidate(pristine, candidate, output, tokenizer=_FixedTokenizer(2), baseline_token_cap=100, chat_template=FROZEN_LLAMA3_CHAT_TEMPLATE),
                [],
            )
            (candidate / "data_research" / "support.py").write_text("VALUE = 2\n", encoding="utf-8")
            errors = policy_contract.audit_candidate(pristine, candidate, output, tokenizer=_FixedTokenizer(2), baseline_token_cap=100, chat_template=FROZEN_LLAMA3_CHAT_TEMPLATE)
            self.assertIn("invalid_output:provenance.json", errors)

    def test_policy_requires_one_terminal_ledger_event_for_every_attempt_directory(self):
        from autolab.tasks.magpie_data_pipeline.shared import policy_contract

        def fixture(root):
            pristine, candidate, output = root / "pristine", root / "candidate", root / "output"
            for project in (pristine, candidate):
                (project / "data_research").mkdir(parents=True)
                (project / "data_research" / "pipeline.py").write_text("pass\n", encoding="utf-8")
                (project / "data_research" / "pipeline_config.toml").write_text("[pipeline]\n", encoding="utf-8")
            output.mkdir()
            self._write_valid_output(output, policy_contract, candidate)
            failed = output / "attempts" / "failed-control"
            failed.mkdir()
            (failed / "status.json").write_text(
                json.dumps({"attempt_id": "failed-control", "status": "failed"}),
                encoding="utf-8",
            )
            with (output / "experiments.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"attempt_id": "failed-control", "outcome": "failed", "selected": False}) + "\n")
            return pristine, candidate, output

        with tempfile.TemporaryDirectory() as directory:
            pristine, candidate, output = fixture(Path(directory))
            self.assertEqual(
                policy_contract.audit_candidate(pristine, candidate, output, tokenizer=_FixedTokenizer(2), baseline_token_cap=100, chat_template=FROZEN_LLAMA3_CHAT_TEMPLATE),
                [],
            )
            orphan = output / "attempts" / "orphan"
            orphan.mkdir()
            (orphan / "status.json").write_text(json.dumps({"attempt_id": "orphan", "status": "failed"}), encoding="utf-8")
            errors = policy_contract.audit_candidate(pristine, candidate, output, tokenizer=_FixedTokenizer(2), baseline_token_cap=100, chat_template=FROZEN_LLAMA3_CHAT_TEMPLATE)
            self.assertIn("invalid_output:experiments.jsonl", errors)

        with tempfile.TemporaryDirectory() as directory:
            pristine, candidate, output = fixture(Path(directory))
            with (output / "experiments.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"attempt_id": "ghost", "outcome": "failed", "selected": False}) + "\n")
            errors = policy_contract.audit_candidate(pristine, candidate, output, tokenizer=_FixedTokenizer(2), baseline_token_cap=100, chat_template=FROZEN_LLAMA3_CHAT_TEMPLATE)
            self.assertIn("invalid_output:experiments.jsonl", errors)

    def test_policy_rejects_attempts_ancestor_symlink_escape(self):
        from autolab.tasks.magpie_data_pipeline.shared import policy_contract

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pristine, candidate, output = root / "pristine", root / "candidate", root / "output"
            for project in (pristine, candidate):
                (project / "data_research").mkdir(parents=True)
                (project / "data_research" / "pipeline.py").write_text("pass\n", encoding="utf-8")
                (project / "data_research" / "pipeline_config.toml").write_text("[pipeline]\n", encoding="utf-8")
            output.mkdir()
            self._write_valid_output(output, policy_contract, candidate)
            outside = root / "outside-attempts"
            (output / "attempts").rename(outside)
            (output / "attempts").symlink_to(outside, target_is_directory=True)
            errors = policy_contract.audit_candidate(pristine, candidate, output, tokenizer=_FixedTokenizer(2), baseline_token_cap=100, chat_template=FROZEN_LLAMA3_CHAT_TEMPLATE)
            self.assertTrue(any(error.startswith("unsafe_output:attempts") for error in errors), errors)

    def test_policy_audit_rejects_source_escapes_and_unsafe_output_files(self):
        from autolab.tasks.magpie_data_pipeline.shared import policy_contract

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pristine = root / "pristine"
            candidate = root / "candidate"
            output = root / "output"
            for project in (pristine, candidate):
                (project / "data_research").mkdir(parents=True)
                (project / "data_research" / "pipeline.py").write_text("pass\n", encoding="utf-8")
                (project / "data_research" / "pipeline_config.toml").write_text("[pipeline]\n", encoding="utf-8")
            (candidate / "data_research" / "sitecustomize.py").write_text("x\n", encoding="utf-8")
            output.mkdir()
            for name in policy_contract.REQUIRED_OUTPUT_FILES:
                (output / name).write_text("{}\n", encoding="utf-8")
            (output / "dataset.jsonl").unlink()
            (output / "dataset.jsonl").symlink_to(output / "dataset_manifest.json")
            errors = policy_contract.audit_candidate(pristine, candidate, output, tokenizer=_FixedTokenizer(2), baseline_token_cap=100, chat_template=FROZEN_LLAMA3_CHAT_TEMPLATE)
            self.assertIn("forbidden_source:data_research/sitecustomize.py", errors)
            self.assertIn("unsafe_output:dataset.jsonl", errors)

    def test_policy_audit_rejects_empty_outputs_and_frozen_directory_type_changes(self):
        from autolab.tasks.magpie_data_pipeline.shared import policy_contract

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pristine = root / "pristine"
            candidate = root / "candidate"
            output = root / "output"
            for project in (pristine, candidate):
                (project / "data_research").mkdir(parents=True)
                (project / "frozen_dir").mkdir()
            (candidate / "frozen_dir").rmdir()
            (candidate / "frozen_dir").write_text("not a directory\n", encoding="utf-8")
            output.mkdir()
            for name in policy_contract.REQUIRED_OUTPUT_FILES:
                (output / name).write_text("{}\n", encoding="utf-8")
            errors = policy_contract.audit_candidate(pristine, candidate, output, tokenizer=_FixedTokenizer(2), baseline_token_cap=100, chat_template=FROZEN_LLAMA3_CHAT_TEMPLATE)
            self.assertIn("forbidden_type_changed:frozen_dir", errors)

    def test_policy_contract_is_a_flat_verifier_mirror(self):
        task_root = Path(__file__).parents[1]
        self.assertEqual(
            (task_root / "shared" / "policy_contract.py").read_bytes(),
            (task_root / "tests" / "policy_contract.py").read_bytes(),
        )


class AssetPreparationTests(unittest.TestCase):
    """Pinned local snapshots must produce immutable root-only public assets."""

    def test_prepare_assets_counts_canonical_baseline_tokens_and_selects_three_full_subsets(self):
        from autolab.tasks.magpie_data_pipeline.environment import prepare_assets

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.jsonl"
            baseline_rows = [row("base-1", ("human", "one two"), ("gpt", "three")), row("base-2", ("human", "four"), ("gpt", "five six"))]
            baseline.write_text("".join(json.dumps(item) + "\n" for item in baseline_rows), encoding="utf-8")
            sources = {}
            for source in prepare_assets.EVALUATION_SOURCES:
                path = root / f"{source}.jsonl"
                rows = [
                    {"id": f"{source}-{index}", "source": source, "language": "en", **({"messages": [{"role": "user", "content": f"prompt {index}"}]} if source == "wildbench" else {"prompt": f"prompt {index}"})}
                    for index in range(260)
                ]
                path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
                sources[source] = path
            output = root / "assets"
            result = prepare_assets.prepare_assets(
                output_dir=output,
                baseline_path=baseline,
                source_paths=sources,
                tokenizer=_FixedTokenizer(3),
                chat_template=FROZEN_LLAMA3_CHAT_TEMPLATE,
            )
            budget = json.loads((output / "baseline_budget.json").read_text(encoding="utf-8"))
            manifest = json.loads((output / "eval_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(budget["canonical_tokens"], 6)
            self.assertEqual(result["canonical_tokens"], 6)
            self.assertEqual(set(manifest["sources"]), set(prepare_assets.EVALUATION_SOURCES))
            for entries in manifest["sources"].values():
                self.assertEqual(len(entries), 256)
                self.assertEqual(len({entry["id"] for entry in entries}), 256)
            self.assertEqual((output / "baseline_budget.json").stat().st_mode & 0o777, 0o444)
            self.assertEqual((output / "eval_manifest.json").stat().st_mode & 0o777, 0o600)

    def test_prepare_assets_fails_when_a_source_lacks_eligible_rows_or_existing_asset_differs(self):
        from autolab.tasks.magpie_data_pipeline.environment import prepare_assets

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.jsonl"
            baseline.write_text(json.dumps(row("base", ("human", "one"), ("gpt", "two"))) + "\n", encoding="utf-8")
            sources = {}
            for source in prepare_assets.EVALUATION_SOURCES:
                path = root / f"{source}.jsonl"
                count = 255 if source == prepare_assets.EVALUATION_SOURCES[-1] else 256
                rows = [
                    {"id": f"{source}-{index}", "source": source, "language": "en", **({"messages": [{"role": "user", "content": "p"}]} if source == "wildbench" else {"prompt": "p"})}
                    for index in range(count)
                ]
                path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
                sources[source] = path
            with self.assertRaisesRegex(ValueError, "source exhausted"):
                prepare_assets.prepare_assets(root / "assets", baseline, sources, _FixedTokenizer(1), FROZEN_LLAMA3_CHAT_TEMPLATE)

            output = root / "assets"
            output.mkdir(exist_ok=True)
            (output / "baseline_budget.json").write_text('{"wrong":true}\n', encoding="utf-8")
            sources[prepare_assets.EVALUATION_SOURCES[-1]].write_text(
                "".join(json.dumps({"id": f"fixed-{index}", "source": prepare_assets.EVALUATION_SOURCES[-1], "language": "en", "messages": [{"role": "user", "content": "p"}]}) + "\n" for index in range(256)),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "mismatched"):
                prepare_assets.prepare_assets(output, baseline, sources, _FixedTokenizer(1), FROZEN_LLAMA3_CHAT_TEMPLATE)

    def test_staging_converts_real_source_shapes_and_preserves_wildbench_user_history(self):
        from autolab.tasks.magpie_data_pipeline.environment import stage_snapshots

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = {
                "baseline": [{"uuid": "base-1", "messages": [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]}],
                "alpaca_eval_2": [{"instruction": "write a poem", "input": "about rain"}],
                "arena_hard": [{"uid": "arena-1", "prompt": "solve this"}],
                "wildbench": [{"session_id": "wild-1", "conversation_input": [{"role": "user", "content": "first"}, {"role": "assistant", "content": "reply"}, {"role": "user", "content": "follow up"}]}],
            }
            stage_snapshots.stage_rows(raw, root)
            baseline = [json.loads(line) for line in (root / "snapshots" / "magpie-baseline.jsonl").read_text().splitlines()]
            wild = [json.loads(line) for line in (root / "snapshots" / "wildbench.jsonl").read_text().splitlines()]
            self.assertEqual(baseline[0], row("base-1", ("human", "question"), ("gpt", "answer")))
            self.assertEqual(wild[0]["messages"], [{"role": "user", "content": "first"}, {"role": "assistant", "content": "reply"}, {"role": "user", "content": "follow up"}])
            self.assertTrue((root / "Magpie-Pro-MT-300K-v0.1.jsonl").is_file())

    def test_staging_rejects_ambiguous_directory_instead_of_recursing_repo_jsonl(self):
        from autolab.tasks.magpie_data_pipeline.environment import stage_snapshots
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "one.jsonl").write_text('{"uid":"one","prompt":"a"}\n')
            (root / "two.jsonl").write_text('{"uid":"two","prompt":"b"}\n')
            with self.assertRaisesRegex(ValueError, "explicit file"):
                stage_snapshots.read_rows_file(root)

    def test_staging_dispatches_parquet_before_attempting_utf8_jsonl(self):
        from autolab.tasks.magpie_data_pipeline.environment import stage_snapshots
        import types
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.parquet"
            path.write_bytes(b"not utf8 parquet fixture")
            parquet = types.ModuleType("pyarrow.parquet")
            parquet.read_table = mock.Mock(return_value=mock.Mock(to_pylist=lambda: [{"id": "x"}]))
            with mock.patch.dict("sys.modules", {"pyarrow": types.ModuleType("pyarrow"), "pyarrow.parquet": parquet}):
                self.assertEqual(stage_snapshots.read_rows_file(path), [{"id": "x"}])

    def test_prepared_wildbench_manifest_keeps_history_instead_of_a_flattened_prompt(self):
        from autolab.tasks.magpie_data_pipeline import environment
        from autolab.tasks.magpie_data_pipeline.environment import prepare_assets

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.jsonl"
            baseline.write_text(json.dumps(row("base", ("human", "q"), ("gpt", "a"))) + "\n", encoding="utf-8")
            sources = {}
            for source in prepare_assets.EVALUATION_SOURCES:
                entries = [
                    {"id": f"{source}-{index}", "language": "en", **({"messages": [{"role": "user", "content": "one turn"}]} if source == "wildbench" else {"prompt": "one turn"})}
                    for index in range(256)
                ]
                if source == "wildbench":
                    entries[0] = {"id": "wild-history", "language": "en", "messages": [{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"}, {"role": "user", "content": "three"}]}
                path = root / f"{source}.jsonl"
                path.write_text("".join(json.dumps(item) + "\n" for item in entries), encoding="utf-8")
                sources[source] = path
            prepare_assets.prepare_assets(root / "assets", baseline, sources, _FixedTokenizer(1), FROZEN_LLAMA3_CHAT_TEMPLATE)
            entries = json.loads((root / "assets" / "eval_manifest.json").read_text())["sources"]["wildbench"]
            history = next(item for item in entries if item["id"] == "wild-history")
            self.assertEqual(history["messages"][-1]["content"], "three")
            self.assertNotIn("prompt", history)


class HarborContainerTests(unittest.TestCase):
    """The image build order must yield runnable, staged offline assets."""

    def test_dockerfile_stages_snapshots_before_runtime_offline_and_installs_absolute_launcher(self):
        dockerfile = (Path(__file__).parents[1] / "environment" / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("stage_snapshots.py", dockerfile)
        self.assertIn("COPY --chmod=755 environment/magpie_async_run.sh /task-tools/magpie_async_run", dockerfile)
        self.assertLess(dockerfile.index("huggingface-cli download"), dockerfile.rindex("HF_HUB_OFFLINE=1"))
        self.assertIn("/opt/infer-venv/bin/python /task-tools/magpie_async_run.py", (Path(__file__).parents[1] / "environment" / "magpie_async_run.sh").read_text(encoding="utf-8"))

    def test_uid_1000_can_read_only_public_runtime_assets(self):
        dockerfile = (Path(__file__).parents[1] / "environment" / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("useradd --create-home --uid 1000 agent", dockerfile)
        self.assertIn("chmod 0444 /task-assets/baseline_budget.json", dockerfile)
        self.assertIn("chmod 0600 /task-assets/eval_manifest.json", dockerfile)
        self.assertIn("/task-assets/raw /task-assets/snapshots /task-assets/magpie", dockerfile)
        permission_probe = dockerfile[dockerfile.index("runuser -u agent --") :]
        for public in (
            "/models/Meta-Llama-3-8B/config.json",
            "/models/Meta-Llama-3-70B-Instruct/config.json",
            "/task-assets/Magpie-Pro-MT-300K-v0.1.jsonl",
            "/task-assets/baseline_budget.json",
        ):
            with self.subTest(public=public):
                self.assertIn(f"test -r {public}", permission_probe)
        for hidden in (
            "/task-assets/eval_manifest.json",
            "/tests/test.sh",
            "/models/Qwen3.5-9B/config.json",
            "/models/Llama-3-8B-Magpie-Align-SFT-v0.1/config.json",
        ):
            with self.subTest(hidden=hidden):
                self.assertIn(f"test ! -r {hidden}", permission_probe)
        for removed in ("/task-assets/raw", "/task-assets/snapshots", "/task-assets/magpie"):
            with self.subTest(removed=removed):
                self.assertIn(f"test ! -e {removed}", permission_probe)


class FrozenTrainingVerifierTests(unittest.TestCase):
    """The root-side training path may vary only dataset and output locations."""

    def setUp(self):
        from autolab.tasks.magpie_data_pipeline.tests import train_candidate

        self.train_candidate = train_candidate
        self.reader = mock.patch.object(train_candidate, "_safe_open", side_effect=self._safe_open, create=True).start()
        self.addCleanup(self.reader.stop)

    @staticmethod
    def _safe_open(path):
        payload = Path(path).read_bytes()
        if payload == b"CORRUPT":
            raise OSError("truncated or overlapping safetensors payload")
        dtype = "F16" if b"F16" in payload else "F32"
        return _FakeSafeOpen({"model.embed_tokens.weight": (dtype, [1])})

    def test_frozen_recipe_has_every_scientific_setting_and_no_adapter_algorithms(self):
        import yaml

        recipe = (Path(__file__).parents[1] / "environment" / "frozen_train.yaml").read_text(encoding="utf-8")
        parsed = yaml.safe_load(recipe)
        expected = {
            "base_model": "/models/Meta-Llama-3-8B",
            "model_type": "LlamaForCausalLM",
            "tokenizer_type": "AutoTokenizer",
            "load_in_8bit": False,
            "load_in_4bit": False,
            "strict": False,
            "chat_template": "llama3",
            "sequence_len": 8192,
            "sample_packing": True,
            "eval_sample_packing": False,
            "val_set_size": 0.001,
            "num_epochs": 2,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "optimizer": "paged_adamw_8bit",
            "learning_rate": 0.00002,
            "lr_scheduler": "cosine",
            "warmup_steps": 100,
            "weight_decay": 0.0,
            "bf16": "auto",
            "tf32": False,
            "flash_attention": True,
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "train_on_inputs": False,
            "group_by_length": False,
            "evals_per_epoch": 5,
            "saves_per_epoch": 1,
            "seed": 42,
            "save_safetensors": True,
        }
        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertEqual(parsed[key], value)
        self.assertEqual(parsed["datasets"], [{"path": "__CANONICAL_DATASET__", "type": "sharegpt", "conversation": "llama3"}])
        self.assertEqual(parsed["special_tokens"], {"pad_token": "<|end_of_text|>"})
        self.assertNotIn("conversation", parsed)
        self.assertNotIn("pad_token", parsed)
        self.assertNotRegex(recipe.casefold(), r"\b(lora|qlora|dpo|orpo|adapter)\b")

    def test_effective_recipe_only_substitutes_canonical_dataset_and_output(self):
        from autolab.tasks.magpie_data_pipeline.tests import train_candidate

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "canonical.jsonl"
            source.write_text('{"id":"x"}\n', encoding="utf-8")
            output = root / "hf_model"
            effective = train_candidate.write_effective_recipe(source, output, root / "effective.yaml")
            rendered = effective.read_text(encoding="utf-8")
            self.assertIn(str(source), rendered)
            self.assertIn(str(output), rendered)
            self.assertNotIn("__CANONICAL_DATASET__", rendered)
            self.assertNotIn("__VERIFIER_OUTPUT__", rendered)
            self.assertEqual(train_candidate.training_command(effective), [
                "/opt/train-venv/bin/accelerate", "launch", "--num_processes", "8", "-m", "axolotl.cli.train", str(effective),
            ])

    def test_cli_rejects_scientific_overrides(self):
        from autolab.tasks.magpie_data_pipeline.tests import train_candidate

        with self.assertRaises(SystemExit):
            train_candidate.parse_args(["--dataset", "/tmp/data", "--output", "/tmp/model", "--seed", "0"])

    def test_dataset_check_writes_regular_canonical_file_and_summary(self):
        from autolab.tasks.magpie_data_pipeline.tests import dataset_check

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "dataset.jsonl"
            source.write_text(
                json.dumps(row("z", ("human", "question"), ("gpt", "answer"))) + "\n"
                + json.dumps(row("a", ("human", "other"), ("gpt", "reply"))) + "\n",
                encoding="utf-8",
            )
            destination = root / "verifier" / "canonical.jsonl"
            summary = dataset_check.canonicalize_dataset(source, destination, tokenizer=TinyTokenizer(), baseline_token_cap=100, hidden_prompts=[], chat_template=FROZEN_LLAMA3_CHAT_TEMPLATE)
            self.assertEqual(summary.records, 2)
            self.assertFalse(destination.is_symlink())
            self.assertTrue(destination.is_file())
            self.assertEqual([json.loads(line)["id"] for line in destination.read_text().splitlines()], ["a", "z"])
            payload = json.loads((destination.parent / "canonical_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["dataset_sha256"], summary.sha256)
            self.assertEqual(payload["records"], 2)

    def test_malformed_dataset_never_reaches_training_command(self):
        from autolab.tasks.magpie_data_pipeline.tests import dataset_check, train_candidate

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bad.jsonl"
            source.write_text("not-json\n", encoding="utf-8")
            with mock.patch.object(train_candidate.subprocess, "run") as launch:
                with self.assertRaises(dataset_check.DatasetValidationError):
                    dataset_check.canonicalize_dataset(source, root / "canonical.jsonl", tokenizer=TinyTokenizer(), baseline_token_cap=10, hidden_prompts=[], chat_template=FROZEN_LLAMA3_CHAT_TEMPLATE)
                launch.assert_not_called()

    def test_dataset_check_rejects_precreated_nonprivate_or_symlinked_verifier_directory(self):
        from autolab.tasks.magpie_data_pipeline.tests import dataset_check

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "dataset.jsonl"
            source.write_text(json.dumps(row("one", ("human", "question"), ("gpt", "answer"))) + "\n", encoding="utf-8")
            verifier = root / "verifier"
            verifier.mkdir(mode=0o755)
            os.chmod(verifier, 0o755)
            with self.assertRaisesRegex(dataset_check.DatasetValidationError, "private"):
                dataset_check.canonicalize_dataset(source, verifier / "canonical.jsonl", tokenizer=TinyTokenizer(), baseline_token_cap=100, hidden_prompts=[], chat_template=FROZEN_LLAMA3_CHAT_TEMPLATE)
            verifier.rmdir()
            target = root / "private"
            target.mkdir(mode=0o700)
            os.chmod(target, 0o700)
            verifier.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(dataset_check.DatasetValidationError, "ordinary"):
                dataset_check.canonicalize_dataset(source, verifier / "canonical.jsonl", tokenizer=TinyTokenizer(), baseline_token_cap=100, hidden_prompts=[], chat_template=FROZEN_LLAMA3_CHAT_TEMPLATE)

    def test_checkpoint_requires_base_architecture_tokenizers_and_shards(self):
        from autolab.tasks.magpie_data_pipeline.tests import train_candidate

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, output = root / "base", root / "output"
            base.mkdir()
            output.mkdir()
            config = {"architectures": ["LlamaForCausalLM"], "hidden_size": 64, "rope_theta": 500000.0, "vocab_size": 128}
            for location in (base, output):
                (location / "config.json").write_text(json.dumps(config), encoding="utf-8")
                (location / "tokenizer.json").write_text("tokenizer", encoding="utf-8")
                (location / "tokenizer_config.json").write_text("config", encoding="utf-8")
                (location / "special_tokens_map.json").write_text("special", encoding="utf-8")
            write_safetensors(base / "model.safetensors")
            write_safetensors(output / "model.safetensors")
            checkpoint = output / "checkpoint-100"
            checkpoint.mkdir()
            (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 100, "log_history": [{"loss": 1.25}]}), encoding="utf-8")
            checked = train_candidate.validate_checkpoint(output, base_model=base)
            self.assertEqual(checked["loss"], 1.25)
            (output / "model.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "shard"):
                train_candidate.validate_checkpoint(output, base_model=base)

    def test_checkpoint_rejects_bin_weights_and_tensor_mismatch(self):
        from autolab.tasks.magpie_data_pipeline.tests import train_candidate

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, output = root / "base", root / "output"
            base.mkdir()
            output.mkdir()
            config = {"architectures": ["LlamaForCausalLM"], "hidden_size": 64, "rope_theta": 500000.0, "vocab_size": 128}
            for location in (base, output):
                (location / "config.json").write_text(json.dumps(config), encoding="utf-8")
                for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
                    (location / name).write_text(name, encoding="utf-8")
            write_safetensors(base / "model.safetensors")
            write_safetensors(output / "model.safetensors")
            checkpoint = output / "checkpoint-1"
            checkpoint.mkdir()
            (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 1, "train_loss": 1.0}), encoding="utf-8")
            (output / "pytorch_model.bin").write_bytes(b"forbidden")
            with self.assertRaisesRegex(ValueError, "weight"):
                train_candidate.validate_checkpoint(output, base_model=base)
            (output / "pytorch_model.bin").unlink()
            write_safetensors(output / "model.safetensors", {"model.embed_tokens.weight": ("F16", [1])})
            with self.assertRaisesRegex(ValueError, "tensor"):
                train_candidate.validate_checkpoint(output, base_model=base)

    def test_checkpoint_rejects_unindexed_extra_safetensors_weight_file(self):
        from autolab.tasks.magpie_data_pipeline.tests import train_candidate

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, output = root / "base", root / "output"
            base.mkdir()
            output.mkdir()
            config = {"architectures": ["LlamaForCausalLM"], "hidden_size": 64, "vocab_size": 128}
            for location in (base, output):
                (location / "config.json").write_text(json.dumps(config), encoding="utf-8")
                for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
                    (location / name).write_text(name, encoding="utf-8")
                write_safetensors(location / "model.safetensors")
            checkpoint = output / "checkpoint-1"
            checkpoint.mkdir()
            (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 1, "train_loss": 1.0}), encoding="utf-8")
            write_safetensors(output / "unexpected.safetensors")
            with self.assertRaisesRegex(ValueError, "unexpected"):
                train_candidate.validate_checkpoint(output, base_model=base)

    def test_checkpoint_fails_closed_when_official_reader_rejects_corrupt_shard(self):
        from autolab.tasks.magpie_data_pipeline.tests import train_candidate

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, output = root / "base", root / "output"
            base.mkdir()
            output.mkdir()
            config = {"architectures": ["LlamaForCausalLM"], "hidden_size": 64, "vocab_size": 128}
            for location in (base, output):
                (location / "config.json").write_text(json.dumps(config), encoding="utf-8")
                for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
                    (location / name).write_text(name, encoding="utf-8")
                write_safetensors(location / "model.safetensors")
            checkpoint = output / "checkpoint-1"
            checkpoint.mkdir()
            (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 1, "train_loss": 1.0}), encoding="utf-8")
            (output / "model.safetensors").write_bytes(b"CORRUPT")
            for failure in ("truncated payload", "overlapping offsets", "invalid shard metadata"):
                with self.subTest(failure=failure):
                    with mock.patch.object(train_candidate, "_safe_open", side_effect=OSError(failure), create=True):
                        with self.assertRaisesRegex(ValueError, "official safetensors reader"):
                            train_candidate.validate_checkpoint(output, base_model=base)

    def test_training_uses_only_absolute_frozen_accelerate_and_clears_gpu_mask(self):
        from autolab.tasks.magpie_data_pipeline.tests import train_candidate

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, base, output = root / "canonical.jsonl", root / "base", root / "hf_model"
            dataset.write_text(json.dumps(row("one", ("human", "q"), ("gpt", "a"))) + "\n", encoding="utf-8")
            base.mkdir()
            config = {"architectures": ["LlamaForCausalLM"], "hidden_size": 64, "rope_theta": 500000.0, "vocab_size": 128}
            for location in (base,):
                (location / "config.json").write_text(json.dumps(config), encoding="utf-8")
                for name, content in (("tokenizer.json", "tokenizer"), ("tokenizer_config.json", "config"), ("special_tokens_map.json", "special")):
                    (location / name).write_text(content, encoding="utf-8")
            write_safetensors(base / "model.safetensors")

            def export_checkpoint(command, *, check, env):
                self.assertTrue(check)
                self.assertEqual(command[:6], ["/opt/train-venv/bin/accelerate", "launch", "--num_processes", "8", "-m", "axolotl.cli.train"])
                self.assertNotIn("CUDA_VISIBLE_DEVICES", env)
                output.mkdir()
                (output / "config.json").write_text(json.dumps(config), encoding="utf-8")
                for name, content in (("tokenizer.json", "tokenizer"), ("tokenizer_config.json", "config"), ("special_tokens_map.json", "special")):
                    (output / name).write_text(content, encoding="utf-8")
                write_safetensors(output / "model.safetensors")
                checkpoint = output / "checkpoint-1"
                checkpoint.mkdir()
                (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 1, "train_loss": 0.75}), encoding="utf-8")

            with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}, clear=False):
                with mock.patch.object(train_candidate.subprocess, "run", side_effect=export_checkpoint) as launch:
                    summary = train_candidate.train_candidate(dataset, output, base_model=base)
            launch.assert_called_once()
            self.assertEqual(summary["loss"], 0.75)
            self.assertTrue((root / "train_summary.json").is_file())

    def test_training_rejects_dangling_output_symlink_and_nonprivate_parent_before_launch(self):
        from autolab.tasks.magpie_data_pipeline.tests import train_candidate

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "canonical.jsonl"
            dataset.write_text(json.dumps(row("one", ("human", "q"), ("gpt", "a"))) + "\n", encoding="utf-8")
            verifier = root / "verifier"
            verifier.mkdir(mode=0o700)
            os.chmod(verifier, 0o700)
            output = verifier / "hf_model"
            output.symlink_to(root / "absent-model", target_is_directory=True)
            with mock.patch.object(train_candidate.subprocess, "run") as launch:
                with self.assertRaisesRegex(ValueError, "symlink"):
                    train_candidate.train_candidate(dataset, output, base_model=root)
                launch.assert_not_called()
            output.unlink()
            os.chmod(verifier, 0o755)
            with mock.patch.object(train_candidate.subprocess, "run") as launch:
                with self.assertRaisesRegex(ValueError, "private"):
                    train_candidate.train_candidate(dataset, output, base_model=root)
                launch.assert_not_called()


class ReferencePathClosureTests(unittest.TestCase):
    """The published reference must exercise the documented same-session path."""

    def test_reference_path_is_complete_and_uses_only_the_local_lifecycle(self):
        task_root = Path(__file__).parents[1]
        solution = task_root / "solution"
        solve = solution / "solve.sh"
        reference = solution / "reference.md"

        self.assertTrue(solve.is_file())
        self.assertTrue(reference.is_file())
        mirror_groups = {
            "task_contract.py": ("shared", "environment", "tests"),
            "dataset_contract.py": ("shared", "environment", "tests"),
            "async_contract.py": ("shared", "environment"),
            "evaluation_contract.py": ("shared", "tests"),
            "policy_contract.py": ("shared", "tests"),
        }
        for name, directories in mirror_groups.items():
            payloads = [(task_root / directory / name).read_bytes() for directory in directories]
            self.assertEqual(payloads[1:], [payloads[0]] * (len(payloads) - 1), name)

        reference_text = reference.read_text(encoding="utf-8")
        instruction_text = (task_root / "instruction.md").read_text(encoding="utf-8")
        policy_text = (task_root / "policy.yaml").read_text(encoding="utf-8")
        solve_text = solve.read_text(encoding="utf-8")
        for artifact in __import__("autolab.tasks.magpie_data_pipeline.shared.policy_contract", fromlist=["REQUIRED_OUTPUT_FILES"]).REQUIRED_OUTPUT_FILES:
            absolute = f"/app/output/{artifact}"
            self.assertIn(artifact, instruction_text)
            self.assertIn(absolute, reference_text)
        self.assertIn("policy.yaml", reference_text)
        self.assertRegex(reference_text.casefold(), r"same[-\s]session")
        self.assertRegex(reference_text.casefold(), r"no\s+claim of an external resumer")
        self.assertNotIn("benchmark score", reference_text.casefold())
        self.assertNotIn("hidden prompt", reference_text.casefold())
        self.assertNotIn("dataset row", reference_text.casefold())
        self.assertIn("/task-tools/magpie_async_run", solve_text)
        self.assertIn('MAGPIE_MAX_POLLS:-900', solve_text)
        self.assertIn("at most 900 iterations", reference_text)

        runtime_paths = task_contract.RUNTIME_PATHS
        self.assertEqual(runtime_paths, __import__("autolab.tasks.magpie_data_pipeline.environment.task_contract", fromlist=["RUNTIME_PATHS"]).RUNTIME_PATHS)
        self.assertEqual(runtime_paths, __import__("autolab.tasks.magpie_data_pipeline.tests.task_contract", fromlist=["RUNTIME_PATHS"]).RUNTIME_PATHS)
        self.assertEqual(runtime_paths["baseline_dataset"], "/task-assets/Magpie-Pro-MT-300K-v0.1.jsonl")
        dataset_runner = (task_root / "environment" / "generate_dataset.py").read_text(encoding="utf-8")
        self.assertIn('RUNTIME_PATHS["baseline_dataset"]', dataset_runner)
        self.assertIn('RUNTIME_PATHS["baseline_budget"]', dataset_runner)
        dockerfile = (task_root / "environment" / "Dockerfile").read_text(encoding="utf-8")
        for name, path in runtime_paths.items():
            if name in {"train_python", "infer_python"}:
                self.assertIn(str(Path(path).parent.parent), dockerfile, name)
            elif name not in {"baseline_dataset", "baseline_budget"}:
                self.assertIn(path, dockerfile, name)
        self.assertIn("@", (task_root / "task.toml").read_text(encoding="utf-8"))
        self.assertIn("@", policy_text)
        metadata = (task_root / "task.toml").read_text(encoding="utf-8")
        for revision in task_contract.ASSET_REVISIONS.values():
            pin = revision.rsplit("@", 1)[-1]
            self.assertIn(pin, metadata, revision)
            self.assertIn(pin, dockerfile, revision)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "commands.jsonl"
            runner = root / "magpie_async_run"
            runner.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "with open(os.environ['MAGPIE_TEST_LOG'], 'a', encoding='utf-8') as handle:\n"
                "    handle.write(json.dumps(sys.argv[1:]) + '\\n')\n"
                "if sys.argv[1] == 'status':\n"
                "    print('{\\\"status\\\": \\\"completed\\\"}')\n",
                encoding="utf-8",
            )
            os.chmod(runner, 0o755)
            result = subprocess.run(
                ["bash", str(solve)],
                env={**os.environ, "MAGPIE_ASYNC_RUN": str(runner), "MAGPIE_TEST_LOG": str(log), "MAGPIE_POLL_SECONDS": "0"},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            commands = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(commands[0][0], "submit")
            self.assertIn("--attempt-id", commands[0])
            self.assertIn("--hypothesis", commands[0])
            self.assertGreaterEqual(sum(command[0] == "status" for command in commands), 1)
            self.assertEqual(commands[-2], ["select", "--attempt-id", "filtered-prefix-001"])
            self.assertEqual(commands[-1], ["complete-control"])

    def test_repository_tracks_no_python_bytecode(self):
        task_root = Path(__file__).parents[1]
        repository = task_root.parents[2]
        relative = task_root.relative_to(repository)
        result = subprocess.run(
            ["git", "-C", str(repository), "ls-files", "--", str(relative)],
            check=True,
            capture_output=True,
            text=True,
        )
        tracked = result.stdout.splitlines()
        self.assertFalse([path for path in tracked if path.endswith(".pyc") or "/__pycache__/" in path])

    def test_reference_path_parses_completed_status_as_json_not_a_display_format(self):
        """A valid compact lifecycle payload must still select and complete."""

        solve = Path(__file__).parents[1] / "solution" / "solve.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "commands.jsonl"
            runner = root / "magpie_async_run"
            runner.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "with open(os.environ['MAGPIE_TEST_LOG'], 'a', encoding='utf-8') as handle:\n"
                "    handle.write(json.dumps(sys.argv[1:]) + '\\n')\n"
                "if sys.argv[1] == 'status':\n"
                "    print('{\\\"attempt_id\\\":\\\"filtered-prefix-001\\\",\\\"dataset_sha256\\\":\\\"abc\\\",\\\"status\\\":\\\"completed\\\"}')\n",
                encoding="utf-8",
            )
            os.chmod(runner, 0o755)
            result = subprocess.run(
                ["bash", str(solve)],
                env={
                    **os.environ,
                    "MAGPIE_ASYNC_RUN": str(runner),
                    "MAGPIE_TEST_LOG": str(log),
                    "MAGPIE_POLL_SECONDS": "0",
                    "MAGPIE_MAX_POLLS": "1",
                },
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            commands = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(commands[-2], ["select", "--attempt-id", "filtered-prefix-001"])
            self.assertEqual(commands[-1], ["complete-control"])
