#!/usr/bin/env python3
"""Focused unit tests for the six-scale positional task contract."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch


TASK_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ladder = load_module(
    "scaling_ladder_task",
    TASK_ROOT / "environment" / "task-tools" / "scaling_ladder_task.py",
)
score = load_module("scaling_ladder_score", TASK_ROOT / "tests" / "score.py")
proxy = load_module("blaunch_proxy", TASK_ROOT / "cluster" / "blaunch_proxy.py")
proxy_client = load_module(
    "blaunch_proxy_client", TASK_ROOT / "cluster" / "blaunch_proxy_client.py"
)


def write_baselines(path: Path) -> dict[str, dict[str, float]]:
    scales = {
        scale: {
            "paloma_bits_per_byte": 1.0,
            "paloma_macro_bits_per_byte": 1.2,
            "longppl": 100.0,
        }
        for scale in ladder.SCALE_ORDER
    }
    path.write_text(json.dumps({"version": 1, "scales": scales}) + "\n")
    return scales


class ScalingLadderContractTest(unittest.TestCase):
    def test_task_slug_is_positional_encoding(self) -> None:
        with (TASK_ROOT / "task.toml").open("rb") as handle:
            task_name = tomllib.load(handle)["task"]["name"]
        self.assertEqual(task_name.rsplit("/", 1)[-1], "positional-encoding")

    def test_positional_encoding_path_spelling_is_consistent(self) -> None:
        training_root = TASK_ROOT / "environment" / "project-overlay" / "examples" / "training"
        legacy_name = "postional_" + "encoding"
        self.assertTrue((training_root / "positional_encoding").is_dir())
        self.assertFalse((training_root / legacy_name).exists())
        for path in TASK_ROOT.rglob("*"):
            if path.is_file() and path.suffix != ".pyc":
                self.assertNotIn(legacy_name, path.read_text(errors="ignore"), str(path))

    def test_offline_history_is_complete_baseline_manifest(self) -> None:
        history_path = TASK_ROOT / "baselines" / "positional_adamh_baselines.json"
        payload = json.loads(history_path.read_text())
        self.assertEqual(payload["version"], 1)
        loaded = ladder.load_baselines(history_path)
        self.assertEqual(set(loaded), set(ladder.SCALE_ORDER))
        for scale in ladder.SCALE_ORDER:
            row = payload["scales"][scale]
            longppl_point = next(
                item
                for item in row["longppl_trajectory"]
                if item["iteration"] == row["longppl_eval_step"]
            )
            self.assertEqual(longppl_point["longppl"], row["longppl"])
            self.assertEqual(loaded[scale]["longppl"], row["longppl"])

    def test_runtime_uses_only_merged_baseline_manifest(self) -> None:
        legacy_name = "scale_" + "baselines.json"
        self.assertFalse((TASK_ROOT / "baselines" / legacy_name).exists())
        runtime_files = (
            TASK_ROOT / "task.toml",
            TASK_ROOT / "instruction.md",
            TASK_ROOT / "cluster" / "launch_harbor_lsf.sh",
            TASK_ROOT / "cluster" / "preflight.sh",
            TASK_ROOT / "environment" / "task-tools" / "scaling_ladder_task.py",
            TASK_ROOT / "tests" / "score.py",
            TASK_ROOT / "tests" / "test.sh",
            TASK_ROOT / "tests" / "validate_submission.py",
        )
        for path in runtime_files:
            self.assertNotIn(legacy_name, path.read_text(), str(path))

    def test_source_hash_ignores_deterministic_tool_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            project.mkdir()
            subprocess.run(["git", "init", "-q", str(project)], check=True)
            (project / ".gitignore").write_text(
                ".ruff_cache/\n*.egg-info/\nwandb/\n"
            )
            (project / "candidate.py").write_text("value = 1\n")
            before, _ = ladder.project_inventory(project)
            (project / ".ruff_cache").mkdir()
            (project / ".ruff_cache" / "state").write_text("generated\n")
            (project / "package.egg-info").mkdir()
            (project / "package.egg-info" / "PKG-INFO").write_text("generated\n")
            (project / "wandb").mkdir()
            (project / "wandb" / "debug.log").write_text("generated\n")
            after, _ = ladder.project_inventory(project)
            self.assertEqual(before, after)

    def test_blaunch_proxy_forwards_only_safe_lsf_profile_fields(self) -> None:
        environment = {
            "HOME": "/app/home",
            "PATH": "/app/poisoned-path",
            "APPTAINER": "/usr/bin/apptainer",
            "APPTAINER_BIND": "/outer:/inner",
            "APPTAINER_WORKDIR": "/app/workdir",
            "SINGULARITY_WORKDIR": "/app/workdir",
            "LSB_JOBID": "untrusted-override",
            "LSB_MCPU_HOSTS": "outside 8",
            "LSF_QUEUE": "untrusted-override",
            "LSF_SLOTS": "128",
            "LSF_SPAN": "span[ptile=8]",
            "LSF_WALLTIME": "08:00",
            "LSF_MEMORY": "65536",
            "PROFILE": "positional-scaling-E5",
        }
        expected = {
            "APPTAINER": "/usr/bin/apptainer",
            "LSF_SLOTS": "128",
            "LSF_SPAN": "span[ptile=8]",
            "LSF_WALLTIME": "08:00",
            "LSF_MEMORY": "65536",
            "PROFILE": "positional-scaling-E5",
        }
        self.assertEqual(proxy.safe_client_environment(environment), expected)
        with patch.dict(os.environ, environment, clear=True):
            self.assertEqual(proxy_client.forwarded_environment(), expected)

    def test_locked_scale_contracts_span_550m_to_2p545b(self) -> None:
        self.assertEqual(tuple(ladder.SCALE_CONTRACTS), ladder.SCALE_ORDER)
        self.assertEqual(ladder.SCALE_CONTRACTS["E0"]["parameters"], 550_337_664)
        self.assertEqual(ladder.SCALE_CONTRACTS["E5"]["parameters"], 2_544_614_912)
        self.assertEqual(
            [ladder.SCALE_CONTRACTS[item]["gpus"] for item in ladder.SCALE_ORDER],
            [8, 8, 8, 32, 32, 128],
        )

    def test_longppl_gate_is_strict_and_paloma_is_scale_relative(self) -> None:
        baselines = {
            scale: {
                "paloma_bits_per_byte": 1.0,
                "paloma_macro_bits_per_byte": 1.2,
                "longppl": 100.0,
            }
            for scale in ladder.SCALE_ORDER
        }
        metrics = {
            "paloma_bits_per_byte": 1.01,
            "paloma_macro_bits_per_byte": 1.212,
            "longppl": 100.0,
        }
        guards = ladder.metric_guards("E0", metrics, 550_337_664, baselines)
        self.assertFalse(guards["longppl_improvement"])
        metrics["longppl"] = 99.999
        self.assertTrue(all(ladder.metric_guards("E0", metrics, 550_337_664, baselines).values()))

    def test_all_scales_can_prepare_in_parallel_without_prerequisites(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            project = root / "project"
            project.mkdir()
            (project / "candidate.py").write_text("value = 1\n")
            subprocess.run(["git", "init", "-q", str(project)], check=True)
            hypothesis = root / "hypothesis.md"
            hypothesis.write_text("test hypothesis\n")
            baselines = root / "baselines.json"
            write_baselines(baselines)
            with patch.dict(os.environ, {"POSITIONAL_OUTPUT_ROOT": str(output)}):
                source_hashes = set()
                for scale in ladder.SCALE_ORDER:
                    attempt_id = f"candidate-{scale.lower()}"
                    ladder.prepare(
                        argparse.Namespace(
                            attempt=output / "attempts" / attempt_id,
                            attempt_id=attempt_id,
                            scale=scale,
                            hypothesis_file=hypothesis,
                            project=project,
                            baselines=baselines,
                        )
                    )
                    contract = json.loads(
                        (output / "attempts" / attempt_id / "run_contract.json").read_text()
                    )
                    source_hashes.add(contract["source_inventory_sha256"])
                    self.assertTrue(
                        (output / contract["source_snapshot"]).is_file(),
                        f"missing immutable source snapshot for {scale}",
                    )
                self.assertEqual(len(source_hashes), 1)

                # An independently changed source can also start at any scale;
                # the complete-ladder staging gate, not prepare(), rejects a
                # mixture of hashes.
                (project / "candidate.py").write_text("value = 2\n")
                ladder.prepare(
                    argparse.Namespace(
                        attempt=output / "attempts" / "changed-e5",
                        attempt_id="changed-e5",
                        scale="E5",
                        hypothesis_file=hypothesis,
                        project=project,
                        baselines=baselines,
                    )
                )
                changed_contract = json.loads(
                    (output / "attempts" / "changed-e5" / "run_contract.json").read_text()
                )
                self.assertNotIn(changed_contract["source_inventory_sha256"], source_hashes)

    def test_host_leases_are_disjoint(self) -> None:
        script = TASK_ROOT / "environment" / "task-tools" / "host_lease.py"
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "leases.json"
            lsb_hosts = " ".join(f"host{i} 8" for i in range(6))
            first = subprocess.check_output(
                [
                    sys.executable, str(script), "acquire", "--state", str(state),
                    "--attempt-id", "first", "--slots", "32", "--pid", str(os.getpid()),
                    "--lsb-hosts", lsb_hosts,
                ],
                text=True,
            ).strip()
            second = subprocess.check_output(
                [
                    sys.executable, str(script), "acquire", "--state", str(state),
                    "--attempt-id", "second", "--slots", "16", "--pid", str(os.getpid()),
                    "--lsb-hosts", lsb_hosts,
                ],
                text=True,
            ).strip()
            first_hosts = set(first.split("\t", 1)[1].split()[::2])
            second_hosts = set(second.split("\t", 1)[1].split()[::2])
            self.assertEqual(len(first_hosts), 4)
            self.assertEqual(len(second_hosts), 2)
            self.assertFalse(first_hosts & second_hosts)

    def test_score_is_zero_if_one_rung_only_matches_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "runs"
            logs = root / "logs"
            baseline_path = root / "baselines.json"
            baselines = write_baselines(baseline_path)
            policy = root / "policy.json"
            policy.write_text(json.dumps({"policy_gate": 1}))
            for scale, contract in score.SCALES.items():
                run = runs / scale
                step = run / "eval_harness" / f"step_{contract['iterations']:08d}"
                long_step = run / "longppl" / f"step_{contract['iterations']:08d}"
                step.mkdir(parents=True)
                long_step.mkdir(parents=True)
                (step / "results.json").write_text(json.dumps({
                    "paloma_aggregate": {
                        "bits_per_byte": 1.0, "macro_bits_per_byte": 1.2, "subsets": 16,
                    }
                }))
                protocol = {
                    "iteration": contract["iterations"], "world_size": contract["gpus"],
                    "longppl_commit": "b4f80af0015dd12202438e1e440cf12316f1cb23",
                    "minimum_source_tokens": 16384, "maximum_source_length": 32768,
                    "training_sequence_length": 4096, "truncation_length": 4096,
                    "sliding_window": 1024,
                    "target_tokenizer_revision": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
                    "mode": "offline",
                }
                candidate_longppl = 100.0 if scale == "E3" else 99.0
                (long_step / "results.json").write_text(json.dumps({
                    "results": {
                        "longppl": candidate_longppl, "longppl_loss": 4.6,
                        "ppl": 15.0, "samples": 50,
                    },
                    "longppl_eval": protocol,
                }))
                (run / "run.log").write_text(
                    "number of parameters on (tensor, pipeline) model parallel rank (0, 0): "
                    f"{contract['parameters']}\n"
                )
                baselines[scale]["longppl"] = 100.0
            baseline_path.write_text(json.dumps({"version": 1, "scales": baselines}))
            argv = [
                "score.py", "--runs-root", str(runs), "--baselines", str(baseline_path),
                "--policy-report", str(policy), "--logs-dir", str(logs),
            ]
            with patch.object(sys, "argv", argv):
                score.main()
            self.assertEqual(json.loads((logs / "reward.json").read_text())["reward"], 0.0)


if __name__ == "__main__":
    unittest.main()
