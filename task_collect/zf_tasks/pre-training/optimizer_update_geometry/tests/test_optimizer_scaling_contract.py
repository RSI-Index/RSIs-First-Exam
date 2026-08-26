#!/usr/bin/env python3
"""Focused unit tests for the six-scale optimizer task contract."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
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
    "optimizer_scaling_task",
    TASK_ROOT / "environment" / "task-tools" / "optimizer_scaling_task.py",
)
proxy = load_module("blaunch_proxy", TASK_ROOT / "cluster" / "blaunch_proxy.py")
proxy_client = load_module(
    "blaunch_proxy_client", TASK_ROOT / "cluster" / "blaunch_proxy_client.py"
)


def write_baselines(path: Path) -> dict[str, dict[str, object]]:
    scales = {
        scale: {
            "paloma_bits_per_byte": 1.0,
            "paloma_macro_bits_per_byte": 1.2,
            "scoring_gpu_seconds": 300.0,
            "scoring_fixed_window_loss": 2.0,
            "paloma_trajectory": [
                {"iteration": 1, "cumulative_gpu_seconds": 100.0, "bits_per_byte": 1.0, "macro_bits_per_byte": 1.2},
                {"iteration": 2, "cumulative_gpu_seconds": 200.0, "bits_per_byte": 1.0, "macro_bits_per_byte": 1.2},
                {"iteration": 3, "cumulative_gpu_seconds": 300.0, "bits_per_byte": 1.0, "macro_bits_per_byte": 1.2},
            ],
            "loss_cost_curve": [
                {"iteration": 1, "cumulative_gpu_seconds": 100.0, "fixed_window_loss": 2.0},
                {"iteration": 2, "cumulative_gpu_seconds": 200.0, "fixed_window_loss": 2.0},
                {"iteration": 3, "cumulative_gpu_seconds": 300.0, "fixed_window_loss": 2.0},
            ],
        }
        for scale in ladder.SCALE_ORDER
    }
    path.write_text(json.dumps({"version": 2, "scales": scales}) + "\n")
    return scales


class ScalingLadderContractTest(unittest.TestCase):
    def test_active_trusted_budget_hold_blocks_checkpoint_continuation(self) -> None:
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
            contract_root = root / "run-contract"
            contract_root.mkdir()
            (contract_root / "CONTROL_HOLD_E2_RESUME_BUDGET_TEST.json").write_text(
                json.dumps(
                    {
                        "type": "trusted_budget_safety_hold",
                        "status": "active",
                        "applies_to_scale": "E2",
                        "gate": {"status": "hold"},
                    }
                )
            )
            attempt = output / "attempts" / "held-e2-resume"
            with patch.dict(
                os.environ,
                {
                    "OPTIMIZER_OUTPUT_ROOT": str(output),
                    "OPTIMIZER_RUN_CONTRACT_ROOT": str(contract_root),
                },
            ):
                with self.assertRaisesRegex(
                    SystemExit, "checkpoint continuation is held"
                ):
                    ladder.prepare(
                        argparse.Namespace(
                            attempt=attempt,
                            attempt_id="held-e2-resume",
                            scale="E2",
                            hypothesis_file=hypothesis,
                            project=project,
                            baselines=baselines,
                            resume_iteration=25_000,
                            resume_attempt_id="interrupted-e2",
                        )
                    )
            self.assertFalse(attempt.exists())

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
            "PROFILE": "optimizer-scaling-E5",
        }
        expected = {
            "APPTAINER": "/usr/bin/apptainer",
            "LSF_SLOTS": "128",
            "LSF_SPAN": "span[ptile=8]",
            "LSF_WALLTIME": "08:00",
            "LSF_MEMORY": "65536",
            "PROFILE": "optimizer-scaling-E5",
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

    def test_all_scales_run_paloma_every_5000_updates(self) -> None:
        profile = (
            TASK_ROOT
            / "environment"
            / "task-tools"
            / "optimizer_scaling_ladder_profile.sh"
        )
        for scale in ladder.SCALE_ORDER:
            configured = subprocess.check_output(
                [
                    "bash",
                    "-c",
                    (
                        'set -euo pipefail; export OPTIMIZER_SCALE="$1"; '
                        'source "$2"; printf "%s %s %s\\n" '
                        '"$EVAL_INTERVAL" "$LM_EVAL_INTERVAL" "$SAVE_INTERVAL"'
                    ),
                    "profile-test",
                    scale,
                    str(profile),
                ],
                text=True,
            ).strip()
            self.assertEqual(configured, "5000 5000 5000", scale)

    def test_per_run_gate_checks_integrity_not_single_point_quality(self) -> None:
        metrics = {
            "paloma_bits_per_byte": 1.01,
            "paloma_macro_bits_per_byte": 1.212,
            "cost_trace_rows": ladder.SCALE_CONTRACTS["E0"]["iterations"],
        }
        guards = ladder.metric_guards("E0", metrics, 550_337_664)
        self.assertTrue(all(guards.values()))
        metrics["cost_trace_rows"] -= 1
        self.assertFalse(ladder.metric_guards("E0", metrics, 550_337_664)["complete_cost_trace"])

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
            with patch.dict(os.environ, {"OPTIMIZER_OUTPUT_ROOT": str(output)}):
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

    def test_early_stop_matches_the_exact_brokered_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempt = root / "output" / "attempts" / "candidate-e0"
            process_root = root / "proc"
            matching = process_root / "123"
            unrelated = process_root / "456"
            matching.mkdir(parents=True)
            unrelated.mkdir()
            matching.joinpath("cmdline").write_bytes(
                b"python\0/run-contract/blaunch_proxy_client.py\0-z\0node0\0"
                + f"RUN_DIR={attempt}\0--host-worker\00".encode()
            )
            unrelated.joinpath("cmdline").write_bytes(
                b"python\0/run-contract/blaunch_proxy_client.py\0-z\0node1\0"
                b"RUN_DIR=/app/output/attempts/other-e0\0--host-worker\00"
            )
            with patch("os.kill") as mocked_kill:
                stopped = ladder.signal_training_workers(attempt, process_root)
            self.assertEqual(stopped, [123])
            mocked_kill.assert_called_once_with(123, ladder.signal.SIGTERM)

    def test_early_stopped_ledger_records_executed_not_target_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            project = root / "project"
            project.mkdir()
            subprocess.run(["git", "init", "-q", str(project)], check=True)
            (project / "candidate.py").write_text("value = 1\n")
            hypothesis = root / "hypothesis.md"
            hypothesis.write_text("screen hypothesis\n")
            baselines = root / "baselines.json"
            write_baselines(baselines)
            attempt = output / "attempts" / "screen-e0"
            with patch.dict(os.environ, {"OPTIMIZER_OUTPUT_ROOT": str(output)}):
                ladder.prepare(argparse.Namespace(
                    attempt=attempt,
                    attempt_id="screen-e0",
                    scale="E0",
                    hypothesis_file=hypothesis,
                    project=project,
                    baselines=baselines,
                ))
                with (attempt / "optimizer_cost_trace.jsonl").open("w") as stream:
                    for iteration in range(1, 4):
                        stream.write(json.dumps({
                            "iteration": iteration,
                            "cumulative_gpu_seconds": 100.0 * iteration,
                            "lm_loss": 2.0,
                            "skipped": 0,
                        }) + "\n")
                (attempt / "early_stop.json").write_text(json.dumps({
                    "requested_at": "2026-08-01T00:00:00+00:00",
                    "reason": "screen rejected",
                }))
                with self.assertRaises(SystemExit):
                    ladder.finish(argparse.Namespace(
                        attempt=attempt,
                        project=project,
                        exit_code=130,
                        baselines=baselines,
                    ))
            status = json.loads((attempt / "status.json").read_text())
            ledger = json.loads((output / "experiments.jsonl").read_text())
            self.assertEqual(status["status"], "early_stopped")
            self.assertEqual(status["charged_updates"], 3)
            self.assertEqual(status["charged_gpu_seconds"], 300.0)
            self.assertEqual(ledger["optimizer_updates"], 3)
            self.assertEqual(ledger["training_tokens"], 3 * 16 * 4096)
            self.assertEqual(ledger["target_training_tokens"], 2_904_358_912)

    def test_resumed_cost_trace_charges_only_executed_updates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            attempt = Path(temporary) / "resume-e1"
            attempt.mkdir()
            (attempt / "run_contract.json").write_text(json.dumps({
                "resume": {"iteration": 5000, "attempt_id": "source-e1"},
            }))
            with (attempt / "optimizer_cost_trace.jsonl").open("w") as stream:
                for offset, iteration in enumerate(range(5001, 5004), 1):
                    stream.write(json.dumps({
                        "iteration": iteration,
                        "cumulative_gpu_seconds": 10.0 * offset,
                        "lm_loss": 2.0,
                        "skipped": 0,
                    }) + "\n")
            self.assertEqual(ladder.charged_attempt_cost(attempt), (30.0, 3, None))
            rows = ladder.read_cost_trace(attempt)
            self.assertEqual([row["iteration"] for row in rows], [5001, 5002, 5003])

    def test_resumed_scoring_view_joins_checkpoint_prefix_and_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            attempts = Path(temporary) / "attempts"
            source = attempts / "source-e0"
            resumed = attempts / "resumed-e0"
            source.mkdir(parents=True)
            resumed.mkdir()
            common = {"scale": "E0", "source_inventory_sha256": "source-hash"}
            (source / "run_contract.json").write_text(json.dumps({
                **common,
                "attempt_id": "source-e0",
            }))
            (resumed / "run_contract.json").write_text(json.dumps({
                **common,
                "attempt_id": "resumed-e0",
                "resume": {"iteration": 3, "attempt_id": "source-e0"},
            }))
            with (source / "optimizer_cost_trace.jsonl").open("w") as stream:
                for iteration in range(1, 5):
                    stream.write(json.dumps({
                        "iteration": iteration,
                        "cumulative_gpu_seconds": 10.0 * iteration,
                        "lm_loss": 2.0,
                        "skipped": 0,
                    }) + "\n")
            with (resumed / "optimizer_cost_trace.jsonl").open("w") as stream:
                for offset, iteration in enumerate(range(4, 6), 1):
                    stream.write(json.dumps({
                        "iteration": iteration,
                        "cumulative_gpu_seconds": 12.0 * offset,
                        "lm_loss": 1.9,
                        "skipped": 0,
                    }) + "\n")
            for attempt, iteration, micro in (
                (source, 2, 1.2),
                (source, 4, 1.1),
                (resumed, 5, 1.0),
            ):
                step = attempt / "eval_harness" / f"step_{iteration:08d}"
                step.mkdir(parents=True)
                (step / "results.json").write_text(json.dumps({
                    "paloma_aggregate": {
                        "bits_per_byte": micro,
                        "macro_bits_per_byte": micro + 0.1,
                    }
                }))

            trace = ladder.read_scoring_cost_trace(resumed)
            self.assertEqual([row["iteration"] for row in trace], [1, 2, 3, 4, 5])
            self.assertEqual(
                [row["cumulative_gpu_seconds"] for row in trace],
                [10.0, 20.0, 30.0, 42.0, 54.0],
            )
            trajectory = ladder.read_scoring_paloma_trajectory(resumed, trace)
            self.assertEqual([row["iteration"] for row in trajectory], [2, 5])
            self.assertEqual(
                [row["cumulative_gpu_seconds"] for row in trajectory],
                [20.0, 54.0],
            )

    def test_weighted_matched_cost_score_tolerates_one_small_early_regression(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempt = root / "attempt"
            attempt.mkdir()
            baseline_path = root / "baselines.json"
            baselines = write_baselines(baseline_path)
            candidate_micro = (1.005, 0.999, 0.99)
            for iteration, micro in enumerate(candidate_micro, 1):
                step = attempt / "eval_harness" / f"step_{iteration:08d}"
                step.mkdir(parents=True)
                (step / "results.json").write_text(json.dumps({
                    "paloma_aggregate": {
                        "bits_per_byte": micro,
                        "macro_bits_per_byte": 1.2,
                        "subsets": 16,
                    }
                }))
            with (attempt / "optimizer_cost_trace.jsonl").open("w") as stream:
                for iteration in range(1, 4):
                    stream.write(json.dumps({
                        "iteration": iteration,
                        "cumulative_gpu_seconds": 100.0 * iteration,
                        "lm_loss": 1.9,
                        "skipped": 0,
                    }) + "\n")
            result = ladder.evaluate_matched_cost_scale(
                "E0", attempt, baselines["E0"]
            )
            self.assertGreater(result["score"], 0.0)
            self.assertLessEqual(
                result["matched"][0]["candidate_micro_bpb"],
                1.01 * result["matched"][0]["baseline_micro_bpb"],
            )

    def test_matched_cost_uses_shared_measured_overlap_without_extrapolation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempt = root / "attempt"
            attempt.mkdir()
            baseline_path = root / "baselines.json"
            baselines = write_baselines(baseline_path)
            baselines["E0"]["paloma_trajectory"] = [
                {
                    "iteration": 2,
                    "cumulative_gpu_seconds": 250.0,
                    "bits_per_byte": 1.0,
                    "macro_bits_per_byte": 1.2,
                },
                {
                    "iteration": 3,
                    "cumulative_gpu_seconds": 300.0,
                    "bits_per_byte": 1.0,
                    "macro_bits_per_byte": 1.2,
                },
            ]
            for iteration, micro in enumerate((1.1, 1.0, 0.9), 1):
                step = attempt / "eval_harness" / f"step_{iteration:08d}"
                step.mkdir(parents=True)
                (step / "results.json").write_text(json.dumps({
                    "paloma_aggregate": {
                        "bits_per_byte": micro,
                        "macro_bits_per_byte": micro + 0.2,
                        "subsets": 16,
                    }
                }))
            with (attempt / "optimizer_cost_trace.jsonl").open("w") as stream:
                for iteration in range(1, 4):
                    stream.write(json.dumps({
                        "iteration": iteration,
                        "cumulative_gpu_seconds": 100.0 * iteration,
                        "lm_loss": 1.9,
                        "skipped": 0,
                    }) + "\n")

            result = ladder.evaluate_matched_cost_scale(
                "E0", attempt, baselines["E0"]
            )
            self.assertEqual(result["matched_cost_policy"], "common_measured_overlap")
            self.assertEqual(result["nominal_start_gpu_seconds"], 100.0)
            self.assertEqual(result["shared_measured_start_gpu_seconds"], 250.0)
            self.assertEqual(
                [point["gpu_seconds"] for point in result["matched"]],
                [250.0, 275.0, 300.0],
            )
            self.assertEqual(
                [point["weight"] for point in result["matched"]],
                [0.2, 0.3, 0.5],
            )


if __name__ == "__main__":
    unittest.main()
