#!/usr/bin/env python3
"""Contract tests for the six-rung learning-rate Harbor task."""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pytest


TASK_ROOT = Path(__file__).resolve().parents[1]
TOOLS = TASK_ROOT / "environment" / "task-tools"
RUNTIME = (
    TASK_ROOT
    / "environment"
    / "project-overlay"
    / "examples"
    / "training"
    / "lr_schedule"
    / "runtime"
)

SCALE_ORDER = ("E0", "E1", "E2", "E3", "E4", "E5")
EXPECTED_SCALES = {
    "E0": {"parameters": 550_337_664, "iterations": 44_317, "tokens": 2_904_358_912, "gbs": 16, "mbs": 2, "gpus": 8},
    "E1": {"parameters": 837_007_744, "iterations": 55_125, "tokens": 3_612_672_000, "gbs": 16, "mbs": 2, "gpus": 8},
    "E2": {"parameters": 998_036_992, "iterations": 38_014, "tokens": 4_982_571_008, "gbs": 32, "mbs": 4, "gpus": 8},
    "E3": {"parameters": 1_384_584_448, "iterations": 40_283, "tokens": 10_559_946_752, "gbs": 64, "mbs": 2, "gpus": 32},
    "E4": {"parameters": 1_934_716_160, "iterations": 56_477, "tokens": 14_805_106_688, "gbs": 64, "mbs": 2, "gpus": 32},
    "E5": {"parameters": 2_544_614_912, "iterations": 35_510, "tokens": 18_617_466_880, "gbs": 128, "mbs": 1, "gpus": 128},
}


def load_module(name: str, path: Path):
    assert path.is_file(), f"missing module: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def runtime_modules():
    controller = load_module("lr_schedule_controller", RUNTIME / "lr_schedule_controller.py")
    candidate = load_module("lr_schedule_candidate", RUNTIME / "lr_schedule_candidate.py")
    return controller, candidate


def state_module():
    return load_module("lr_schedule_task", TOOLS / "lr_schedule_task.py")


def tiny_manifest() -> dict[str, object]:
    scales: dict[str, object] = {}
    for scale in SCALE_ORDER:
        scales[scale] = {
            **EXPECTED_SCALES[scale],
            "global_batch_size": EXPECTED_SCALES[scale]["gbs"],
            "model": f"fixture-{scale}",
            "final_update": EXPECTED_SCALES[scale]["iterations"],
            "scoring_update": EXPECTED_SCALES[scale]["iterations"],
            "reference_status": "exact_final",
            "paloma_bits_per_byte": 1.0,
            "paloma_macro_bits_per_byte": 1.2,
            "scoring_gpu_seconds": 100.0,
            "scoring_fixed_window_loss": 2.0,
            "wandb_project": "offline-fixture",
            "wandb_run_id": f"fixture-{scale.lower()}",
            "wandb_state": "finished",
            "source_inventory_sha256": "a" * 64,
            "checkpoint": {"iteration": EXPECTED_SCALES[scale]["iterations"], "path_sha256": "b" * 64},
            "paloma_trajectory": [
                {"iteration": 1, "cumulative_gpu_seconds": 50.0, "bits_per_byte": 1.0, "macro_bits_per_byte": 1.2},
                {"iteration": EXPECTED_SCALES[scale]["iterations"], "cumulative_gpu_seconds": 100.0, "bits_per_byte": 1.0, "macro_bits_per_byte": 1.2},
            ],
            "loss_cost_curve": [
                {"iteration": 1, "cumulative_gpu_seconds": 50.0, "fixed_window_loss": 2.1},
                {"iteration": EXPECTED_SCALES[scale]["iterations"], "cumulative_gpu_seconds": 100.0, "fixed_window_loss": 2.0},
            ],
            "run_contract": {
                "seed": 0,
                "sequence_length": 4096,
                "tensor_parallel": 1,
                "pipeline_parallel": 1,
                "context_parallel": 1,
            },
        }
    return {
        "version": 3,
        "evidence_scope": "locked_wsd_exact_final",
        "cost_definition": "active_training_gpu_seconds_excluding_evaluation",
        "scales": scales,
    }


def candidate_scales(micro: tuple[float, float] = (0.99, 0.98)) -> dict[str, object]:
    return {
        scale: {
            "final_update": EXPECTED_SCALES[scale]["iterations"],
            "scoring_gpu_seconds": 100.5,
            "source_inventory_sha256": "c" * 64,
            "paloma_trajectory": [
                {"iteration": 1, "bits_per_byte": micro[0], "macro_bits_per_byte": 1.19},
                {"iteration": EXPECTED_SCALES[scale]["iterations"], "bits_per_byte": micro[1], "macro_bits_per_byte": 1.18},
            ],
        }
        for scale in SCALE_ORDER
    }


def test_task_metadata_and_layout() -> None:
    required = [
        TASK_ROOT / "task.toml",
        TASK_ROOT / "policy.yaml",
        TASK_ROOT / "environment" / "learning_rate.def",
        TASK_ROOT / "environment" / "materialize_project.sh",
        TOOLS / "lr_schedule_ladder_profile.sh",
        TOOLS / "run_lr_schedule_ladder.sh",
        TOOLS / "run_lr_schedule_candidate_ladder.sh",
        TOOLS / "lr_schedule_task.py",
        RUNTIME / "lr_schedule_candidate.py",
        RUNTIME / "lr_schedule_controller.py",
        RUNTIME / "pretrain_gpt_marin_adamh.py",
        TASK_ROOT / "tests" / "validate_submission.py",
        TASK_ROOT / "cluster" / "launch_harbor_lsf.sh",
        TASK_ROOT / "baselines" / "lr_schedule_baselines.json",
    ]
    assert not [str(path.relative_to(TASK_ROOT)) for path in required if not path.is_file()]

    task = tomllib.loads((TASK_ROOT / "task.toml").read_text())
    assert task["task"]["name"] == "more-task/learning-rate"
    assert "schedule mechanism" in task["task"]["description"]
    assert "controller" not in task["task"]["description"]
    assert "normalized progress" in task["metadata"]["optimization"]["scope"]
    assert "peak learning rates" in task["metadata"]["optimization"]["scope"]
    assert task["metadata"]["baseline"]["manifest"] == "/task-data/lr_schedule_baselines.json"
    assert task["metadata"]["run_contract"]["scale_gpus"] == [8, 8, 8, 32, 32, 128]
    assert task["metadata"]["run_contract"]["outer_allocation_gpus"] == 256
    assert task["environment"]["env"]["WANDB_MODE"] == "offline"
    assert task["environment"]["env"]["LR_SCHEDULE_BASELINES"] == "/task-data/lr_schedule_baselines.json"


def test_policy_exposes_only_candidate_module() -> None:
    public = (TASK_ROOT / "policy.yaml").read_text()
    verifier = (TASK_ROOT / "tests" / "policy.yaml").read_text()
    assert "task: learning-rate" in public
    assert "task: learning-rate" in verifier
    allowed = "examples/training/lr_schedule/runtime/lr_schedule_candidate.py"
    assert public.count(allowed) == 1
    assert verifier.count(allowed) == 1
    rules = public
    for forbidden in ("absolute step", "horizon", "baseline", "Paloma", "collective"):
        assert forbidden.lower() in rules.lower()


def test_locked_scale_contracts_and_profiles() -> None:
    ladder = state_module()
    assert tuple(ladder.SCALE_ORDER) == SCALE_ORDER
    for scale, expected in EXPECTED_SCALES.items():
        assert {key: ladder.SCALE_CONTRACTS[scale][key] for key in expected} == expected
    profile = (TOOLS / "lr_schedule_ladder_profile.sh").read_text()
    for scale, values in EXPECTED_SCALES.items():
        assert f"{scale})" in profile
        assert str(values["iterations"]) in profile
        assert str(values["tokens"]) in profile
    assert "TENSOR_MODEL_PARALLEL_SIZE=1" in profile
    assert "PIPELINE_MODEL_PARALLEL_SIZE=1" in profile


@pytest.mark.parametrize(
    "progress,expected",
    [(0.0, 0.0), (0.05, 0.5), (0.1, 1.0), (0.8, 1.0), (0.9, 0.5), (1.0, 0.0)],
)
def test_default_schedule_exactly_reproduces_wsd(
    progress: float, expected: float
) -> None:
    runtime, candidate = runtime_modules()
    schedule = candidate.build_schedule()
    assert runtime.evaluate_schedule(schedule, progress) == pytest.approx(expected)
    assert not hasattr(schedule, "step")


@pytest.mark.parametrize(
    "multiplier,error",
    [
        (-1.0, r"\[0, 1\]"),
        (1.000001, r"\[0, 1\]"),
        (math.inf, "finite"),
    ],
)
def test_schedule_output_validation(multiplier, error) -> None:
    runtime, _ = runtime_modules()

    class Invalid:
        def multiplier(self, progress):
            del progress
            return multiplier

    with pytest.raises(ValueError, match=error):
        runtime.evaluate_schedule(Invalid(), 0.0)


@pytest.mark.parametrize("progress", [-0.001, 1.001, math.inf])
def test_schedule_accepts_only_normalized_progress(progress: float) -> None:
    runtime, candidate = runtime_modules()
    with pytest.raises(ValueError, match="progress"):
        runtime.evaluate_schedule(candidate.build_schedule(), progress)


@dataclass
class FakeScheduler:
    num_steps: int = 25

    def get_lr(self, group: dict[str, float]) -> float:
        return group["max_lr"] * (1.0 - self.num_steps / 100.0)


@dataclass
class FakeOptimizer:
    param_groups: list[dict[str, float]]


def test_multiplier_uses_locked_peak_lrs_without_wsd_compounding() -> None:
    runtime, _ = runtime_modules()

    class Half:
        def multiplier(self, progress):
            del progress
            return 0.5

    optimizer = FakeOptimizer([{"max_lr": 0.4, "lr": 0.0}, {"max_lr": 0.04, "lr": 0.0}])
    harness = runtime.ScheduleHarness(Half())
    first = harness.apply(optimizer, progress=0.25)
    second = harness.apply(optimizer, progress=0.25)
    assert first["peak_lrs"] == pytest.approx([0.4, 0.04])
    assert first["effective_lrs"] == pytest.approx([0.2, 0.02])
    assert second["effective_lrs"] == pytest.approx(first["effective_lrs"])
    assert optimizer.param_groups[0]["lr"] / optimizer.param_groups[1]["lr"] == pytest.approx(10.0)


def test_stateless_schedule_resume_is_source_bound(tmp_path: Path) -> None:
    runtime, _ = runtime_modules()
    checkpoint = tmp_path / "iter_0000123"
    checkpoint.mkdir()
    runtime.validate_stateless_resume(
        checkpoint_iteration=123,
        expected_iteration=123,
        checkpoint_source_sha256="d" * 64,
        expected_source_sha256="d" * 64,
    )
    with pytest.raises(ValueError, match="source"):
        runtime.validate_stateless_resume(
            checkpoint_iteration=123,
            expected_iteration=123,
            checkpoint_source_sha256="d" * 64,
            expected_source_sha256="e" * 64,
        )
    assert not (checkpoint / "lr_schedule_state.json").exists()


def test_runtime_requires_trusted_source_inventory_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, _ = runtime_modules()
    monkeypatch.delenv("LR_SCHEDULE_SOURCE_INVENTORY_SHA256", raising=False)
    with pytest.raises(RuntimeError, match="source inventory"):
        runtime.trusted_source_inventory_sha256()
    monkeypatch.setenv("LR_SCHEDULE_SOURCE_INVENTORY_SHA256", "b" * 64)
    assert runtime.trusted_source_inventory_sha256() == "b" * 64
    monkeypatch.setenv("LR_SCHEDULE_SOURCE_INVENTORY_SHA256", "not-a-hash")
    with pytest.raises(RuntimeError, match="SHA-256"):
        runtime.trusted_source_inventory_sha256()


def test_trace_contains_reproducible_schedule_evidence(tmp_path: Path) -> None:
    runtime, _ = runtime_modules()
    path = tmp_path / "lr_schedule_trace.jsonl"
    runtime.append_schedule_trace(
        path,
        {
            "iteration": 3,
            "progress": 0.5,
            "peak_lrs": [0.3, 0.03],
            "multiplier": 0.75,
            "effective_lrs": [0.225, 0.0225],
            "source_sha256": "f" * 64,
        },
    )
    row = json.loads(path.read_text())
    assert set(row) >= {
        "iteration",
        "progress",
        "peak_lrs",
        "multiplier",
        "effective_lrs",
        "source_sha256",
    }
    assert "state" not in row
    assert "diagnostics" not in row


def test_policy_enforces_schedule_shape_discovery_not_adaptive_control() -> None:
    policy = json.loads(
        subprocess.check_output(
            [
                "python3",
                "-c",
                "import json,sys,yaml; print(json.dumps(yaml.safe_load(open(sys.argv[1]))))",
                str(TASK_ROOT / "policy.yaml"),
            ],
            text=True,
        )
    )
    constraints = policy["constraints"]
    assert constraints["schedule_inputs"] == ["normalized_progress"]
    assert constraints["peak_multiplier_maximum"] == 1.0
    assert constraints["persistent_state_scalars_maximum"] == 0


def test_policy_checker_rejects_hidden_schedule_metadata_inputs(tmp_path: Path) -> None:
    relative = Path("examples/training/lr_schedule/runtime/lr_schedule_candidate.py")
    clean = tmp_path / "clean"
    candidate = tmp_path / "candidate"
    (clean / relative).parent.mkdir(parents=True)
    (candidate / relative).parent.mkdir(parents=True)
    (clean / relative).write_text(
        "class S:\n    def multiplier(self, progress): return 1.0\n"
        "def build_schedule(): return S()\n"
    )
    (candidate / relative).write_text(
        "import os\n"
        "class S:\n"
        "    def multiplier(self, progress):\n"
        "        return 1.0 if os.environ.get('LR_SCHEDULE_SCALE') else progress\n"
        "def build_schedule(): return S()\n"
    )
    report = tmp_path / "report.json"
    result = subprocess.run(
        [
            "python3",
            str(TASK_ROOT / "tests" / "policy_check.py"),
            "--policy",
            str(TASK_ROOT / "tests" / "policy.yaml"),
            "--clean",
            str(clean),
            "--candidate",
            str(candidate),
            "--report",
            str(report),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    payload = json.loads(report.read_text())
    assert payload["semantic_violations"]
    assert "os" in json.dumps(payload["semantic_violations"])


def test_policy_checker_rejects_persistent_schedule_state(tmp_path: Path) -> None:
    relative = Path("examples/training/lr_schedule/runtime/lr_schedule_candidate.py")
    clean = tmp_path / "clean"
    candidate = tmp_path / "candidate"
    (clean / relative).parent.mkdir(parents=True)
    (candidate / relative).parent.mkdir(parents=True)
    source = (
        "class S:\n"
        "    calls = 0\n"
        "    def multiplier(self, progress):\n"
        "        self.calls += 1\n"
        "        return progress\n"
        "def build_schedule(): return S()\n"
    )
    (clean / relative).write_text(source.replace("        self.calls += 1\n", ""))
    (candidate / relative).write_text(source)
    report = tmp_path / "report.json"
    result = subprocess.run(
        [
            "python3",
            str(TASK_ROOT / "tests" / "policy_check.py"),
            "--policy",
            str(TASK_ROOT / "tests" / "policy.yaml"),
            "--clean",
            str(clean),
            "--candidate",
            str(candidate),
            "--report",
            str(report),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    violations = json.loads(report.read_text())["semantic_violations"]
    assert "persistent mutation" in " ".join(violations)


def test_hypothesis_validator_requires_mechanism_and_discriminating_ablation(
    tmp_path: Path,
) -> None:
    ladder = state_module()
    hypothesis = tmp_path / "hypothesis.md"
    hypothesis.write_text(
        """# Candidate

## Mechanism
Redistribute LR mass from the stable phase to a smooth tail.

## Exact schedule
A single bounded equation of normalized progress.

## Predicted trajectory
Lower late-training Paloma BPB with no early regression.

## Rejection criterion
Reject if trajectory area is non-positive.

## Discriminating ablation
Remove the smooth-tail component while preserving peak LR.

## Planned experiments
Develop on E0/E2, freeze, then transfer unchanged.
"""
    )
    sections = ladder.validate_hypothesis(hypothesis)
    assert sections["mechanism"].startswith("Redistribute")

    tuning_only = tmp_path / "tuning.md"
    tuning_only.write_text(
        """# Candidate

## Mechanism
Grid search warmup and decay ratios.

## Exact schedule
Choose the best parameter combination.

## Predicted trajectory
The best official score will improve.

## Rejection criterion
Reject parameter settings with lower official scores.

## Discriminating ablation
Try a neighboring parameter.

## Planned experiments
Bayesian optimization over all rungs.
"""
    )
    with pytest.raises(ValueError, match="tuning-only"):
        ladder.validate_hypothesis(tuning_only)


def test_confirmation_scales_freeze_one_hypothesis_and_source(tmp_path: Path) -> None:
    ladder = state_module()
    first = ladder.bind_confirmation_source(
        tmp_path,
        scale="E1",
        source_hash="a" * 64,
        hypothesis_hash="b" * 64,
    )
    repeated = ladder.bind_confirmation_source(
        tmp_path,
        scale="E5",
        source_hash="a" * 64,
        hypothesis_hash="b" * 64,
    )
    assert repeated == first
    with pytest.raises(ValueError, match="frozen"):
        ladder.bind_confirmation_source(
            tmp_path,
            scale="E3",
            source_hash="c" * 64,
            hypothesis_hash="b" * 64,
        )
    assert ladder.bind_confirmation_source(
        tmp_path,
        scale="E0",
        source_hash="c" * 64,
        hypothesis_hash="d" * 64,
    ) is None


def test_manifest_requires_six_exact_final_trajectories() -> None:
    ladder = state_module()
    payload = tiny_manifest()
    assert ladder.validate_baseline_manifest(payload) is payload["scales"]
    broken = json.loads(json.dumps(payload))
    broken["scales"]["E4"]["reference_status"] = "recorded_nonfinal_scoring_update"
    with pytest.raises(ValueError, match="E4.*exact-final"):
        ladder.validate_baseline_manifest(broken)
    broken = json.loads(json.dumps(payload))
    broken["scales"]["E5"]["paloma_trajectory"] = broken["scales"]["E5"]["paloma_trajectory"][:1]
    with pytest.raises(ValueError, match="E5.*two Paloma"):
        ladder.validate_baseline_manifest(broken)


def test_reward_implements_final_and_trajectory_gates() -> None:
    ladder = state_module()
    baselines = ladder.validate_baseline_manifest(tiny_manifest())
    result = ladder.score_ladder(baselines, candidate_scales())
    assert result["passes"] is True
    assert result["reward"] > 1.0
    assert result["final_micro_gain"] == pytest.approx(1.0 / 0.98)
    assert result["final_macro_gain"] == pytest.approx(1.2 / 1.18)
    assert all(item["trajectory_area"] > 0.0 for item in result["scales"].values())

    regression = candidate_scales((1.006, 0.98))
    failed = ladder.score_ladder(baselines, regression)
    assert failed["passes"] is False
    assert failed["reward"] == 0.0
    assert failed["guards"]["intermediate_micro"] is False


def test_only_the_lr_candidate_counts_as_participant_source() -> None:
    ladder = state_module()
    allowed = "examples/training/lr_schedule/runtime/lr_schedule_candidate.py"
    assert ladder.candidate_source_changed([allowed]) is True
    assert ladder.candidate_source_changed([]) is False
    assert ladder.candidate_source_changed(
        ["3rdparty/Megatron-LM/megatron/core/optimizer/optimizer.py"]
    ) is False


def test_per_run_integrity_requires_a_complete_schedule_trace() -> None:
    ladder = state_module()
    scale = "E0"
    metrics = {
        "paloma_bits_per_byte": 1.0,
        "paloma_macro_bits_per_byte": 1.2,
        "cost_trace_rows": EXPECTED_SCALES[scale]["iterations"],
        "schedule_trace_rows": EXPECTED_SCALES[scale]["iterations"],
        "schedule_source_consistent": True,
    }
    guards = ladder.metric_guards(
        scale, metrics, EXPECTED_SCALES[scale]["parameters"]
    )
    assert all(guards.values())
    metrics["schedule_trace_rows"] -= 1
    assert ladder.metric_guards(
        scale, metrics, EXPECTED_SCALES[scale]["parameters"]
    )["complete_schedule_trace"] is False
    assert ladder.candidate_source_changed(
        ["examples/training/lr_schedule/runtime/lr_schedule_controller.py"]
    ) is False


def test_submission_validator_uses_lr_artifacts_and_manifest_v3() -> None:
    validator = (TASK_ROOT / "tests" / "validate_submission.py").read_text()
    assert 'baseline_payload.get("version") != 3' in validator
    assert 'root / "LR_SCHEDULE.md"' in validator
    assert "OPTIMIZER.md" not in validator
    assert "mechanism_ablation" in validator
    assert "confirmation_freeze" in validator
    assert 'root / "science_freeze.json"' in validator
    assert 'row.get("state")' in validator
    assert "multiplier > 1.0" in validator
    assert "lr_schedule_candidate.py" in validator


def test_runners_and_cluster_scripts_are_syntax_valid() -> None:
    scripts = [
        TOOLS / "lr_schedule_ladder_profile.sh",
        TOOLS / "run_lr_schedule_ladder.sh",
        TOOLS / "run_lr_schedule_candidate_ladder.sh",
        TASK_ROOT / "environment" / "materialize_project.sh",
        TASK_ROOT / "cluster" / "preflight.sh",
        TASK_ROOT / "cluster" / "build_images.sh",
        TASK_ROOT / "cluster" / "launch_harbor_lsf.sh",
    ]
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)
    single = (TOOLS / "run_lr_schedule_ladder.sh").read_text()
    assert "host_lease.py" in single
    assert "--eval-only" in single
    assert "--resume-checkpoint" in single
    assert "export LR_SCHEDULE_SOURCE_INVENTORY_SHA256=${source_hash}" in single
    parallel = (TOOLS / "run_lr_schedule_candidate_ladder.sh").read_text()
    assert "E0,E1,E2,E3,E4,E5" in parallel


def test_parallel_runner_accepts_only_seed_zero_and_selected_scales(tmp_path: Path) -> None:
    hypothesis = tmp_path / "hypothesis.md"
    hypothesis.write_text(
        """# Fixture

## Mechanism
A smooth late tail reallocates LR mass.

## Exact schedule
One bounded function of normalized progress.

## Predicted trajectory
Late Paloma improves without early regression.

## Rejection criterion
Reject non-positive aggregate trajectory gain.

## Discriminating ablation
Remove the late-tail term.

## Planned experiments
Develop on E0/E2 and freeze before transfer.
"""
    )
    runner = TOOLS / "run_lr_schedule_candidate_ladder.sh"
    result = subprocess.run(
        [
            str(runner),
            "--candidate-id",
            "fixture",
            "--seed",
            "0",
            "--scales",
            "E0,E2",
            "--hypothesis-file",
            str(hypothesis),
            "--dry-run",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "scale: E0" in result.stdout
    assert "scale: E2" in result.stdout
    assert "scale: E1" not in result.stdout
    rejected = subprocess.run(
        [
            str(runner),
            "--candidate-id",
            "fixture-bad-seed",
            "--seed",
            "1",
            "--hypothesis-file",
            str(hypothesis),
            "--dry-run",
        ],
        text=True,
        capture_output=True,
    )
    assert rejected.returncode != 0
    assert "seed" in rejected.stderr.lower()


def test_environment_materializer_uses_correct_spelling() -> None:
    materializer = (TASK_ROOT / "environment" / "materialize_project.sh").read_text()
    assert "examples/training/lr_schedule" in materializer
    assert "postional_encoding" not in materializer


def test_verifier_bundle_is_complete_and_task_specific() -> None:
    required = [
        TASK_ROOT / "tests" / "learning_rate_tests.def",
        TASK_ROOT / "tests" / "test.sh",
        TASK_ROOT / "tests" / "policy_check.py",
        TASK_ROOT / "tests" / "score.py",
    ]
    assert all(path.is_file() for path in required)
    for path in required:
        text = path.read_text()
        assert "optimizer_scaling_task" not in text
        assert "run_optimizer" not in text


def test_baseline_endpoint_evaluator_is_exact_and_lsf_only() -> None:
    path = TASK_ROOT / "cluster" / "evaluate_baseline_endpoints_lsf.sh"
    assert path.is_file()
    source = path.read_text()
    subprocess.run(["bash", "-n", str(path)], check=True)
    assert "iter_0056477" in source
    assert "iter_0035510" in source
    assert "mt-lr-base-e4-yuetai" in source
    assert "mt-lr-base-e5-yuetai" in source
    assert "EVAL_ONLY=1" in source
    assert "MARIN_LADDER_GPU_COUNT_OVERRIDE" in source
    assert "E4) slots=8" in source
    assert "E5) slots=16" in source
    assert "--run" in source


def test_offline_baseline_builder_merges_exact_endpoints(tmp_path: Path) -> None:
    builder = TASK_ROOT / "baselines" / "build_lr_schedule_baselines.py"
    assert builder.is_file()
    assert "wandb.Api" not in builder.read_text()
    endpoints = []
    for scale, micro in (("E4", 0.9), ("E5", 0.8)):
        path = tmp_path / f"{scale}.json"
        path.write_text(
            json.dumps(
                {
                    "paloma_aggregate": {
                        "bits_per_byte": micro,
                        "macro_bits_per_byte": micro + 0.1,
                        "subsets": 16,
                    }
                }
            )
        )
        endpoints.append(path)
    output = tmp_path / "baselines.json"
    subprocess.run(
        [
            sys.executable,
            str(builder),
            "--source",
            str(TASK_ROOT / "baselines" / "lr_schedule_baselines.json"),
            "--e4-result",
            str(endpoints[0]),
            "--e5-result",
            str(endpoints[1]),
            "--output",
            str(output),
        ],
        check=True,
    )
    payload = json.loads(output.read_text())
    state_module().validate_baseline_manifest(payload)
    for record in payload["scales"].values():
        provenance = record["source_provenance"]
        assert provenance["bridge_commit"] == "accefd7b3a448ab45b015edc04c9d3b70f7cb3b7"
        assert provenance["task_infrastructure_commit"] == "3a6c6fb3811a10f532d26be7dd032457795a8e41"


def test_baseline_file_is_release_ready() -> None:
    payload = json.loads((TASK_ROOT / "baselines" / "lr_schedule_baselines.json").read_text())
    state_module().validate_baseline_manifest(payload)
