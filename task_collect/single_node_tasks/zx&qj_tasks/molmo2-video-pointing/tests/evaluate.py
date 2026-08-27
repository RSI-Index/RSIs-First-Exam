#!/usr/bin/env python3
"""Frozen final evaluator for the molmo2 video pointing task.

Runs the pinned repository's release evaluation protocol on the staged
submission checkpoint and writes /app/output/verifier-metrics.json.

The candidate's workspace code (/app/project) is used to build and run the
submitted model, exactly as during training; the reward-integrity gate has
already verified that the protected eval-path files (olmo/eval/**,
launch_scripts/eval.py) are byte-identical to the clean checkout before
this script runs.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from pathlib import Path

# Container paths; overridable only so this file can be exercised outside the
# container. The task image always uses the defaults.
OUTPUT_ROOT = Path(os.environ.get("MVP_OUTPUT_ROOT", "/app/output"))
SUBMISSION = OUTPUT_ROOT / "submission"
METRICS_PATH = OUTPUT_ROOT / "verifier-metrics.json"
# Predictions and metrics are written outside the agent-writable output tree so
# a planted metrics.json cannot be picked up as this run's result.
EVAL_ROOT = Path(os.environ.get("MVP_EVAL_ROOT", "/logs/verifier/eval"))
CLEAN_PROJECT = Path(os.environ.get("MVP_CLEAN_PROJECT", "/opt/project"))
CANDIDATE_PROJECT = Path(os.environ.get("MVP_CANDIDATE_PROJECT", "/app/project"))
START_CHECKPOINT = os.environ.get(
    "MOLMO2_START_CHECKPOINT", "/datasets/molmo2-official/checkpoints/Molmo2-4B-Pretrain"
)
EVAL_TASKS = [
    ("vixmo_points_count_clip_63s", "val"),
    ("vixmo_points_point_eval", "val"),
]
MAX_VIDEO_FRAMES = 128
MAX_SEQUENCE_LENGTH = 16_384
PARAMETER_BUDGET_FACTOR = 1.005
SMOKE = os.environ.get("MVP_SINGLE_GPU_VERIFIER_SMOKE", "0") == "1"
SMOKE_MAX_EXAMPLES = int(os.environ.get("MVP_SMOKE_MAX_EXAMPLES", "8"))
NPROC = int(os.environ.get("MVP_VERIFIER_NPROC", "1" if SMOKE else "8"))
# The release protocol evaluates with a long runtime context window and bf16
# autocast (see launch_scripts/eval_molmo2.py); the <=16384 invariant above
# applies to the submitted training-time config, not to evaluation.
EVAL_SEQUENCE_LENGTH = 64000

COUNT_PARAMS_SNIPPET = """
import json, sys
from olmo.models.molmo.molmo import MolmoConfig
cfg = MolmoConfig.load(sys.argv[1] + "/config.yaml", key="model", validate_paths=False)
# Count parameters from a fresh instantiation; do not load any init weights.
cfg.llm.init_path = None
vit = getattr(getattr(cfg, "vision_backbone", None), "vit", None)
if vit is not None:
    vit.init_path = None
model = cfg.build_model()
print("PARAM_COUNT=" + str(sum(p.numel() for p in model.parameters())))
video = getattr(cfg.mm_preprocessor, "video", None)
print("MAX_FRAMES=" + str(getattr(video, "max_frames", None)))
print("MAX_SEQ_LEN=" + str(cfg.llm.max_sequence_length))
print("TOKENIZER=" + json.dumps(str(cfg.llm.tokenizer)))
"""


def fail(reason: str) -> None:
    METRICS_PATH.write_text(json.dumps({"error": reason}, indent=2) + "\n")
    print(f"verifier failure: {reason}", file=sys.stderr)
    sys.exit(1)


def inspect_model(project: Path, checkpoint: str) -> dict:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project)
    result = subprocess.run(
        [sys.executable, "-c", COUNT_PARAMS_SNIPPET, checkpoint],
        cwd=str(project),
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )
    info: dict = {}
    for line in result.stdout.splitlines():
        for key, cast in [
            ("PARAM_COUNT=", int),
            ("MAX_FRAMES=", str),
            ("MAX_SEQ_LEN=", int),
            ("TOKENIZER=", str),
        ]:
            if line.startswith(key):
                info[key.rstrip("=").lower()] = cast(line[len(key):])
    if "param_count" not in info:
        raise RuntimeError(f"model inspection did not report a parameter count: {result.stdout}")
    return info


def find_metric(metrics: dict, suffix: str):
    """Metric dicts may prefix keys with the task label; match by suffix."""
    if suffix in metrics:
        return metrics[suffix]
    for key, value in metrics.items():
        if key.split("/")[-1] == suffix and isinstance(value, (int, float)):
            return value
    return None


def run_eval(task: str, split: str, save_root: Path) -> dict:
    save_dir = save_root / f"{task}-{split}"
    save_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(CANDIDATE_PROJECT)
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    environment.setdefault("NCCL_TIMEOUT_MINUTES", "20")
    command = [
        "torchrun", "--standalone", "--nnodes=1", f"--nproc-per-node={NPROC}",
        str(CANDIDATE_PROJECT / "launch_scripts" / "eval.py"),
        str(SUBMISSION),
        f"--task={task}",
        f"--split={split}",
        "--device_batch_size=1",
        "--num_workers=4",
        f"--save_dir={save_dir}",
        "--precision=amp_bf16",
        "--skip_if_metrics_cached=False",
        f"--model.llm.max_sequence_length={EVAL_SEQUENCE_LENGTH}",
    ]
    if SMOKE:
        command.append(f"--max_examples={SMOKE_MAX_EXAMPLES}")
    subprocess.run(command, cwd=str(CANDIDATE_PROJECT), env=environment, check=True)
    metric_files = sorted(glob.glob(str(save_dir / "**" / "metrics.json"), recursive=True))
    if not metric_files:
        raise RuntimeError(f"no metrics.json produced for {task}:{split} under {save_dir}")
    payload = json.loads(Path(metric_files[-1]).read_text())
    return payload.get("metrics", payload)


def main() -> None:
    if not (SUBMISSION / "config.yaml").exists():
        fail("missing staged submission checkpoint (submission/config.yaml)")

    try:
        pristine = inspect_model(CLEAN_PROJECT, START_CHECKPOINT)
        candidate = inspect_model(CANDIDATE_PROJECT, str(SUBMISSION))
    except Exception as error:  # noqa: BLE001
        fail(f"model inspection failed: {error}")
        return

    parameter_budget = int(pristine["param_count"] * PARAMETER_BUDGET_FACTOR)
    if candidate["param_count"] > parameter_budget:
        fail(
            f"parameter budget exceeded: {candidate['param_count']} > {parameter_budget}"
        )
    if candidate["tokenizer"] != pristine["tokenizer"]:
        fail("tokenizer/vocabulary changed from the starting checkpoint")
    max_frames = candidate["max_frames"]
    if max_frames not in ("None", "null", "") and int(float(max_frames)) > MAX_VIDEO_FRAMES:
        fail(f"video max_frames {max_frames} exceeds the {MAX_VIDEO_FRAMES}-frame cap")
    if candidate["max_seq_len"] > MAX_SEQUENCE_LENGTH:
        fail(
            f"llm.max_sequence_length {candidate['max_seq_len']} exceeds {MAX_SEQUENCE_LENGTH}"
        )

    save_root = EVAL_ROOT
    results = {}
    for task, split in EVAL_TASKS:
        try:
            results[task] = run_eval(task, split, save_root)
        except Exception as error:  # noqa: BLE001
            fail(f"official release evaluation failed on {task}:{split}: {error}")
            return

    count_metrics = results["vixmo_points_count_clip_63s"]
    point_metrics = results["vixmo_points_point_eval"]
    count_correct = find_metric(count_metrics, "correct")
    point_f1 = find_metric(point_metrics, "f1")
    if count_correct is None or point_f1 is None:
        fail("release evaluation did not report the frozen metrics (correct, f1)")

    metrics = {
        "count_correct": float(count_correct),
        "count_close": find_metric(count_metrics, "close"),
        "point_precision": find_metric(point_metrics, "precision"),
        "point_recall": find_metric(point_metrics, "recall"),
        "point_f1": float(point_f1),
        "video_point_score": (float(count_correct) + float(point_f1)) / 2.0,
        "parameters": candidate["param_count"],
        "parameter_budget": parameter_budget,
        "pristine_parameters": pristine["param_count"],
        "eval_world_size": NPROC,
        "smoke": SMOKE,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
