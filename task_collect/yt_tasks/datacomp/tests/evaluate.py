#!/usr/bin/env python3
"""Frozen DataComp verifier: subset -> pristine reshard/train/evaluate."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


UPSTREAM_COMMIT = "4a8df1992566ef8334773f7152e1855b1f716162"
PROJECT = Path("/opt/project")
OUTPUT = Path("/app/output")
LOG_ROOT = Path("/logs/verifier")
WORK = LOG_ROOT / "work"
DATA_ROOT = Path(os.environ.get("DATACOMP_ROOT", "/datasets/datacomp/small"))
SHARDS = DATA_ROOT / "commonpool" / "shards"
EVAL_DATA = DATA_ROOT / "eval"
MANIFESTS = DATA_ROOT / "manifests"
AVAILABLE = MANIFESTS / "available_uids.npy"
SUBMISSION = OUTPUT / "submission" / "selected_uids.npy"
UID_DTYPE = np.dtype("u8,u8")
EXPECTED_GPUS = 8
EXPECTED_RESULTS = 40
EXPECTED_SCORED = 38


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(name: str) -> tuple[dict[str, Any], str]:
    path = MANIFESTS / name
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "complete":
        raise RuntimeError(f"asset manifest is not complete: {path}")
    if value.get("upstream_commit") != UPSTREAM_COMMIT:
        raise RuntimeError(f"asset manifest upstream mismatch: {path}")
    return value, sha256(path)


def validate_submission() -> dict[str, Any]:
    if not SUBMISSION.is_file():
        raise RuntimeError(f"missing submission: {SUBMISSION}")
    selected = np.load(SUBMISSION, mmap_mode="r", allow_pickle=False)
    available = np.load(AVAILABLE, mmap_mode="r", allow_pickle=False)
    for label, array in (("selected", selected), ("available", available)):
        if array.ndim != 1 or array.dtype != UID_DTYPE:
            raise RuntimeError(
                f"{label} UID array must be 1-D {UID_DTYPE}, got {array.shape} {array.dtype}"
            )
    if not 0 < len(selected) <= len(available):
        raise RuntimeError(
            f"invalid selected UID count: {len(selected)} of {len(available)}"
        )
    if len(selected) > 1 and np.any(selected[1:] <= selected[:-1]):
        raise RuntimeError("submitted UIDs are not strictly sorted and unique")
    positions = np.searchsorted(available, selected)
    in_bounds = positions < len(available)
    matches = np.zeros(len(selected), dtype=bool)
    matches[in_bounds] = available[positions[in_bounds]] == selected[in_bounds]
    if not bool(matches.all()):
        raise RuntimeError("submission contains UIDs outside the sealed successful pool")

    selected_digest = sha256(SUBMISSION)
    provenance = json.loads((OUTPUT / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("selected_uids_sha256") != selected_digest:
        raise RuntimeError("provenance selected UID digest mismatch")
    if int(provenance.get("selected_uid_count", -1)) != len(selected):
        raise RuntimeError("provenance selected UID count mismatch")
    if provenance.get("web_search") != "disabled":
        raise RuntimeError("provenance web_search must be disabled")
    return {
        "selected_uid_count": int(len(selected)),
        "available_uid_count": int(len(available)),
        "selected_fraction": float(len(selected) / len(available)),
        "selected_uids_sha256": selected_digest,
        "available_uids_sha256": sha256(AVAILABLE),
    }


def run(command: list[str], *, cwd: Path, log_name: str) -> float:
    log_path = LOG_ROOT / f"{log_name}.log"
    started = time.time()
    print(f"[verifier] {' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return time.time() - started


def main() -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if torch.cuda.device_count() != EXPECTED_GPUS:
        raise RuntimeError(
            f"frozen small verifier requires exactly {EXPECTED_GPUS} visible GPUs, "
            f"found {torch.cuda.device_count()}"
        )
    source_head = subprocess.check_output(
        ["git", "-C", str(PROJECT), "rev-parse", "HEAD"], text=True
    ).strip()
    if source_head != UPSTREAM_COMMIT:
        raise RuntimeError(f"clean verifier source mismatch: {source_head}")
    if subprocess.check_output(
        ["git", "-C", str(PROJECT), "status", "--short"], text=True
    ).strip():
        raise RuntimeError("clean verifier source is not pristine")

    subset = validate_submission()
    metadata_manifest, metadata_manifest_sha = load_manifest("metadata.json")
    image_manifest, image_manifest_sha = load_manifest("commonpool-images.json")
    eval_manifest, eval_manifest_sha = load_manifest("evaluation.json")
    if image_manifest["available_uids"]["sha256"] != subset["available_uids_sha256"]:
        raise RuntimeError("available UID array does not match image manifest")
    if int(metadata_manifest["pool_rows"]) != 12_800_000:
        raise RuntimeError("metadata manifest is not the official small pool")
    if int(eval_manifest["task_count"]) != EXPECTED_RESULTS:
        raise RuntimeError("evaluation manifest task count mismatch")

    if WORK.exists():
        shutil.rmtree(WORK)
    data_dir = WORK / "data"
    train_root = WORK / "train"
    eval_root = WORK / "eval"
    data_dir.mkdir(parents=True)
    train_root.mkdir(parents=True)
    eval_root.mkdir(parents=True)

    reshard_seconds = run(
        [
            sys.executable,
            str(PROJECT / "resharder.py"),
            "--input-dir",
            str(SHARDS),
            "--output-dir",
            str(data_dir),
            "--subset-file",
            str(SUBMISSION),
            "--num-workers",
            "32",
        ],
        cwd=PROJECT,
        log_name="reshard",
    )
    reshard_meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    if int(reshard_meta["output_count"]) != subset["selected_uid_count"]:
        raise RuntimeError(
            "pristine resharder did not find every submitted record: "
            f"{reshard_meta['output_count']} != {subset['selected_uid_count']}"
        )

    train_seconds = run(
        [
            "torchrun",
            "--standalone",
            "--nproc_per_node=8",
            str(PROJECT / "train.py"),
            "--scale",
            "small",
            "--data_dir",
            str(data_dir),
            "--output_dir",
            str(train_root),
            "--exp_name",
            "datacomp-small",
            "--workers",
            "4",
            "--precision",
            "amp",
            "--num_checkpoints",
            "5",
            "--seed",
            "0",
        ],
        cwd=PROJECT,
        log_name="train",
    )
    trained_experiment = train_root / "datacomp-small"
    info_path = trained_experiment / "info.pkl"
    with info_path.open("rb") as handle:
        training_info = pickle.load(handle)
    checkpoint = Path(training_info["checkpoint"])
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise RuntimeError(f"frozen training did not produce a checkpoint: {checkpoint}")
    expected_config = {
        "batch_size": 4096,
        "learning_rate": 5e-4,
        "train_num_samples": 12_800_000,
        "warmup": 500,
        "model": "ViT-B-32",
        "beta2": None,
    }
    if training_info.get("scale") != "small" or training_info.get("scale_config") != expected_config:
        raise RuntimeError(f"frozen training config mismatch: {training_info}")

    eval_seconds = run(
        [
            sys.executable,
            str(PROJECT / "evaluate.py"),
            "--track",
            "filtering",
            "--train_output_dir",
            str(trained_experiment),
            "--output_dir",
            str(eval_root),
            "--data_dir",
            str(EVAL_DATA),
            "--samples",
            str(SUBMISSION),
            "--skip_hf",
            "--skip_db",
            "--skip_notification",
        ],
        cwd=PROJECT,
        log_name="evaluate",
    )
    results_path = eval_root / "eval_results.jsonl"
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != EXPECTED_RESULTS:
        raise RuntimeError(f"official evaluation produced {len(rows)} results, expected {EXPECTED_RESULTS}")

    sys.path.insert(0, str(PROJECT))
    from aggregate_scores import get_aggregate_scores

    aggregates = {key: float(value) for key, value in get_aggregate_scores(results_path).items()}
    if not math.isfinite(aggregates["Average"]):
        raise RuntimeError("official aggregate Average is not finite")
    scored = [row for row in rows if row["metrics"].get("main_metric") is not None]
    if len(scored) != EXPECTED_SCORED:
        raise RuntimeError(f"official scored result count is {len(scored)}, expected {EXPECTED_SCORED}")

    shutil.copy2(results_path, LOG_ROOT / "eval_results.jsonl")
    data_bytes = sum(path.stat().st_size for path in data_dir.rglob("*") if path.is_file())
    metrics = {
        "metric": "official DataComp aggregate Average",
        "upstream_commit": UPSTREAM_COMMIT,
        "scale": "small",
        "track": "filtering",
        "model": "ViT-B-32",
        "seed": 0,
        "world_size": EXPECTED_GPUS,
        "global_batch_size": 4096,
        "nominal_training_samples": 12_800_000,
        "warmup": 500,
        "learning_rate": 5e-4,
        "precision": "amp",
        "num_checkpoints": 5,
        **subset,
        "successful_pool_records": int(image_manifest["successful_records"]),
        "failed_pool_records": int(image_manifest["unavailable_or_failed_records"]),
        "resharded_bytes": data_bytes,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "result_count": len(rows),
        "scored_result_count": len(scored),
        "aggregates": aggregates,
        "raw_main_metrics": {
            row["dataset"]: float(row["metrics"]["main_metric"])
            for row in scored
        },
        "elapsed_seconds": {
            "reshard": round(reshard_seconds, 3),
            "train": round(train_seconds, 3),
            "evaluate": round(eval_seconds, 3),
            "total": round(reshard_seconds + train_seconds + eval_seconds, 3),
        },
        "asset_manifest_sha256": {
            "metadata": metadata_manifest_sha,
            "commonpool_images": image_manifest_sha,
            "evaluation": eval_manifest_sha,
        },
        "verifier_commands": "see reshard.log, train.log, and evaluate.log",
    }
    (LOG_ROOT / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Work data and checkpoint are verifier-generated and can be reconstructed
    # exactly from the sealed inputs, submission digest, and retained logs.
    shutil.rmtree(WORK)


if __name__ == "__main__":
    main()
