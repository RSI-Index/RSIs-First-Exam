#!/usr/bin/env python3
"""Validate or execute one fixed-budget LiT/COCO candidate run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from lit_coco_autoresearch import load_candidate_config


TASK_ID = "bigvision_lit_coco_alignment"
UPSTREAM_COMMIT = "8921d5141504390a8a4f7b2dacb3b3c042237290"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode() + b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def regular(path: Path, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"missing regular {label}: {path}")
    return path


def complete_baseline(path: Path) -> dict:
    value = json.loads(regular(path, "baseline contract").read_text())
    if value.get("task_id") != TASK_ID or value.get("status") != "complete":
        raise RuntimeError("Big Vision baseline contract is incomplete or invalid")
    required = (
        "reference_run_id", "reference_metric", "imagenet_retention_lower_bound",
        "trainable_parameter_ceiling", "training_flop_ceiling", "image_parameter_digest",
        "asset_manifest_sha256", "runtime_image_id",
    )
    for field in required:
        if value.get(field) in (None, ""):
            raise RuntimeError(f"completed baseline contract is missing {field}")
    return value


def validate_assets(baseline: dict) -> Path:
    manifest = regular(
        Path(os.environ.get("BIGVISION_ASSET_MANIFEST", "/datasets/initializers/harbor_manifest.json")),
        "asset manifest",
    )
    if sha256(manifest) != baseline["asset_manifest_sha256"]:
        raise RuntimeError("staged asset manifest differs from the sealed baseline contract")
    regular(Path(os.environ.get("BIGVISION_IMAGE_INIT", "/datasets/initializers/vit_b16_augreg.npz")), "image initializer")
    text_root = Path(os.environ.get("BIGVISION_TEXT_INIT", "/datasets/initializers/bert_base"))
    regular(text_root / "bert_model.ckpt.index", "BERT initializer index")
    regular(text_root / "vocab.txt", "BERT vocabulary")
    tfds_root = Path(os.environ.get("TFDS_DATA_DIR", "/datasets/tfds"))
    if not tfds_root.is_dir():
        raise RuntimeError(f"missing staged TFDS root: {tfds_root}")
    return manifest


def validate_experiments(path: Path) -> int:
    regular(path, "experiments ledger")
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid experiments.jsonl line {line_number}") from exc
        if not isinstance(row, dict) or not row.get("hypothesis") or not row.get("result"):
            raise RuntimeError(f"experiments.jsonl line {line_number} lacks hypothesis/result")
        rows.append(row)
    if not rows:
        raise RuntimeError("experiments.jsonl must truthfully record at least one optimization attempt")
    return len(rows)


def validate_candidate(candidate_root: Path, output_root: Path) -> dict:
    sources = [
        regular(candidate_root / "text_tower.py", "candidate text tower"),
        regular(candidate_root / "alignment_loss.py", "candidate alignment loss"),
    ]
    config_path = regular(candidate_root / "config.toml", "candidate config")
    config = load_candidate_config(config_path)
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "candidate_config.canonical.json").write_text(canonical + "\n")
    return {
        "candidate_source_sha256": source_digest(sources),
        "candidate_config_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "candidate_config": config,
    }


def run_full(candidate_root: Path, output_root: Path, identity: dict) -> None:
    baseline = complete_baseline(Path(os.environ.get("BIGVISION_BASELINE_CONTRACT", "/opt/contracts/baseline_contract.json")))
    asset_manifest = validate_assets(baseline)
    attempt_count = validate_experiments(output_root / "experiments.jsonl")
    capacity_path = output_root / "capacity_record.json"
    subprocess.run(["python3", "/task-tools/capacity_probe.py", "--output", str(capacity_path)], check=True)
    capacity = json.loads(capacity_path.read_text())
    if capacity["trainable_parameter_count"] > int(baseline["trainable_parameter_ceiling"]):
        raise RuntimeError("candidate exceeds the sealed trainable-parameter ceiling")
    if capacity["training_flops"] > float(baseline["training_flop_ceiling"]):
        raise RuntimeError("candidate exceeds the sealed training-FLOP ceiling")

    training_root = output_root / "training"
    if training_root.exists():
        raise RuntimeError(f"fresh training directory required: {training_root}")
    training_root.mkdir()
    command = [
        "python3", "/task-tools/contrastive_autoresearch.py",
        "--config=/task-tools/lit_coco_autoresearch.py",
        f"--workdir={training_root}",
    ]
    start = datetime.now(timezone.utc)
    subprocess.run(command, check=True, env={**os.environ, "BIGVISION_FINAL_EVAL": "0"})
    end = datetime.now(timezone.utc)
    checkpoint = regular(training_root / "checkpoint.npz", "fixed-trainer checkpoint")
    artifact = output_root / "artifacts" / "model.npz"
    export_record = output_root / "model_export.json"
    subprocess.run([
        "python3", "/task-tools/export_model.py", "--checkpoint", str(checkpoint),
        "--output", str(artifact), "--record", str(export_record),
    ], check=True)
    exported = json.loads(export_record.read_text())
    if exported["image_parameter_digest"] != baseline["image_parameter_digest"]:
        raise RuntimeError("exported image parameters are not byte-identical to the sealed baseline image tree")

    config_artifact = output_root / "artifacts" / "config.json"
    config_artifact.write_text((output_root / "candidate_config.canonical.json").read_text())
    ledger = {
        "schema_version": 1,
        "status": "pass",
        "task_id": TASK_ID,
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "world_size": 8,
        "global_batch_size": 4096,
        "optimizer_steps": 5000,
        "pair_exposures": 20_480_000,
        "optimizer": "scale_by_adam",
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "warmup_steps": 150,
        "seed": 0,
        "upstream_commit": UPSTREAM_COMMIT,
        "training_command": command,
        "model_sha256": exported["model_sha256"],
        "image_parameter_digest": exported["image_parameter_digest"],
        "capacity_record_sha256": sha256(capacity_path),
    }
    (output_root / "run_ledger.json").write_text(json.dumps(ledger, sort_keys=True, indent=2) + "\n")
    provenance = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "candidate_source_sha256": identity["candidate_source_sha256"],
        "candidate_config_sha256": identity["candidate_config_sha256"],
        "model_sha256": exported["model_sha256"],
        "input_manifest_sha256": sha256(asset_manifest),
        "training_command": command,
        "upstream_commit": UPSTREAM_COMMIT,
        "optimization_attempts": attempt_count,
        "image_parameter_digest": exported["image_parameter_digest"],
        "capacity_record_sha256": sha256(capacity_path),
        "web_search": "disabled",
    }
    (output_root / "provenance.json").write_text(json.dumps(provenance, sort_keys=True, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("validate", "full"), default="validate")
    parser.add_argument("--candidate-root", type=Path, default=Path("/app/project/candidate"))
    parser.add_argument("--output-root", type=Path, default=Path("/app/output"))
    args = parser.parse_args()
    identity = validate_candidate(args.candidate_root, args.output_root)
    if args.stage == "full":
        run_full(args.candidate_root, args.output_root, identity)
    else:
        print(json.dumps({"status": "pass", **{key: identity[key] for key in ("candidate_source_sha256", "candidate_config_sha256")}}, sort_keys=True))


if __name__ == "__main__":
    main()
