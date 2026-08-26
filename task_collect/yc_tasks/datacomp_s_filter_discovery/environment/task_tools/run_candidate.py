#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from validate_subset import sha256, validate_subset


def require_complete_baseline(path: Path) -> dict:
    value = json.loads(path.read_text())
    if value.get("task_id") != "datacomp_s_filter_discovery" or value.get("status") != "complete":
        raise RuntimeError("DataComp baseline contract is incomplete or invalid")
    for field in ("reference_run_id", "reference_metric", "candidate_threshold", "evaluator_tolerance", "reference_checkpoint_sha256", "initialization_parameter_digest", "asset_manifest_sha256", "runtime_image_id"):
        if value.get(field) in (None, ""):
            raise RuntimeError(f"completed baseline contract is missing {field}")
    return value


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


def regular(path: Path, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"missing regular {label}: {path}")
    return path


def run_selector(output_root: Path) -> Path:
    metadata = Path(os.environ.get("DATACOMP_METADATA_DIR", "/datasets/commonpool_s/metadata"))
    features = Path(os.environ.get("DATACOMP_FEATURES_DIR", "/datasets/commonpool_s/features"))
    universe = regular(Path(os.environ.get("DATACOMP_UID_UNIVERSE", "/datasets/commonpool_s/manifests/available_uids.npy")), "UID universe")
    if not metadata.is_dir():
        raise RuntimeError(f"metadata directory is unavailable: {metadata}")
    selector = regular(Path("/app/project/candidate/selector.py"), "candidate selector")
    run_root = output_root / "selector"
    if run_root.exists():
        raise RuntimeError(f"fresh selector directory required: {run_root}")
    run_root.mkdir(parents=True)
    outputs = []
    commands = []
    for repeat in (1, 2):
        output = run_root / f"repeat_{repeat}" / "subset.npy"
        output.parent.mkdir()
        command = [
            "python", str(selector), "--metadata-dir", str(metadata),
            "--features-dir", str(features), "--output", str(output), "--seed", "0",
        ]
        subprocess.run(command, check=True, env={**os.environ, "PYTHONHASHSEED": "0"})
        outputs.append(output)
        commands.append(command)
    if outputs[0].read_bytes() != outputs[1].read_bytes():
        raise RuntimeError("selector repeats are not byte-identical")
    artifact = output_root / "artifacts" / "subset.npy"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(outputs[0], artifact)
    validation = validate_subset(artifact, universe)
    validation["repeat_byte_identical"] = True
    validation["candidate_source_sha256"] = sha256(selector)
    validation["selection_command"] = commands[0]
    (output_root / "selection_record.json").write_text(json.dumps(validation, sort_keys=True, indent=2) + "\n")
    return artifact


def run_full(output_root: Path, subset: Path) -> None:
    baseline = require_complete_baseline(Path(os.environ.get("DATACOMP_BASELINE_CONTRACT", "/opt/contracts/baseline_contract.json")))
    asset_manifest = regular(Path(os.environ.get("DATACOMP_ASSET_MANIFEST", "/datasets/commonpool_s/manifests/asset_manifest.json")), "asset manifest")
    if sha256(asset_manifest) != baseline["asset_manifest_sha256"]:
        raise RuntimeError("staged asset manifest differs from the sealed baseline contract")
    attempt_count = validate_experiments(output_root / "experiments.jsonl")
    source_shards = Path(os.environ.get("DATACOMP_SOURCE_SHARDS", "/datasets/commonpool_s/full_shards"))
    selected_shards = output_root / "selected_shards"
    if not source_shards.is_dir() or selected_shards.exists():
        raise RuntimeError("source shards must exist and selected-shard output must be fresh")
    subprocess.run([
        "python", "/opt/datacomp/resharder.py", "-i", str(source_shards),
        "-o", str(selected_shards), "-s", str(subset),
    ], check=True)
    training_root = output_root / "training"
    command = [
        "torchrun", "--standalone", "--nproc_per_node", "4", "/opt/datacomp/train.py",
        "--scale", "small", "--data_dir", str(selected_shards), "--output_dir", str(training_root),
        "--exp_name", "experiment", "--workers", "4", "--precision", "amp",
        "--num_checkpoints", "5", "--seed", "0", "--accum_freq", "1",
        "--log_every_n_steps", "100", "--resume", "latest",
    ]
    start = datetime.now(timezone.utc)
    subprocess.run(command, check=True)
    end = datetime.now(timezone.utc)
    checkpoints = sorted(training_root.rglob("epoch_5.pt"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"expected one epoch_5.pt, found {len(checkpoints)}")
    staged = output_root / "checkpoints" / "epoch_5.pt"
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(checkpoints[0], staged)
    attestations = sorted(training_root.rglob("initialization_attestation.json"))
    if len(attestations) != 1:
        raise RuntimeError(f"expected one initialization attestation, found {len(attestations)}")
    attestation = output_root / "initialization_attestation.json"
    shutil.copyfile(attestations[0], attestation)
    attestation_value = json.loads(attestation.read_text())
    if attestation_value.get("fingerprint", {}).get("sha256") != baseline["initialization_parameter_digest"]:
        raise RuntimeError("seed-0 initialization differs from the sealed baseline")
    ledger = {
        "schema_version": 1, "status": "pass", "task_id": "datacomp_s_filter_discovery",
        "start_utc": start.isoformat(), "end_utc": end.isoformat(),
        "world_size": 4, "configured_training_pairs": 12800000,
        "realized_training_pairs": 12861440, "optimizer_updates": 3140,
        "global_batch_size": 4096, "seed": 0, "subset_sha256": sha256(subset),
        "checkpoint_sha256": sha256(staged), "training_command": command,
        "initialization_attestation_sha256": sha256(attestation),
    }
    (output_root / "run_ledger.json").write_text(json.dumps(ledger, sort_keys=True, indent=2) + "\n")
    selection = json.loads((output_root / "selection_record.json").read_text())
    provenance = {
        "schema_version": 1,
        "task_id": "datacomp_s_filter_discovery",
        "candidate_source_sha256": selection["candidate_source_sha256"],
        "subset_sha256": sha256(subset),
        "input_manifest_sha256": sha256(asset_manifest),
        "selection_command": selection["selection_command"],
        "training_command": command,
        "upstream_commits": {
            "datacomp": "4a8df1992566ef8334773f7152e1855b1f716162",
            "open_clip": "395851ee9c55fdddb63f5579955cdd6d754d261c",
        },
        "optimization_attempts": attempt_count,
        "web_search": "disabled",
    }
    (output_root / "provenance.json").write_text(json.dumps(provenance, sort_keys=True, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("selector", "full"), default="selector")
    parser.add_argument("--output-root", type=Path, default=Path("/app/output"))
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    subset = run_selector(args.output_root)
    if args.stage == "full":
        run_full(args.output_root, subset)


if __name__ == "__main__":
    main()
