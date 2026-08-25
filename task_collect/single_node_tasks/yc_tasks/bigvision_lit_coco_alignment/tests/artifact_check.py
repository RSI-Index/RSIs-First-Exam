#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import tomllib
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_digest(arrays: dict[str, np.ndarray], prefix: str) -> str:
    digest = hashlib.sha256()
    selected = [(key, value) for key, value in arrays.items() if key.startswith(prefix)]
    if not selected:
        raise ValueError(f"no arrays match {prefix}")
    for key, value in sorted(selected):
        array = np.ascontiguousarray(value)
        digest.update(key.encode() + b"\0")
        digest.update(array.dtype.str.encode() + b"\0")
        digest.update(json.dumps(array.shape).encode() + b"\0")
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def inspect_npz(path: Path) -> dict:
    arrays = {}
    with np.load(path, allow_pickle=False) as archive:
        if not archive.files:
            raise ValueError("empty model artifact")
        for key in archive.files:
            try:
                value = archive[key]
            except ValueError as exc:
                raise ValueError(f"object array is forbidden: {key}") from exc
            if value.dtype.hasobject:
                raise ValueError(f"object array is forbidden: {key}")
            if not np.issubdtype(value.dtype, np.number) or value.ndim == 0:
                raise ValueError(f"model array must be non-scalar numeric: {key}")
            if np.issubdtype(value.dtype, np.inexact) and not np.isfinite(value).all():
                raise ValueError(f"non-finite model array: {key}")
            arrays[key] = value
    record = {
        "array_count": len(arrays),
        "model_sha256": sha256(path),
        "trainable_parameter_count": sum(int(np.prod(value.shape)) for key, value in arrays.items() if not key.startswith("img/")),
    }
    if any(key.startswith("img/") for key in arrays):
        record["image_parameter_digest"] = array_digest(arrays, "img/")
    return record


def regular(path: Path) -> Path:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing regular artifact: {path}")
    return path


def source_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode() + b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def canonical_candidate_config(path: Path) -> tuple[dict, str]:
    value = tomllib.loads(path.read_text())
    tables = {
        "text": {"config", "head_zeroinit"},
        "model": {"temperature_init"},
        "loss": {"name", "label_smoothing"},
    }
    if set(value) != {"schema_version", *tables} or value.get("schema_version") != 1:
        raise ValueError("candidate config top-level schema mismatch")
    for name, keys in tables.items():
        if not isinstance(value.get(name), dict) or set(value[name]) != keys:
            raise ValueError(f"candidate config [{name}] schema mismatch")
    if value["text"]["config"] not in {"base", "large"} or not isinstance(value["text"]["head_zeroinit"], bool):
        raise ValueError("invalid text configuration")
    if value["loss"]["name"] != "bidirectional_contrastive":
        raise ValueError("invalid loss interface name")
    temperature = value["model"]["temperature_init"]
    smoothing = value["loss"]["label_smoothing"]
    if not isinstance(temperature, (int, float)) or not 0.01 <= float(temperature) <= 100.0:
        raise ValueError("invalid temperature_init")
    if not isinstance(smoothing, (int, float)) or not 0.0 <= float(smoothing) <= 0.2:
        raise ValueError("invalid label_smoothing")
    return value, json.dumps(value, sort_keys=True, separators=(",", ":"))


def validate_experiments(path: Path) -> int:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid experiments line {line_number}") from exc
        if not isinstance(row, dict) or not row.get("hypothesis") or not row.get("result"):
            raise ValueError(f"experiments line {line_number} lacks hypothesis/result")
        rows.append(row)
    if not rows:
        raise ValueError("experiments ledger is empty")
    return len(rows)


def validate(output: Path, baseline_path: Path, candidate_root: Path) -> dict:
    baseline = json.loads(regular(baseline_path).read_text())
    if baseline.get("status") != "complete":
        raise ValueError("baseline contract is not complete")
    paths = {
        "model": output / "artifacts" / "model.npz",
        "config": output / "artifacts" / "config.json",
        "provenance": output / "provenance.json",
        "experiments": output / "experiments.jsonl",
        "ledger": output / "run_ledger.json",
        "capacity": output / "capacity_record.json",
    }
    candidate_paths = [candidate_root / "candidate" / "text_tower.py", candidate_root / "candidate" / "alignment_loss.py"]
    candidate_config_path = candidate_root / "candidate" / "config.toml"
    for path in [*paths.values(), *candidate_paths, candidate_config_path]:
        regular(path)
    model = inspect_npz(paths["model"])
    if "image_parameter_digest" not in model:
        raise ValueError("model artifact requires an img/ parameter tree")
    with np.load(paths["model"], allow_pickle=False) as archive:
        if not any(key.startswith("txt/") for key in archive.files):
            raise ValueError("model artifact requires a txt/ parameter tree")
    config, canonical_config = canonical_candidate_config(candidate_config_path)
    if paths["config"].read_text() != canonical_config + "\n":
        raise ValueError("config artifact differs from the current canonical candidate config")
    ledger = json.loads(paths["ledger"].read_text())
    expected = {
        "status": "pass", "task_id": "bigvision_lit_coco_alignment", "world_size": 8,
        "global_batch_size": 4096, "optimizer_steps": 5000, "pair_exposures": 20_480_000,
        "optimizer": "scale_by_adam", "learning_rate": 0.001, "weight_decay": 0.01,
        "warmup_steps": 150, "seed": 0,
        "upstream_commit": "8921d5141504390a8a4f7b2dacb3b3c042237290",
    }
    mismatches = {key: {"expected": wanted, "actual": ledger.get(key)} for key, wanted in expected.items() if ledger.get(key) != wanted}
    if mismatches:
        raise ValueError(f"fixed training ledger mismatch: {mismatches}")
    if ledger.get("model_sha256") != model["model_sha256"]:
        raise ValueError("model hash differs from training ledger")
    if model["image_parameter_digest"] != baseline.get("image_parameter_digest"):
        raise ValueError("image parameter digest differs from sealed baseline")
    capacity = json.loads(paths["capacity"].read_text())
    if capacity.get("status") != "pass" or capacity.get("counter_version") != "bv-jax-lowered-fwd-bwd-v1":
        raise ValueError("invalid capacity record")
    if capacity.get("trainable_parameter_count") != model["trainable_parameter_count"]:
        raise ValueError("capacity/model parameter counts disagree")
    if int(capacity["trainable_parameter_count"]) > int(baseline["trainable_parameter_ceiling"]):
        raise ValueError("trainable parameter retention ceiling exceeded")
    flop_value = float(capacity.get("training_flops", math.nan))
    if not math.isfinite(flop_value) or flop_value > float(baseline["training_flop_ceiling"]):
        raise ValueError("training FLOP ceiling exceeded")
    if ledger.get("capacity_record_sha256") != sha256(paths["capacity"]):
        raise ValueError("capacity record hash differs from training ledger")
    provenance = json.loads(paths["provenance"].read_text())
    for key in ("model_sha256", "image_parameter_digest", "capacity_record_sha256"):
        if provenance.get(key) != ledger.get(key):
            raise ValueError(f"provenance/ledger mismatch: {key}")
    attempt_count = validate_experiments(paths["experiments"])
    expected_lineage = {
        "task_id": "bigvision_lit_coco_alignment",
        "candidate_source_sha256": source_digest(candidate_paths),
        "candidate_config_sha256": hashlib.sha256(canonical_config.encode()).hexdigest(),
        "model_sha256": model["model_sha256"],
        "input_manifest_sha256": baseline.get("asset_manifest_sha256"),
        "training_command": ledger.get("training_command"),
        "upstream_commit": "8921d5141504390a8a4f7b2dacb3b3c042237290",
        "optimization_attempts": attempt_count,
        "image_parameter_digest": model["image_parameter_digest"],
        "capacity_record_sha256": sha256(paths["capacity"]),
        "web_search": "disabled",
    }
    mismatches = {key: {"expected": value, "actual": provenance.get(key)} for key, value in expected_lineage.items() if provenance.get(key) != value}
    if mismatches:
        raise ValueError(f"provenance lineage mismatch: {mismatches}")
    return {"status": "pass", "model": model, "ledger": ledger, "capacity": capacity}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/app/output"))
    parser.add_argument("--baseline", type=Path, default=Path("/tests/baseline_contract.json"))
    parser.add_argument("--candidate-root", type=Path, default=Path("/app/project"))
    parser.add_argument("--record", type=Path, default=Path("/logs/verifier/artifact_check.json"))
    args = parser.parse_args()
    record = validate(args.output, args.baseline, args.candidate_root)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
