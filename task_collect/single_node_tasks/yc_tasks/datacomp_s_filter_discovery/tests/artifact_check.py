#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_checkpoint(checkpoint: Path, sanitized: Path) -> dict:
    import torch

    try:
        value = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("checkpoint failed safe weights-only loading") from exc
    if not isinstance(value, dict) or not isinstance(value.get("state_dict"), dict):
        raise ValueError("checkpoint must contain one state_dict mapping")
    state = value["state_dict"]
    if not state:
        raise ValueError("checkpoint state_dict is empty")
    for key, tensor in state.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("state_dict must contain only string-to-tensor entries")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"non-finite checkpoint tensor: {key}")
    sanitized.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state}, sanitized)
    return {"tensor_count": len(state), "sanitized_checkpoint_sha256": sha256(sanitized)}


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


def inspect_subset(path: Path) -> dict:
    value = np.load(path, allow_pickle=False)
    expected = np.dtype("u8,u8")
    if value.ndim != 1 or value.dtype != expected or value.dtype.hasobject or len(value) == 0:
        raise ValueError("invalid subset array schema")
    if len(value) > 1:
        left0, left1 = value["f0"][:-1], value["f1"][:-1]
        right0, right1 = value["f0"][1:], value["f1"][1:]
        if not bool(np.all((left0 < right0) | ((left0 == right0) & (left1 < right1)))):
            raise ValueError("subset is not unique and lexicographically sorted")
    return {"count": len(value), "subset_sha256": sha256(path)}


def validate(output: Path, sanitized: Path, baseline_path: Path, candidate_root: Path) -> dict:
    ledger_path = output / "run_ledger.json"
    checkpoint = output / "checkpoints" / "epoch_5.pt"
    selection = output / "selection_record.json"
    attestation_path = output / "initialization_attestation.json"
    subset_path = output / "artifacts" / "subset.npy"
    provenance_path = output / "provenance.json"
    experiments_path = output / "experiments.jsonl"
    selector_path = candidate_root / "candidate" / "selector.py"
    for path in (ledger_path, checkpoint, selection, attestation_path, subset_path, provenance_path, experiments_path, selector_path, baseline_path):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing regular artifact: {path}")
    ledger = json.loads(ledger_path.read_text())
    selected = json.loads(selection.read_text())
    expected = {
        "status": "pass", "world_size": 4, "configured_training_pairs": 12800000,
        "realized_training_pairs": 12861440, "optimizer_updates": 3140,
        "global_batch_size": 4096, "seed": 0,
    }
    mismatches = {key: {"expected": value, "actual": ledger.get(key)} for key, value in expected.items() if ledger.get(key) != value}
    if mismatches:
        raise ValueError(f"fixed training ledger mismatch: {mismatches}")
    if selected.get("status") != "pass" or selected.get("repeat_byte_identical") is not True:
        raise ValueError("selector validation is incomplete")
    actual = sha256(checkpoint)
    if ledger.get("checkpoint_sha256") != actual:
        raise ValueError("checkpoint hash differs from the training ledger")
    if ledger.get("initialization_attestation_sha256") != sha256(attestation_path):
        raise ValueError("initialization attestation hash differs from the training ledger")
    attestation = json.loads(attestation_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    expected_attestation = {
        "model": "ViT-B-32", "pretrained": "", "pretrained_image": False,
        "distill_model": None, "distill_pretrained": None, "rank": 0, "resume": None,
        "seed": 0, "stage": "post-model-construction-pre-optimizer",
    }
    mismatches = {key: {"expected": value, "actual": attestation.get(key)} for key, value in expected_attestation.items() if attestation.get(key) != value}
    if mismatches:
        raise ValueError(f"fresh initialization attestation mismatch: {mismatches}")
    fingerprint = attestation.get("fingerprint", {})
    if fingerprint.get("algorithm") != "sha256-state-dict-v1" or fingerprint.get("parameter_count") != 151277313:
        raise ValueError("unexpected ViT-B/32 initialization fingerprint metadata")
    if fingerprint.get("sha256") != baseline.get("initialization_parameter_digest"):
        raise ValueError("initialization parameter digest differs from sealed baseline")
    subset = inspect_subset(subset_path)
    provenance = json.loads(provenance_path.read_text())
    attempt_count = validate_experiments(experiments_path)
    expected_lineage = {
        "task_id": "datacomp_s_filter_discovery",
        "candidate_source_sha256": sha256(selector_path),
        "subset_sha256": subset["subset_sha256"],
        "input_manifest_sha256": baseline.get("asset_manifest_sha256"),
        "selection_command": selected.get("selection_command"),
        "training_command": ledger.get("training_command"),
        "upstream_commits": {
            "datacomp": "4a8df1992566ef8334773f7152e1855b1f716162",
            "open_clip": "395851ee9c55fdddb63f5579955cdd6d754d261c",
        },
        "optimization_attempts": attempt_count,
        "web_search": "disabled",
    }
    mismatches = {key: {"expected": value, "actual": provenance.get(key)} for key, value in expected_lineage.items() if provenance.get(key) != value}
    if mismatches:
        raise ValueError(f"provenance lineage mismatch: {mismatches}")
    if selected.get("candidate_source_sha256") != sha256(selector_path) or selected.get("subset_sha256") != subset["subset_sha256"]:
        raise ValueError("selection record is not bound to current source/subset")
    safe = sanitize_checkpoint(checkpoint, sanitized)
    return {"status": "pass", "checkpoint": str(checkpoint), "checkpoint_sha256": actual, "ledger": ledger, "selection": selected, "subset": subset, "initialization": attestation, "sanitized": safe}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/app/output"))
    parser.add_argument("--record", type=Path, default=Path("/logs/verifier/artifact_check.json"))
    parser.add_argument("--sanitized", type=Path, default=Path("/logs/verifier/sanitized/epoch_5.pt"))
    parser.add_argument("--baseline", type=Path, default=Path("/tests/baseline_contract.json"))
    parser.add_argument("--candidate-root", type=Path, default=Path("/app/project"))
    args = parser.parse_args()
    record = validate(args.output, args.sanitized, args.baseline, args.candidate_root)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
