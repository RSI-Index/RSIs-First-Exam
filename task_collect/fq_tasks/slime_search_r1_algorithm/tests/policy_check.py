#!/usr/bin/env python3
"""Verify submission provenance and frozen-source integrity before GPU evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/task-tools")

from async_contract import read_events, read_json
from policy_contract import tree_sha256, validate_provenance
from task_contract import (
    FINAL_ROLLOUT_ID,
    NUM_ROLLOUT,
    build_training_arguments,
    canonical_command_sha256,
    load_algorithm_config,
)


OUTPUT = Path("/app/output")
ALGORITHM = Path("/app/project/rl_research")
ATTESTATIONS = Path("/var/lib/slime-task/attestations")
TRUSTED_RUNS = Path("/var/lib/slime-task/runs")


def audit() -> list[str]:
    errors = []
    try:
        provenance = read_json(OUTPUT / "provenance.json")
    except Exception as exc:
        return [f"cannot read provenance: {exc}"]
    errors.extend(validate_provenance(provenance, ALGORITHM))

    try:
        control = read_json(OUTPUT / "control-complete.json")
        selection = read_json(OUTPUT / "selected-attempt.json")
    except Exception as exc:
        errors.append(f"cannot read final lifecycle state: {exc}")
    else:
        selected = provenance.get("selected_attempt")
        if control.get("status") != "complete" or control.get("selected_attempt") != selected:
            errors.append("control completion does not match provenance")
        if selection.get("selected_attempt") != selected or selection.get("num_rollout") != NUM_ROLLOUT:
            errors.append("selection does not match the full-budget provenance")

    selected = provenance.get("selected_attempt")
    if isinstance(selected, str):
        attempt_root = OUTPUT / "attempts" / selected
        attempt_train = TRUSTED_RUNS / selected / "train"
        try:
            status = read_json(attempt_root / "status.json")
            run_contract = read_json(attempt_train / "run-contract.json")
            command_record = read_json(attempt_train / "training-command.json")
        except Exception as exc:
            errors.append(f"cannot read selected attempt receipt: {exc}")
        else:
            if status.get("status") != "completed" or status.get("attempt_id") != selected:
                errors.append("selected attempt status is not completed")
            if run_contract.get("num_rollout") != NUM_ROLLOUT or run_contract.get("world_size") != 8:
                errors.append("selected attempt receipt is not a full-budget 8-GPU run")
            if command_record.get("argv") != provenance.get("training_command"):
                errors.append("selected attempt command differs from provenance")
            if command_record.get("sha256") != provenance.get("training_command_sha256"):
                errors.append("selected attempt command hash differs from provenance")
        try:
            config = load_algorithm_config(ALGORITHM / "algorithm_config.toml")
            expected = ["python3", "/opt/slime/train.py", *build_training_arguments(config, attempt_train)]
        except Exception as exc:
            errors.append(f"cannot reconstruct frozen training command: {exc}")
        else:
            if provenance.get("training_command") != expected:
                errors.append("selected training command differs from the frozen launcher")
            if provenance.get("training_command_sha256") != canonical_command_sha256(expected):
                errors.append("selected training command hash differs from the frozen launcher")
        events = read_events(OUTPUT / "experiments.jsonl")
        terminal = [event.get("event") for event in events if event.get("attempt_id") == selected]
        if terminal.count("completed") != 1 or terminal.count("selected") != 1:
            errors.append("selected attempt lacks exactly one completed and selected ledger event")
        try:
            attestation = read_json(ATTESTATIONS / f"{selected}.json")
        except Exception as exc:
            errors.append(f"cannot read root-owned training attestation: {exc}")
        else:
            expected_attestation = {
                "algorithm_sha256": provenance.get("algorithm_sha256"),
                "attempt_id": selected,
                "command_sha256": provenance.get("training_command_sha256"),
                "final_rollout_id": FINAL_ROLLOUT_ID,
                "num_rollout": NUM_ROLLOUT,
                "status": "completed",
                "world_size": 8,
            }
            for field, value in expected_attestation.items():
                if attestation.get(field) != value:
                    errors.append(f"root-owned training attestation has wrong {field}")
            try:
                candidate_hash = tree_sha256(OUTPUT / "hf_model")
            except Exception as exc:
                errors.append(f"cannot hash selected HF checkpoint: {exc}")
            else:
                if attestation.get("hf_model_sha256") != candidate_hash:
                    errors.append("selected HF checkpoint differs from root-owned training attestation")

    try:
        manifest = json.loads(Path("/task-tools/frozen-manifest.json").read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"cannot read frozen-source manifest: {exc}")
    else:
        for path, expected_hash in manifest.items():
            try:
                actual = tree_sha256(path)
            except Exception as exc:
                errors.append(f"cannot hash frozen tree {path}: {exc}")
            else:
                if actual != expected_hash:
                    errors.append(f"frozen tree changed: {path}")
    return errors


def main() -> int:
    errors = audit()
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("policy audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
