#!/usr/bin/env python3
"""Agent-facing validation helpers for the DataComp filtering task.

The final verifier independently reshares and trains from the submitted UID
set.  This tool gives agents the same structural checks without exposing or
modifying the verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


UID_DTYPE = np.dtype("u8,u8")
DEFAULT_AVAILABLE = Path(
    "/datasets/datacomp/small/manifests/available_uids.npy"
)
REQUIRED_PROVENANCE = {
    "method_name",
    "method_description",
    "training_data",
    "selected_uids_sha256",
    "selected_uid_count",
    "training_command",
    "upstream_commits",
    "downloads",
    "evaluation_commands",
    "optimization_attempts",
    "web_search",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_uid_array(path: Path, *, mmap: bool = True) -> np.ndarray:
    if not path.is_file():
        raise RuntimeError(f"missing UID array: {path}")
    array = np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
    if array.ndim != 1:
        raise RuntimeError(f"UID array must be one-dimensional, found {array.shape}")
    if array.dtype != UID_DTYPE:
        raise RuntimeError(
            f"UID array must use the official structured dtype {UID_DTYPE}, "
            f"found {array.dtype}"
        )
    if len(array) == 0:
        raise RuntimeError("UID array must not be empty")
    return array


def validate_subset(path: Path, available_path: Path) -> dict[str, Any]:
    selected = load_uid_array(path)
    available = load_uid_array(available_path)
    if len(selected) > len(available):
        raise RuntimeError(
            f"selected set is larger than the sealed pool: {len(selected)} > {len(available)}"
        )
    if len(selected) > 1 and np.any(selected[1:] <= selected[:-1]):
        raise RuntimeError("selected UIDs must be strictly sorted and unique")

    positions = np.searchsorted(available, selected)
    in_bounds = positions < len(available)
    matches = np.zeros(len(selected), dtype=bool)
    matches[in_bounds] = available[positions[in_bounds]] == selected[in_bounds]
    if not bool(matches.all()):
        first = int(np.flatnonzero(~matches)[0])
        raise RuntimeError(
            f"submitted UID at sorted position {first} is not in the sealed successful pool"
        )

    result = {
        "status": "ok",
        "selected_uid_count": int(len(selected)),
        "available_uid_count": int(len(available)),
        "selected_fraction": float(len(selected) / len(available)),
        "selected_uids_sha256": sha256(path),
        "available_uids_sha256": sha256(available_path),
    }
    return result


def audit_output(output_root: Path, available_path: Path) -> dict[str, Any]:
    subset_path = output_root / "submission" / "selected_uids.npy"
    result = validate_subset(subset_path, available_path)

    provenance_path = output_root / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_PROVENANCE - provenance.keys())
    if missing:
        raise RuntimeError(f"provenance is missing fields: {missing}")
    if provenance["selected_uids_sha256"] != result["selected_uids_sha256"]:
        raise RuntimeError("provenance selected_uids_sha256 does not match submission")
    if int(provenance["selected_uid_count"]) != result["selected_uid_count"]:
        raise RuntimeError("provenance selected_uid_count does not match submission")
    if provenance["web_search"] != "disabled":
        raise RuntimeError("provenance web_search must be 'disabled'")

    ledger_path = output_root / "experiments.jsonl"
    rows = []
    for line_number, line in enumerate(
        ledger_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"ledger row {line_number} is not an object")
        rows.append(value)
    if not rows:
        raise RuntimeError("experiments.jsonl must contain at least one decision cycle")
    selected_rows = [row for row in rows if row.get("status") == "selected"]
    if len(selected_rows) != 1:
        raise RuntimeError(
            f"experiments.jsonl must contain exactly one selected row, found {len(selected_rows)}"
        )
    if int(provenance["optimization_attempts"]) < 0:
        raise RuntimeError("optimization_attempts must be non-negative")
    result.update(
        {
            "provenance": str(provenance_path),
            "experiment_rows": len(rows),
            "selected_attempt": selected_rows[0].get("attempt_id"),
        }
    )
    return result


def run_resharder(
    project: Path,
    input_dir: Path,
    output_dir: Path,
    subset_path: Path,
    workers: int,
) -> None:
    validation = validate_subset(subset_path, DEFAULT_AVAILABLE)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(project / "resharder.py"),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--subset-file",
        str(subset_path),
        "--num-workers",
        str(workers),
    ]
    print(json.dumps({"validation": validation, "command": command}, indent=2))
    subprocess.run(command, cwd=project, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-subset")
    validate.add_argument("--path", type=Path, required=True)
    validate.add_argument("--available", type=Path, default=DEFAULT_AVAILABLE)

    audit = subparsers.add_parser("audit-output")
    audit.add_argument("--output-root", type=Path, default=Path("/app/output"))
    audit.add_argument("--available", type=Path, default=DEFAULT_AVAILABLE)

    reshard = subparsers.add_parser("reshard")
    reshard.add_argument("--project", type=Path, default=Path("/app/project"))
    reshard.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/datasets/datacomp/small/commonpool/shards"),
    )
    reshard.add_argument("--output-dir", type=Path, required=True)
    reshard.add_argument("--subset-file", type=Path, required=True)
    reshard.add_argument("--workers", type=int, default=32)

    args = parser.parse_args()
    if args.command == "validate-subset":
        result = validate_subset(args.path, args.available)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "audit-output":
        result = audit_output(args.output_root, args.available)
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        run_resharder(
            args.project,
            args.input_dir,
            args.output_dir,
            args.subset_file,
            args.workers,
        )


if __name__ == "__main__":
    main()
