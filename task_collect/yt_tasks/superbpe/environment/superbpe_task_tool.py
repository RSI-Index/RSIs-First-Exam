#!/usr/bin/env python3
"""Frozen convenience and artifact-staging tools for the SuperBPE task."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


UPSTREAM = "bbd09768fc28a875cef48e6bdd66e3a17454628e"
DATASET_REVISION = "64b9a7c502482035602810c4e9acda1c1dd21908"
ORDER = ("48", "61", "66", "25", "6", "58", "51", "75", "80", "32")
TOTAL_BYTES = 10_000_000_000
PREFIX_BYTES = 557_641_266
TRAIN_SHA256 = {
    "48": "63f5c1d33b514f3d613c67e8e42b2b502d72535fc105a5f1ce268fa47bf714fa",
    "61": "8e90fa6309551c9082d5dbd39ebf0d2ab57daa025e3b44df4db4ba4d7ef9b212",
    "66": "905de8e397fb24fc04a72a7ed88f73966d21002e9e4ad7283e4997873f311c00",
    "25": "bb991ac0d84bfd5fb49091ee6b0e185d4d58a853a3da1ef8816c0d5cdd252810",
    "6": "ce85a425ec70f884d7260f3eae77c7b0f1043de4538479f8b451b4ff4cb6bc0e",
    "58": "3ae8e9e454057051b3548fabab770361b743b4f7eb2338c7bea2eb5c57ed7c92",
    "51": "d725aac2e89d1314e2a0234c98f74a9171ef200e234a96106c48dbbd8216b44b",
    "75": "66f65bbf463c1a8f2c7341db10c7a7601a8a916e1ef6663350ad11400d5061da",
    "80": "e6d4b9d39f1e20de71ecd728f137732cdcb14aea388a336fafc2dc1d4021f1e0",
    "32": "5c914367e4b26b0b89edcfbce01c5ac08a680474436a970e5bcb7e5cc30d4167",
}
PREFIX_SHA256 = "eba8cbab190c8afcb9a20fcbe08a4539707f291e47fbca1938d0e8caefe76c06"
DEFAULT_BASE = Path(
    "/app/project/tokenizer_json/olmo2_p99_truncate_pretok_10G_200K/merges.txt"
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def data_paths() -> tuple[list[Path], Path]:
    train = Path(os.environ.get("SUPERBPE_TRAIN_DIR", "/datasets/superbpe/train"))
    derived = Path(os.environ.get("SUPERBPE_DERIVED_DIR", str(train.parent / "derived")))
    full = [train / f"{name}.txt" for name in ORDER]
    prefix = derived / f"70_truncated_{PREFIX_BYTES}.txt"
    return full, prefix


def verify_data() -> tuple[list[Path], Path]:
    full, prefix = data_paths()
    missing = [str(path) for path in [*full, prefix] if not path.is_file()]
    if missing:
        raise SystemExit(f"missing staged training files: {missing}")
    count = sum(path.stat().st_size for path in full) + prefix.stat().st_size
    if count != TOTAL_BYTES:
        raise SystemExit(f"fixed training-byte total is {count}, expected {TOTAL_BYTES}")
    for name, path in zip(ORDER, full):
        actual = digest(path)
        if actual != TRAIN_SHA256[name]:
            raise SystemExit(f"SHA256 mismatch for {path}: {actual}")
    actual_prefix = digest(prefix)
    if actual_prefix != PREFIX_SHA256:
        raise SystemExit(f"SHA256 mismatch for {prefix}: {actual_prefix}")
    return full, prefix


def prepare_reference_shape(args: argparse.Namespace) -> None:
    full, prefix = verify_data()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not 0 <= args.transition_merges <= 199_757:
        raise SystemExit("transition-merges must be between 0 and 199757")
    lines = args.base_merges.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith("#version"):
        raise SystemExit(f"invalid base merges file: {args.base_merges}")
    selected = lines[: args.transition_merges + 1]
    (output / "merges.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")
    meta = {
        "total_bytes": TOTAL_BYTES,
        "train_files": [str(path) for path in full] + [str(prefix)],
    }
    (output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    recipe = {
        "status": "repository-derived starting shape; not an exact recovered author command",
        "upstream_commit": UPSTREAM,
        "dataset_revision": DATASET_REVISION,
        "transition_merge_entries": args.transition_merges,
        "base_merges": str(args.base_merges),
        "base_merges_sha256": digest(args.base_merges),
        "training_bytes": TOTAL_BYTES,
    }
    (output / "reference-shape.json").write_text(
        json.dumps(recipe, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output), **recipe}, indent=2))


def stage(args: argparse.Namespace) -> None:
    tokenizer = args.tokenizer.resolve()
    if not tokenizer.is_file():
        raise SystemExit(f"missing tokenizer: {tokenizer}")
    output_root = args.output_root.resolve()
    submission = output_root / "submission"
    submission.mkdir(parents=True, exist_ok=True)
    target = submission / "tokenizer.json"
    shutil.copy2(tokenizer, target)
    for name in ("merges.txt", "vocab.json"):
        sibling = tokenizer.parent / name
        if sibling.is_file():
            shutil.copy2(sibling, submission / name)
    selection = {
        "attempt_id": args.attempt_id,
        "selected_tokenizer": str(tokenizer),
        "staged_tokenizer": str(target),
        "tokenizer_sha256": digest(target),
        "bytes": target.stat().st_size,
    }
    (output_root / "submission-selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(selection, indent=2))


def audit(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    required = [
        root / "submission/tokenizer.json",
        root / "submission-selection.json",
        root / "provenance.json",
        root / "experiments.jsonl",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing required outputs: {missing}")
    selection = json.loads((root / "submission-selection.json").read_text(encoding="utf-8"))
    selected = 0
    selected_attempt = None
    rows = 0
    for lineno, line in enumerate((root / "experiments.jsonl").read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid experiments.jsonl line {lineno}: {exc}") from exc
        rows += 1
        if row.get("status") == "selected":
            selected += 1
            selected_attempt = row.get("attempt_id")
        if int(row.get("training_bytes", 0)) > TOTAL_BYTES:
            raise SystemExit(f"line {lineno} exceeds the 10GB training budget")
    if rows == 0 or selected != 1:
        raise SystemExit(f"ledger must contain rows and exactly one selected row; rows={rows}, selected={selected}")
    if selection.get("attempt_id") != selected_attempt:
        raise SystemExit("submission-selection attempt_id does not match the selected ledger row")
    report = {
        "status": "pass",
        "tokenizer_sha256": digest(root / "submission/tokenizer.json"),
        "ledger_rows": rows,
        "selected_rows": selected,
    }
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-reference-shape")
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--transition-merges", type=int, default=80_000)
    prepare.add_argument("--base-merges", type=Path, default=DEFAULT_BASE)
    prepare.set_defaults(func=prepare_reference_shape)
    stage_parser = sub.add_parser("stage")
    stage_parser.add_argument("--attempt-id", required=True)
    stage_parser.add_argument("--tokenizer", type=Path, required=True)
    stage_parser.add_argument("--output-root", type=Path, default=Path("/app/output"))
    stage_parser.set_defaults(func=stage)
    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--output-root", type=Path, default=Path("/app/output"))
    audit_parser.set_defaults(func=audit)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
