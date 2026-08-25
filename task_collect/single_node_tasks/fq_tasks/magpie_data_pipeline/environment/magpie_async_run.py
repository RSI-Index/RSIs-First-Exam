#!/usr/bin/env python3
"""Submit and select immutable Magpie dataset attempts in one task session."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:  # Supports both package imports in tests and flat /task-tools execution.
    from .async_contract import read_status, transition_status, write_status
    from .generate_dataset import EDITABLE_ROOT, _load_editable_pipeline, run_attempt
except ImportError:  # pragma: no cover - exercised inside the task image
    from async_contract import read_status, transition_status, write_status
    from generate_dataset import EDITABLE_ROOT, _load_editable_pipeline, run_attempt


OUTPUT_ROOT = Path("/app/output")
ATTEMPT_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def attempt_paths(attempt_id: str) -> tuple[Path, Path, Path]:
    if not isinstance(attempt_id, str) or not ATTEMPT_PATTERN.fullmatch(attempt_id):
        raise ValueError("attempt id must match ^[a-z0-9][a-z0-9-]{0,63}$")
    root = OUTPUT_ROOT / "attempts" / attempt_id
    return root, root / "status.json", root / "worker.log"


def _active_attempt() -> str | None:
    attempts = OUTPUT_ROOT / "attempts"
    if not attempts.is_dir():
        return None
    for path in attempts.glob("*/status.json"):
        if path.is_file():
            payload = read_status(path)
            if payload["status"] in {"queued", "running"}:
                return str(payload.get("attempt_id", path.parent.name))
    return None


def submit(args: argparse.Namespace) -> None:
    root, status_path, _ = attempt_paths(args.attempt_id)
    if root.exists():
        raise RuntimeError(f"attempt id is immutable and already exists: {args.attempt_id}")
    active = _active_attempt()
    if active:
        raise RuntimeError(f"another attempt is active: {active}")
    root.mkdir(parents=True)
    command = [sys.executable, str(Path(__file__).resolve()), "worker", "--attempt-id", args.attempt_id, "--hypothesis", args.hypothesis]
    write_status(status_path, {"attempt_id": args.attempt_id, "hypothesis": args.hypothesis, "status": "queued"})
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    payload = {"attempt_id": args.attempt_id, "hypothesis": args.hypothesis, "status": "queued"}
    print(json.dumps({**payload, "status_path": str(status_path), "worker_pid": process.pid}, indent=2, sort_keys=True), flush=True)


def worker(args: argparse.Namespace) -> int:
    root, status_path, log_path = attempt_paths(args.attempt_id)
    for _ in range(100):
        if status_path.is_file():
            break
        time.sleep(0.01)
    if not status_path.is_file():
        raise FileNotFoundError(f"no queued status for attempt {args.attempt_id}")
    transition_status(status_path, "running", pid=os.getpid())
    try:
        run_attempt(args.attempt_id, args.hypothesis, root)
        completed = read_status(status_path)
        if completed["status"] == "running":
            transition_status(status_path, "completed")
        return 0
    except Exception as exc:
        if read_status(status_path)["status"] == "running":
            transition_status(status_path, "failed", error=str(exc))
        try:
            _, config, _, _ = _load_editable_pipeline(EDITABLE_ROOT)
        except Exception:
            config = {}
        _append_experiment_event(
            {"attempt_id": args.attempt_id, "hypothesis": args.hypothesis, "config": config, "outcome": "failed", "selected": False, "error": str(exc)}
        )
        log_path.write_text(str(exc) + "\n", encoding="utf-8")
        return 1


def status(args: argparse.Namespace) -> None:
    _, status_path, _ = attempt_paths(args.attempt_id)
    if not status_path.is_file():
        raise FileNotFoundError(f"no status for attempt {args.attempt_id}")
    print(status_path.read_text(encoding="utf-8"), end="")


def _regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink() and not any(parent.is_symlink() for parent in path.parents)


def _append_experiment_event(event: dict[str, object]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "experiments.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def select(args: argparse.Namespace) -> None:
    root, status_path, _ = attempt_paths(args.attempt_id)
    if not status_path.is_file() or read_status(status_path)["status"] != "completed":
        raise RuntimeError(f"attempt {args.attempt_id} is not completed")
    source = root / "dataset.jsonl"
    provenance_path = root / "provenance.json"
    if not _regular_file(source) or not _regular_file(provenance_path):
        raise RuntimeError(f"attempt {args.attempt_id} has no complete dataset/provenance")
    manifest_path = root / "dataset_manifest.json"
    if not _regular_file(manifest_path):
        raise RuntimeError(f"attempt {args.attempt_id} has no complete dataset manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("dataset_sha256") != hashlib.sha256(source.read_bytes()).hexdigest():
        raise RuntimeError(f"attempt {args.attempt_id} has invalid dataset manifest")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ("dataset.jsonl", "pipeline_config.toml", "provenance.json", "dataset_manifest.json"):
        origin = root / name
        if not _regular_file(origin):
            raise RuntimeError(f"attempt {args.attempt_id} has unsafe or missing {name}")
        temporary = OUTPUT_ROOT / f".{name}.selected.tmp"
        shutil.copyfile(origin, temporary, follow_symlinks=False)
        os.replace(temporary, OUTPUT_ROOT / name)
    dataset_hash = hashlib.sha256((OUTPUT_ROOT / "dataset.jsonl").read_bytes()).hexdigest()
    selection = {"selected_attempt": args.attempt_id, "status": "selected", "dataset_sha256": dataset_hash}
    write_status(OUTPUT_ROOT / "submission-selection.json", selection)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    _append_experiment_event(
        {
            "attempt_id": args.attempt_id,
            "hypothesis": provenance.get("hypothesis", ""),
            "config": provenance.get("pipeline_config", {}),
            "outcome": "selected",
            "selected": True,
            "dataset_sha256": dataset_hash,
        }
    )
    print(json.dumps(selection, indent=2, sort_keys=True))


def complete_control(_args: argparse.Namespace) -> None:
    selection_path = OUTPUT_ROOT / "submission-selection.json"
    dataset_path = OUTPUT_ROOT / "dataset.jsonl"
    if not _regular_file(selection_path) or not _regular_file(dataset_path):
        raise RuntimeError("select a completed attempt before completing control")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    attempt_id = selection.get("selected_attempt")
    if not isinstance(attempt_id, str):
        raise RuntimeError("select a completed attempt before completing control")
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    write_status(OUTPUT_ROOT / "control-complete.json", {"status": "complete", "selected_attempt": attempt_id, "dataset_sha256": dataset_hash})
    print(OUTPUT_ROOT / "control-complete.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    submit_parser = subparsers.add_parser("submit")
    submit_parser.add_argument("--attempt-id", required=True)
    submit_parser.add_argument("--hypothesis", required=True)
    submit_parser.set_defaults(function=submit)
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--attempt-id", required=True)
    worker_parser.add_argument("--hypothesis", required=True)
    worker_parser.set_defaults(function=worker)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--attempt-id", required=True)
    status_parser.set_defaults(function=status)
    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--attempt-id", required=True)
    select_parser.set_defaults(function=select)
    complete_parser = subparsers.add_parser("complete-control")
    complete_parser.set_defaults(function=complete_control)
    args = parser.parse_args()
    return int(args.function(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
