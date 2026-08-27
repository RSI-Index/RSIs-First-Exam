#!/usr/bin/env python3
"""Prepare, gate, summarize, and stage positional scaling-ladder attempts."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCALE_ORDER = ("E0", "E1", "E2", "E3", "E4", "E5")
SCALE_CONTRACTS: dict[str, dict[str, int]] = {
    "E0": {"layers": 12, "hidden": 1152, "ffn": 4608, "heads": 9, "parameters": 550_337_664, "tokens": 2_904_358_912, "iterations": 44_317, "gbs": 16, "mbs": 2, "gpus": 8},
    "E1": {"layers": 15, "hidden": 1408, "ffn": 5632, "heads": 11, "parameters": 837_007_744, "tokens": 3_612_672_000, "iterations": 55_125, "gbs": 16, "mbs": 2, "gpus": 8},
    "E2": {"layers": 16, "hidden": 1536, "ffn": 6144, "heads": 12, "parameters": 998_036_992, "tokens": 4_982_571_008, "iterations": 38_014, "gbs": 32, "mbs": 4, "gpus": 8},
    "E3": {"layers": 18, "hidden": 1792, "ffn": 7168, "heads": 14, "parameters": 1_384_584_448, "tokens": 10_559_946_752, "iterations": 40_283, "gbs": 64, "mbs": 2, "gpus": 32},
    "E4": {"layers": 21, "hidden": 2048, "ffn": 8192, "heads": 16, "parameters": 1_934_716_160, "tokens": 14_805_106_688, "iterations": 56_477, "gbs": 64, "mbs": 2, "gpus": 32},
    "E5": {"layers": 23, "hidden": 2304, "ffn": 9216, "heads": 18, "parameters": 2_544_614_912, "tokens": 18_617_466_880, "iterations": 35_510, "gbs": 128, "mbs": 1, "gpus": 128},
}
PALOMA_TOLERANCE = 1.01
PARAMETER_TOLERANCE = 1.02
PARAMETER_PATTERN = re.compile(
    r"number of parameters on \(tensor, pipeline\) model parallel rank \(0, 0\):\s*([0-9]+)"
)
IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def is_runtime_generated(relative: str) -> bool:
    """Return whether a source-tree file is a deterministic runtime build product."""
    path = Path(relative)
    if any(
        part in {".mypy_cache", ".pytest_cache", ".ruff_cache", "wandb"}
        or part.endswith(".egg-info")
        for part in path.parts
    ):
        return True
    return bool(
        path.parent.as_posix() == "3rdparty/Megatron-LM/megatron/core/datasets"
        and path.name.startswith("helpers_cpp.")
        and path.suffix == ".so"
    )


def safe_relative_path(value: object, label: str) -> Path:
    """Validate an artifact-relative path without resolving through symlinks."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or ".git" in path.parts:
        raise ValueError(f"unsafe {label}: {value}")
    return path


def safe_identifier(value: object, label: str) -> str:
    """Validate a single path-safe task identifier."""
    if not isinstance(value, str) or IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise ValueError(f"unsafe {label}: {value}")
    return value


def require_resolved_below(root: Path, path: Path, label: str) -> None:
    """Require a resolved path to remain inside a trusted root."""
    resolved_root = root.resolve()
    resolved_path = path.resolve(strict=False)
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise SystemExit(f"{label} escapes {resolved_root}: {path}")


def remove_path(path: Path) -> None:
    """Remove one already-validated file, symlink, or directory."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def snapshot_source(
    project: Path,
    output_root: Path,
    attempt_id: str,
    changed_paths: list[str],
    source_hash: str,
) -> Path:
    """Persist the exact changed-file overlay for one staged incumbent."""
    attempt_id = safe_identifier(attempt_id, "attempt_id")
    snapshots_root = output_root / "source-snapshots"
    target = snapshots_root / attempt_id
    temporary = snapshots_root / f".{attempt_id}.tmp"
    if target.is_dir() and not target.is_symlink():
        existing = json.loads((target / "source_manifest.json").read_text())
        if (
            existing.get("attempt_id") == attempt_id
            and existing.get("source_inventory_sha256") == source_hash
        ):
            return target / "source_manifest.json"
        raise SystemExit(f"different source snapshot already exists for {attempt_id}")
    if target.exists() or target.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise SystemExit(f"unsafe source snapshot collision for {attempt_id}")
    (temporary / "files").mkdir(parents=True)
    entries: list[dict[str, str]] = []
    try:
        for changed in changed_paths:
            relative = safe_relative_path(changed, "changed source path")
            source = project / relative
            entry: dict[str, str] = {"path": relative.as_posix()}
            if source.is_symlink():
                entry.update({"kind": "symlink", "target": os.readlink(source)})
            elif source.is_file():
                destination = temporary / "files" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                entry["kind"] = "file"
            elif not source.exists():
                entry["kind"] = "deleted"
            else:
                raise SystemExit(f"unsupported changed source type: {source}")
            entries.append(entry)
        atomic_json(
            temporary / "source_manifest.json",
            {
                "version": 1,
                "attempt_id": attempt_id,
                "source_inventory_sha256": source_hash,
                "entries": entries,
            },
        )
        snapshots_root.mkdir(parents=True, exist_ok=True)
        temporary.replace(target)
    except BaseException:
        if temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        raise
    return target / "source_manifest.json"


def apply_source_snapshot(project: Path, manifest_path: Path) -> str:
    """Apply one validated incumbent overlay to a clean project copy."""
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != 1:
        raise SystemExit("unsupported incumbent source snapshot version")
    files_root = manifest_path.parent / "files"
    if not files_root.is_dir() or files_root.is_symlink():
        raise SystemExit(f"incumbent source snapshot files are missing or unsafe: {files_root}")
    for raw_entry in manifest.get("entries", []):
        if not isinstance(raw_entry, dict):
            raise SystemExit("invalid incumbent source snapshot entry")
        relative = safe_relative_path(raw_entry.get("path"), "snapshot source path")
        destination = project / relative
        require_resolved_below(project, destination.parent, "snapshot destination parent")
        destination.parent.mkdir(parents=True, exist_ok=True)
        remove_path(destination)
        kind = raw_entry.get("kind")
        if kind == "deleted":
            continue
        if kind == "file":
            source = files_root / relative
            if not source.is_file() or source.is_symlink():
                raise SystemExit(f"snapshot file is missing or unsafe: {source}")
            require_resolved_below(files_root, source, "snapshot file")
            shutil.copy2(source, destination)
        elif kind == "symlink":
            target = raw_entry.get("target")
            if not isinstance(target, str):
                raise SystemExit(f"invalid snapshot symlink target for {relative}")
            destination.symlink_to(target)
        else:
            raise SystemExit(f"invalid snapshot entry kind for {relative}: {kind}")
    return str(manifest["source_inventory_sha256"])


def project_inventory(project: Path) -> tuple[str, list[str]]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(project),
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--no-renames",
        ],
        check=True,
        capture_output=True,
    )
    records = [item for item in result.stdout.split(b"\0") if item]
    paths: list[str] = []
    for record in records:
        text = record.decode("utf-8", errors="strict")
        path_text = text[3:]
        if " -> " in path_text:
            path_text = path_text.split(" -> ", 1)[1]
        if not is_runtime_generated(path_text):
            paths.append(path_text)
    # Hash the complete source tree, not only Git's changed-file list.  Task
    # images flatten the pinned Bridge repository and MCore submodule into one
    # baseline commit, but this also remains correct in a development checkout
    # where MCore is still represented as a submodule.
    digest = hashlib.sha256()
    source_files: list[Path] = []
    for candidate in project.rglob("*"):
        relative_parts = candidate.relative_to(project).parts
        if ".git" in relative_parts or "__pycache__" in relative_parts:
            continue
        relative = candidate.relative_to(project).as_posix()
        if is_runtime_generated(relative):
            continue
        if candidate.is_file() or candidate.is_symlink():
            if candidate.suffix == ".pyc":
                continue
            source_files.append(candidate)
    for candidate in sorted(source_files, key=lambda item: item.relative_to(project).as_posix()):
        relative = candidate.relative_to(project).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        if candidate.is_symlink():
            digest.update(b"symlink\0" + os.readlink(candidate).encode())
        else:
            digest.update(b"file\0" + hashlib.sha256(candidate.read_bytes()).digest())
    # Keep the concise Git changed-path list for human provenance.
    return digest.hexdigest(), sorted(set(paths))


def scale_contract(scale: str) -> dict[str, Any]:
    if scale not in SCALE_CONTRACTS:
        raise ValueError(f"unknown scale: {scale}")
    values = SCALE_CONTRACTS[scale]
    return {
        "scale": scale,
        "model": {
            "layers": values["layers"],
            "hidden_size": values["hidden"],
            "ffn_hidden_size": values["ffn"],
            "attention_heads": values["heads"],
            "query_groups": values["heads"],
            "baseline_parameters": values["parameters"],
        },
        "training": {
            "iterations": values["iterations"],
            "tokens": values["tokens"],
            "sequence_length": 4096,
            "global_batch_size": values["gbs"],
            "micro_batch_size": values["mbs"],
            "seed": 0,
            "gpus": values["gpus"],
        },
        "evaluation": {"paloma": True, "longppl": True},
    }


def attempt_contract(attempt: Path) -> tuple[str, dict[str, Any]]:
    payload = json.loads((attempt / "run_contract.json").read_text())
    scale = payload.get("scale")
    if scale not in SCALE_CONTRACTS:
        raise ValueError(f"attempt has invalid scale: {scale}")
    return scale, payload


def load_baselines(path: Path | None = None) -> dict[str, dict[str, float]]:
    baseline_path = path or Path(
        os.environ.get(
            "POSITIONAL_SCALING_BASELINES",
            "/task-data/positional_adamh_baselines.json",
        )
    )
    payload = json.loads(baseline_path.read_text())
    if payload.get("version") != 1 or set(payload.get("scales", {})) != set(SCALE_ORDER):
        raise ValueError(f"invalid six-scale baseline manifest: {baseline_path}")
    result: dict[str, dict[str, float]] = {}
    for scale in SCALE_ORDER:
        raw = payload["scales"][scale]
        result[scale] = {
            "paloma_bits_per_byte": float(raw["paloma_bits_per_byte"]),
            "paloma_macro_bits_per_byte": float(raw["paloma_macro_bits_per_byte"]),
            "longppl": float(raw["longppl"]),
        }
        if any(value <= 0 for value in result[scale].values()):
            raise ValueError(f"non-positive baseline metric for {scale}")
    return result


def read_metrics(attempt: Path, final_iteration: int) -> dict[str, Any]:
    harness_path = attempt / "eval_harness" / f"step_{final_iteration:08d}" / "results.json"
    longppl_path = attempt / "longppl" / f"step_{final_iteration:08d}" / "results.json"
    harness = json.loads(harness_path.read_text())
    longppl = json.loads(longppl_path.read_text())
    paloma = harness["paloma_aggregate"]
    long_results = longppl["results"]
    return {
        "paloma_bits_per_byte": float(paloma["bits_per_byte"]),
        "paloma_macro_bits_per_byte": float(paloma["macro_bits_per_byte"]),
        "longppl": float(long_results["longppl"]),
        "longppl_loss": float(long_results["longppl_loss"]),
        "ordinary_ppl": float(long_results["ppl"]),
        "longppl_samples": int(long_results["samples"]),
        "longppl_key_tokens": int(long_results["key_tokens"]),
    }


def read_metric_trajectory(attempt: Path) -> list[dict[str, Any]]:
    """Read every complete intermediate Paloma and LongPPL result."""
    by_iteration: dict[int, dict[str, Any]] = {}
    for result_path in sorted((attempt / "eval_harness").glob("step_*/results.json")):
        try:
            iteration = int(result_path.parent.name.removeprefix("step_"))
            paloma = json.loads(result_path.read_text())["paloma_aggregate"]
            by_iteration.setdefault(iteration, {"iteration": iteration}).update(
                {
                    "paloma_bits_per_byte": float(paloma["bits_per_byte"]),
                    "paloma_macro_bits_per_byte": float(paloma["macro_bits_per_byte"]),
                }
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    for result_path in sorted((attempt / "longppl").glob("step_*/results.json")):
        try:
            iteration = int(result_path.parent.name.removeprefix("step_"))
            long_results = json.loads(result_path.read_text())["results"]
            by_iteration.setdefault(iteration, {"iteration": iteration}).update(
                {
                    "longppl": float(long_results["longppl"]),
                    "longppl_loss": float(long_results["longppl_loss"]),
                    "ordinary_ppl": float(long_results["ppl"]),
                }
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return [by_iteration[iteration] for iteration in sorted(by_iteration)]


def read_parameter_count(attempt: Path) -> int | None:
    """Read the exact instantiated model parameter count from launcher or node logs."""
    log_paths = [attempt / "run.log", *sorted((attempt / "logs").glob("*.log"))]
    matches: list[int] = []
    for log_path in log_paths:
        if not log_path.is_file():
            continue
        matches.extend(
            int(value)
            for value in PARAMETER_PATTERN.findall(log_path.read_text(errors="replace"))
        )
    return max(matches) if matches else None


def metric_guards(
    scale: str,
    metrics: dict[str, Any],
    parameters: int | None,
    baselines: dict[str, dict[str, float]] | None = None,
) -> dict[str, bool]:
    """Evaluate the scale-relative scientific gates required for submission."""
    reference = (baselines or load_baselines())[scale]
    maximum_parameters = int(SCALE_CONTRACTS[scale]["parameters"] * PARAMETER_TOLERANCE)
    return {
        "paloma_micro": float(metrics["paloma_bits_per_byte"])
        <= reference["paloma_bits_per_byte"] * PALOMA_TOLERANCE,
        "paloma_macro": float(metrics["paloma_macro_bits_per_byte"])
        <= reference["paloma_macro_bits_per_byte"] * PALOMA_TOLERANCE,
        "longppl_improvement": float(metrics["longppl"]) < reference["longppl"],
        "parameter_budget": parameters is not None and parameters <= maximum_parameters,
    }


def passing_attempts(output_root: Path, source_hash: str) -> dict[str, Path]:
    """Return the newest passing attempt per scale for one exact source hash."""
    result: dict[str, Path] = {}
    attempts_root = output_root / "attempts"
    if not attempts_root.is_dir():
        return result
    for status_path in sorted(attempts_root.glob("*/status.json")):
        try:
            status = json.loads(status_path.read_text())
            scale, run_contract = attempt_contract(status_path.parent)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if (
            status.get("status") == "completed"
            and status.get("submission_eligible") is True
            and run_contract.get("source_inventory_sha256") == source_hash
        ):
            result[scale] = status_path.parent
    return result


def prepare(args: argparse.Namespace) -> None:
    attempt = args.attempt.resolve()
    output_root = Path(os.environ.get("POSITIONAL_OUTPUT_ROOT", "/app/output")).resolve()
    if output_root not in attempt.parents:
        raise SystemExit(f"attempt must be below {output_root}: {attempt}")
    if attempt.exists():
        raise SystemExit(f"attempt already exists: {attempt}")
    scale = args.scale
    if scale not in SCALE_CONTRACTS:
        raise SystemExit(f"unknown scale: {scale}")
    load_baselines(args.baselines)
    source_hash, changed_paths = project_inventory(args.project.resolve())
    attempt.mkdir(parents=True)
    hypothesis_target = attempt / "HYPOTHESIS.md"
    hypothesis_target.write_bytes(args.hypothesis_file.read_bytes())
    source_manifest = snapshot_source(
        args.project.resolve(),
        output_root,
        args.attempt_id,
        changed_paths,
        source_hash,
    )
    atomic_json(
        attempt / "run_contract.json",
        {
            "version": 1,
            "attempt_id": args.attempt_id,
            "scale": scale,
            "created_at": utc_now(),
            "source_inventory_sha256": source_hash,
            "source_snapshot": str(source_manifest.relative_to(output_root)),
            "changed_paths": changed_paths,
            "contract": scale_contract(scale),
        },
    )
    atomic_json(
        attempt / "status.json",
        {"status": "running", "scale": scale, "updated_at": utc_now()},
    )


def finish(args: argparse.Namespace) -> None:
    attempt = args.attempt.resolve()
    scale, run_contract = attempt_contract(attempt)
    scale_values = SCALE_CONTRACTS[scale]
    status: dict[str, Any] = {
        "status": "failed",
        "scale": scale,
        "exit_code": args.exit_code,
        "updated_at": utc_now(),
    }
    source_hash, _ = project_inventory(args.project.resolve())
    status["source_inventory_sha256"] = source_hash
    status["source_matches_launch"] = source_hash == run_contract["source_inventory_sha256"]
    tracker = attempt / "checkpoints" / "latest_checkpointed_iteration.txt"
    iteration = int(tracker.read_text().strip()) if tracker.is_file() else None
    status["checkpoint_iteration"] = iteration
    status["trajectory"] = read_metric_trajectory(attempt)
    final_iteration = scale_values["iterations"]
    if args.exit_code == 0 and iteration == final_iteration and status["source_matches_launch"]:
        try:
            status["metrics"] = read_metrics(attempt, final_iteration)
            status["parameters"] = read_parameter_count(attempt)
            status["guards"] = metric_guards(
                scale,
                status["metrics"],
                status["parameters"],
                load_baselines(args.baselines),
            )
            status["submission_eligible"] = all(status["guards"].values())
            status["status"] = "completed"
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            status["error"] = f"incomplete final evaluation: {error}"
    early_stop_path = attempt / "early_stop.json"
    if status["status"] != "completed" and early_stop_path.is_file():
        early_stop = json.loads(early_stop_path.read_text())
        status["status"] = "early_stopped"
        status["early_stop"] = early_stop
    atomic_json(attempt / "status.json", status)
    ledger = attempt.parents[1] / "experiments.jsonl"
    with ledger.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(
            json.dumps(
                {
                    "attempt_id": run_contract["attempt_id"],
                    "scale": scale,
                    "status": status["status"],
                    "training_tokens": scale_values["tokens"],
                    "optimizer_updates": iteration,
                    "source_inventory_sha256": source_hash,
                    "metrics": status.get("metrics", {}),
                    "parameters": status.get("parameters"),
                    "guards": status.get("guards", {}),
                    "submission_eligible": status.get("submission_eligible", False),
                    "trajectory": status.get("trajectory", []),
                    "early_stop": status.get("early_stop"),
                    "updated_at": status["updated_at"],
                },
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    if status["status"] != "completed":
        raise SystemExit(args.exit_code or 1)


def summarize(args: argparse.Namespace) -> None:
    attempt = args.attempt.resolve()
    status = json.loads((attempt / "status.json").read_text())
    print(json.dumps(status, indent=2, sort_keys=True))


def print_source_hash(args: argparse.Namespace) -> None:
    """Print the deterministic inventory hash used by the rung gates."""
    source_hash, _ = project_inventory(args.project.resolve())
    print(source_hash)


def signal_training_workers(attempt: Path, process_root: Path = Path("/proc")) -> list[int]:
    """Send SIGTERM only to this attempt's pretraining worker processes."""
    checkpoint_argument = str(attempt / "checkpoints")
    stopped: list[int] = []
    current_uid = os.getuid()
    for process in process_root.iterdir():
        if not process.name.isdigit():
            continue
        try:
            if process.stat().st_uid != current_uid:
                continue
            command = (process / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
            if "pretrain_gpt_marin_adamh.py" not in command or checkpoint_argument not in command:
                continue
            os.kill(int(process.name), signal.SIGTERM)
            stopped.append(int(process.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return stopped


def stop(args: argparse.Namespace) -> None:
    """Request a scoped early stop for one currently running attempt."""
    attempt = args.attempt.resolve()
    output_root = Path(os.environ.get("POSITIONAL_OUTPUT_ROOT", "/app/output")).resolve()
    attempts_root = output_root / "attempts"
    if attempt.parent != attempts_root or attempt.is_symlink() or not attempt.is_dir():
        raise SystemExit(f"attempt must be a running direct child of {attempts_root}: {attempt}")
    status = json.loads((attempt / "status.json").read_text())
    if status.get("status") != "running":
        raise SystemExit(f"attempt is not running: {status.get('status')}")
    reason = args.reason.strip()
    if not reason or len(reason) > 2000:
        raise SystemExit("--reason must contain 1-2000 non-whitespace characters")
    atomic_json(
        attempt / "early_stop.json",
        {"requested_at": utc_now(), "reason": reason},
    )

    stopped = signal_training_workers(attempt)
    if not stopped:
        raise SystemExit("early-stop request recorded, but no matching training workers were found")
    print(json.dumps({"attempt": str(attempt), "signaled_pids": stopped}, indent=2))


def prune(args: argparse.Namespace) -> None:
    """Remove bulky state from a completed rejected attempt, preserving evidence."""
    attempt = args.attempt.resolve()
    output_root = Path(os.environ.get("POSITIONAL_OUTPUT_ROOT", "/app/output")).resolve()
    attempts_root = output_root / "attempts"
    if attempt.parent != attempts_root:
        raise SystemExit(f"attempt must be a direct child of {attempts_root}: {attempt}")
    if attempt.is_symlink() or not attempt.is_dir():
        raise SystemExit(f"attempt must be a real directory: {attempt}")

    status_path = attempt / "status.json"
    status = json.loads(status_path.read_text())
    if status.get("status") not in {"completed", "early_stopped"}:
        raise SystemExit("only a completed or early-stopped attempt may be pruned")

    submission_path = output_root / "submission.json"
    run_contract = json.loads((attempt / "run_contract.json").read_text())
    attempt_id = safe_identifier(run_contract.get("attempt_id"), "attempt_id")
    if submission_path.is_file():
        submission = json.loads(submission_path.read_text())
        selected = {
            item.get("attempt_id")
            for item in submission.get("scales", {}).values()
            if isinstance(item, dict)
        }
        if attempt_id in selected:
            raise SystemExit("refusing to prune an attempt in the currently staged ladder")

    removed: list[str] = []
    for name in ("checkpoints", "tensorboard", "wandb", "runtime-generated"):
        target = attempt / name
        if not target.exists():
            continue
        if target.is_symlink() or target.parent != attempt:
            raise SystemExit(f"refusing unsafe prune target: {target}")
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        removed.append(name)
    source_snapshot = output_root / "source-snapshots" / attempt_id
    if source_snapshot.exists():
        if source_snapshot.is_symlink() or source_snapshot.parent != output_root / "source-snapshots":
            raise SystemExit(f"refusing unsafe source-snapshot prune target: {source_snapshot}")
        shutil.rmtree(source_snapshot)
        removed.append("source-snapshot")
    source_hash = run_contract.get("source_inventory_sha256")
    running_same_source = False
    if isinstance(source_hash, str) and re.fullmatch(r"[0-9a-f]{64}", source_hash):
        for candidate_status in attempts_root.glob("*/status.json"):
            try:
                candidate = json.loads(candidate_status.read_text())
                candidate_contract = json.loads(
                    (candidate_status.parent / "run_contract.json").read_text()
                )
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if (
                candidate.get("status") == "running"
                and candidate_contract.get("source_inventory_sha256") == source_hash
            ):
                running_same_source = True
                break
        source_worktree = output_root / "source-worktrees" / source_hash
        if source_worktree.is_dir() and not source_worktree.is_symlink() and not running_same_source:
            shutil.rmtree(source_worktree)
            removed.append("source-worktree")
    status["checkpoints_pruned"] = "checkpoints" in removed
    status["pruned_paths"] = removed
    status["pruned_at"] = utc_now()
    atomic_json(status_path, status)
    print(json.dumps({"attempt": str(attempt), "removed": removed}, indent=2, sort_keys=True))


def stage(args: argparse.Namespace) -> None:
    attempt = args.attempt.resolve()
    project = args.project.resolve()
    output_root = Path(os.environ.get("POSITIONAL_OUTPUT_ROOT", "/app/output")).resolve()
    if output_root not in attempt.parents:
        raise SystemExit(f"attempt must be below {output_root}: {attempt}")
    scale, run_contract = attempt_contract(attempt)
    if scale != SCALE_ORDER[-1]:
        raise SystemExit("only a completed E5 attempt may anchor a staged ladder")
    final_iteration = SCALE_CONTRACTS[scale]["iterations"]
    status = json.loads((attempt / "status.json").read_text())
    if status.get("status") != "completed" or status.get("checkpoint_iteration") != final_iteration:
        raise SystemExit(f"only a completed {scale} update-{final_iteration} attempt may be staged")
    if not status.get("submission_eligible"):
        raise SystemExit(f"E5 candidate does not pass every scale-relative gate: {status.get('guards', {})}")
    baselines = load_baselines(args.baselines)
    measured_metrics = read_metrics(attempt, final_iteration)
    measured_parameters = read_parameter_count(attempt)
    measured_guards = metric_guards(scale, measured_metrics, measured_parameters, baselines)
    if not all(measured_guards.values()):
        raise SystemExit(f"raw attempt evidence no longer passes every staging gate: {measured_guards}")
    if status.get("metrics") != measured_metrics or status.get("parameters") != measured_parameters:
        raise SystemExit("status metrics differ from the retained raw attempt evidence")
    if not (attempt / "checkpoints").is_dir():
        raise SystemExit("selected candidate checkpoint was pruned or is missing")
    attempt_id = safe_identifier(run_contract.get("attempt_id"), "attempt_id")
    source_hash, changed_paths = project_inventory(project)
    if source_hash != run_contract.get("source_inventory_sha256"):
        raise SystemExit("source changed after the selected run; rerun the candidate before staging")
    ladder_attempts = passing_attempts(output_root, source_hash)
    missing = [item for item in SCALE_ORDER if item not in ladder_attempts]
    if missing:
        raise SystemExit(f"same-source ladder is incomplete; missing passing scales: {missing}")
    if not args.hypothesis_file.is_file() or not args.hypothesis_file.read_text().strip():
        raise SystemExit("a non-empty hypothesis file is required")
    source_manifest = snapshot_source(
        project,
        output_root,
        attempt_id,
        changed_paths,
        source_hash,
    )
    hypothesis_target = output_root / "HYPOTHESIS.md"
    if args.hypothesis_file.resolve() != hypothesis_target:
        hypothesis_target.write_bytes(args.hypothesis_file.read_bytes())
    scales: dict[str, dict[str, Any]] = {}
    aggregate_log_gain = 0.0
    for ladder_scale in SCALE_ORDER:
        ladder_attempt = ladder_attempts[ladder_scale]
        ladder_status = json.loads((ladder_attempt / "status.json").read_text())
        ladder_contract = SCALE_CONTRACTS[ladder_scale]
        ladder_iteration = ladder_contract["iterations"]
        ladder_metrics = read_metrics(ladder_attempt, ladder_iteration)
        ladder_parameters = read_parameter_count(ladder_attempt)
        ladder_guards = metric_guards(ladder_scale, ladder_metrics, ladder_parameters, baselines)
        if not all(ladder_guards.values()):
            raise SystemExit(f"retained {ladder_scale} evidence no longer passes: {ladder_guards}")
        if ladder_status.get("source_inventory_sha256") != source_hash:
            raise SystemExit(f"retained {ladder_scale} source hash differs from staged source")
        checkpoint_dir = ladder_attempt / "checkpoints"
        if not checkpoint_dir.is_dir():
            raise SystemExit(f"retained {ladder_scale} checkpoint is missing or pruned")
        relative_attempt = ladder_attempt.relative_to(output_root)
        aggregate_log_gain += math.log(baselines[ladder_scale]["longppl"] / ladder_metrics["longppl"])
        scales[ladder_scale] = {
            "attempt_id": ladder_attempt.name,
            "checkpoint_dir": str(relative_attempt / "checkpoints"),
            "training_eval_results": str(
                relative_attempt / "eval_harness" / f"step_{ladder_iteration:08d}" / "results.json"
            ),
            "longppl_results": str(
                relative_attempt / "longppl" / f"step_{ladder_iteration:08d}" / "results.json"
            ),
            "metrics": ladder_metrics,
            "parameters": ladder_parameters,
            "guards": ladder_guards,
        }
    aggregate_log_gain /= len(SCALE_ORDER)

    existing_submission_path = output_root / "submission.json"
    if existing_submission_path.is_file():
        incumbent = json.loads(existing_submission_path.read_text())
        if aggregate_log_gain <= float(incumbent.get("aggregate_log_longppl_gain", float("-inf"))):
            raise SystemExit("complete ladder passes but does not improve the staged incumbent")

    submission = {
        "version": 2,
        "attempt_id": attempt_id,
        "selected_at": utc_now(),
        "anchor_scale": scale,
        "hypothesis_file": "HYPOTHESIS.md",
        "source_inventory_sha256": source_hash,
        "source_snapshot": str(source_manifest.relative_to(output_root)),
        "changed_paths": changed_paths,
        "scale_order": list(SCALE_ORDER),
        "scales": scales,
        "aggregate_log_longppl_gain": aggregate_log_gain,
    }
    atomic_json(output_root / "submission.json", submission)
    print(json.dumps(submission, indent=2, sort_keys=True))


def restore_source(args: argparse.Namespace) -> None:
    """Restore the staged incumbent source into the verifier project path."""
    output_root = args.output_root.resolve()
    clean_project = args.clean_project.resolve()
    project = args.project.resolve()
    if not clean_project.is_dir() or clean_project.is_symlink():
        raise SystemExit(f"clean verifier project is missing or unsafe: {clean_project}")
    if project == clean_project or project == Path("/") or project.parent == project:
        raise SystemExit(f"unsafe verifier project target: {project}")
    submission = json.loads((output_root / "submission.json").read_text())
    manifest_relative = safe_relative_path(submission.get("source_snapshot"), "source_snapshot")
    manifest_path = output_root / manifest_relative
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise SystemExit(f"incumbent source snapshot is missing or unsafe: {manifest_path}")
    require_resolved_below(output_root, manifest_path, "incumbent source snapshot")

    temporary = project.parent / f".{project.name}.incumbent-restore-{os.getpid()}"
    if temporary.exists() or temporary.is_symlink():
        raise SystemExit(f"verifier restore temporary path already exists: {temporary}")
    try:
        shutil.copytree(clean_project, temporary, symlinks=True)
        expected_hash = apply_source_snapshot(temporary, manifest_path)
        actual_hash, _ = project_inventory(temporary)
        if actual_hash != expected_hash or actual_hash != submission.get("source_inventory_sha256"):
            raise SystemExit("restored incumbent source does not match its staged inventory hash")
        if project.is_symlink() or not project.is_dir():
            raise SystemExit(f"existing verifier project is missing or unsafe: {project}")
        shutil.rmtree(project)
        temporary.replace(project)
    except BaseException:
        if temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        raise
    print(project)


def materialize_attempt_source(args: argparse.Namespace) -> None:
    """Rebuild one attempt's immutable source from the clean image project."""
    attempt = args.attempt.resolve()
    output_root = Path(os.environ.get("POSITIONAL_OUTPUT_ROOT", "/app/output")).resolve()
    require_resolved_below(output_root, attempt, "attempt")
    run_contract = json.loads((attempt / "run_contract.json").read_text())
    manifest_relative = safe_relative_path(
        run_contract.get("source_snapshot"), "attempt source_snapshot"
    )
    manifest_path = output_root / manifest_relative
    require_resolved_below(output_root, manifest_path, "attempt source snapshot")
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise SystemExit(f"attempt source snapshot is missing or unsafe: {manifest_path}")
    clean_project = args.clean_project.resolve()
    project = args.project.resolve()
    if not clean_project.is_dir() or clean_project.is_symlink():
        raise SystemExit(f"clean project is missing or unsafe: {clean_project}")
    if project.exists() or project.is_symlink() or project == Path("/"):
        raise SystemExit(f"source materialization target must not exist: {project}")
    project.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(clean_project, project, symlinks=True)
    try:
        expected_hash = apply_source_snapshot(project, manifest_path)
        actual_hash, _ = project_inventory(project)
        if actual_hash != expected_hash or actual_hash != run_contract.get(
            "source_inventory_sha256"
        ):
            raise SystemExit("materialized attempt source does not match its launch hash")
    except BaseException:
        if project.is_dir() and not project.is_symlink():
            shutil.rmtree(project)
        raise
    print(project)


def restore_attempt_source(args: argparse.Namespace) -> None:
    """Restore a prior attempt into /app/project for another run or inspection."""
    project = args.project.resolve()
    temporary = project.parent / f".{project.name}.attempt-restore-{os.getpid()}"
    materialize_attempt_source(
        argparse.Namespace(
            attempt=args.attempt,
            clean_project=args.clean_project,
            project=temporary,
        )
    )
    try:
        if project.is_symlink() or not project.is_dir():
            raise SystemExit(f"existing candidate project is missing or unsafe: {project}")
        shutil.rmtree(project)
        temporary.replace(project)
    except BaseException:
        if temporary.is_dir() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        raise
    print(project)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subparsers = result.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--attempt", type=Path, required=True)
    prepare_parser.add_argument("--attempt-id", required=True)
    prepare_parser.add_argument("--scale", choices=SCALE_ORDER, required=True)
    prepare_parser.add_argument("--hypothesis-file", type=Path, required=True)
    prepare_parser.add_argument("--project", type=Path, default=Path("/app/project"))
    prepare_parser.add_argument("--baselines", type=Path)
    prepare_parser.set_defaults(handler=prepare)
    finish_parser = subparsers.add_parser("finish")
    finish_parser.add_argument("--attempt", type=Path, required=True)
    finish_parser.add_argument("--project", type=Path, default=Path("/app/project"))
    finish_parser.add_argument("--exit-code", type=int, required=True)
    finish_parser.add_argument("--baselines", type=Path)
    finish_parser.set_defaults(handler=finish)
    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--attempt", type=Path, required=True)
    summary_parser.set_defaults(handler=summarize)
    hash_parser = subparsers.add_parser("hash")
    hash_parser.add_argument("--project", type=Path, default=Path("/app/project"))
    hash_parser.set_defaults(handler=print_source_hash)
    stop_parser = subparsers.add_parser("stop")
    stop_parser.add_argument("--attempt", type=Path, required=True)
    stop_parser.add_argument("--reason", required=True)
    stop_parser.set_defaults(handler=stop)
    prune_parser = subparsers.add_parser("prune")
    prune_parser.add_argument("--attempt", type=Path, required=True)
    prune_parser.set_defaults(handler=prune)
    stage_parser = subparsers.add_parser("stage")
    stage_parser.add_argument("--attempt", type=Path, required=True)
    stage_parser.add_argument("--hypothesis-file", type=Path, required=True)
    stage_parser.add_argument("--project", type=Path, default=Path("/app/project"))
    stage_parser.add_argument("--baselines", type=Path)
    stage_parser.set_defaults(handler=stage)
    restore_parser = subparsers.add_parser("restore-source")
    restore_parser.add_argument("--output-root", type=Path, default=Path("/app/output"))
    restore_parser.add_argument("--clean-project", type=Path, default=Path("/opt/project"))
    restore_parser.add_argument("--project", type=Path, default=Path("/app/project"))
    restore_parser.set_defaults(handler=restore_source)
    materialize_parser = subparsers.add_parser("materialize-attempt-source")
    materialize_parser.add_argument("--attempt", type=Path, required=True)
    materialize_parser.add_argument("--clean-project", type=Path, default=Path("/opt/project"))
    materialize_parser.add_argument("--project", type=Path, required=True)
    materialize_parser.set_defaults(handler=materialize_attempt_source)
    restore_attempt_parser = subparsers.add_parser("restore-attempt-source")
    restore_attempt_parser.add_argument("--attempt", type=Path, required=True)
    restore_attempt_parser.add_argument("--clean-project", type=Path, default=Path("/opt/project"))
    restore_attempt_parser.add_argument("--project", type=Path, default=Path("/app/project"))
    restore_attempt_parser.set_defaults(handler=restore_attempt_source)
    return result


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
