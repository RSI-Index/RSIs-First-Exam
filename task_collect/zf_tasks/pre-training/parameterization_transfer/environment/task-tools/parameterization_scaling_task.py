#!/usr/bin/env python3
"""Prepare, gate, summarize, and stage parameterization scaling-ladder attempts."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SCALE_ORDER = ("E0", "E1", "E2", "E3", "E4")
SCALE_CONTRACTS: dict[str, dict[str, int]] = {
    "E0": {"layers": 12, "hidden": 1152, "ffn": 4608, "heads": 9, "parameters": 550_337_664, "tokens": 2_904_358_912, "iterations": 44_317, "gbs": 16, "mbs": 2, "gpus": 8},
    "E1": {"layers": 15, "hidden": 1408, "ffn": 5632, "heads": 11, "parameters": 837_007_744, "tokens": 3_612_672_000, "iterations": 55_125, "gbs": 16, "mbs": 2, "gpus": 8},
    "E2": {"layers": 16, "hidden": 1536, "ffn": 6144, "heads": 12, "parameters": 998_036_992, "tokens": 4_982_571_008, "iterations": 38_014, "gbs": 32, "mbs": 4, "gpus": 8},
    "E3": {"layers": 18, "hidden": 1792, "ffn": 7168, "heads": 14, "parameters": 1_384_584_448, "tokens": 10_559_946_752, "iterations": 40_283, "gbs": 64, "mbs": 2, "gpus": 32},
    "E4": {"layers": 21, "hidden": 2048, "ffn": 8192, "heads": 16, "parameters": 1_934_716_160, "tokens": 14_805_106_688, "iterations": 56_477, "gbs": 64, "mbs": 2, "gpus": 32},
}
SCALE_FULL_FLOPS = {
    "E0": 9.0e18,
    "E1": 1.8e19,
    "E2": 3.0e19,
    "E3": 9.0e19,
    "E4": 1.8e20,
}
VISIBLE_FLOP_BUDGET = 2.4e20
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
        "evaluation": {"paloma": True, "training_loss": True, "active_gpu_time": True},
    }


def attempt_contract(attempt: Path) -> tuple[str, dict[str, Any]]:
    payload = json.loads((attempt / "run_contract.json").read_text())
    scale = payload.get("scale")
    if scale not in SCALE_CONTRACTS:
        raise ValueError(f"attempt has invalid scale: {scale}")
    return scale, payload


def load_baselines(path: Path | None = None) -> dict[str, dict[str, Any]]:
    baseline_path = path or Path(
        os.environ.get("PARAMETERIZATION_SCALING_BASELINES", "/task-data/scale_baselines.json")
    )
    payload = json.loads(baseline_path.read_text())
    if payload.get("version") != 2 or not set(SCALE_ORDER).issubset(payload.get("scales", {})):
        raise ValueError(f"invalid visible-scale baseline manifest: {baseline_path}")
    result: dict[str, dict[str, Any]] = {}
    for scale in SCALE_ORDER:
        raw = payload["scales"][scale]
        if any(float(raw[key]) <= 0 for key in (
            "paloma_bits_per_byte",
            "paloma_macro_bits_per_byte",
            "scoring_gpu_seconds",
            "scoring_fixed_window_loss",
        )):
            raise ValueError(f"non-positive baseline metric for {scale}")
        result[scale] = raw
    return result


def charged_attempt_cost(attempt: Path) -> tuple[float, int, str | None]:
    """Recover charged work even when an attempt failed before finish()."""
    trace_path = attempt / "parameterization_cost_trace.jsonl"
    if not trace_path.is_file():
        return 0.0, 0, None
    resume_iteration = 0
    contract_path = attempt / "run_contract.json"
    if contract_path.is_file():
        contract = json.loads(contract_path.read_text())
        resume = contract.get("resume")
        if isinstance(resume, dict):
            resume_iteration = int(resume.get("iteration", 0))
    last_cost = 0.0
    last_iteration = resume_iteration
    charged_updates = 0
    try:
        for line_number, line in enumerate(trace_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            iteration = int(row["iteration"])
            cost = float(row["cumulative_gpu_seconds"])
            if iteration != last_iteration + 1 or not math.isfinite(cost) or cost <= last_cost:
                return last_cost, charged_updates, f"invalid cost trace at line {line_number}"
            last_iteration = iteration
            last_cost = cost
            charged_updates += 1
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return last_cost, charged_updates, f"unreadable cost trace: {error}"
    return last_cost, charged_updates, None


def research_budget_report(
    output_root: Path,
    baselines: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Charge all visible work against the task-wide 2.4e20-FLOP budget."""
    attempts_root = output_root / "attempts"
    by_scale: dict[str, dict[str, Any]] = {
        scale: {
            "charged_gpu_seconds": 0.0,
            "charged_training_flops": 0.0,
            "charged_training_tokens": 0,
            "attempts": [],
            "compliant": True,
        }
        for scale in SCALE_ORDER
    }
    invalid_attempts: list[dict[str, str]] = []
    running_attempts: list[str] = []
    if attempts_root.is_dir():
        for attempt in sorted(attempts_root.iterdir()):
            contract_path = attempt / "run_contract.json"
            if attempt.is_symlink() or not attempt.is_dir() or not contract_path.is_file():
                continue
            try:
                scale, contract = attempt_contract(attempt)
                attempt_id = safe_identifier(contract.get("attempt_id"), "attempt_id")
                if attempt_id != attempt.name:
                    raise ValueError("attempt directory and contract ID differ")
                cost, updates, trace_error = charged_attempt_cost(attempt)
                status_path = attempt / "status.json"
                status = (
                    json.loads(status_path.read_text()).get("status", "missing")
                    if status_path.is_file()
                    else "missing"
                )
                if status == "running":
                    running_attempts.append(attempt_id)
                entry = {
                    "attempt_id": attempt_id,
                    "status": status,
                    "charged_gpu_seconds": cost,
                    "charged_updates": updates,
                    "charged_training_tokens": updates * SCALE_CONTRACTS[scale]["gbs"] * 4096,
                    "charged_training_flops": (
                        SCALE_FULL_FLOPS[scale]
                        * updates
                        / SCALE_CONTRACTS[scale]["iterations"]
                    ),
                    "trace_error": trace_error,
                }
                by_scale[scale]["attempts"].append(entry)
                by_scale[scale]["charged_gpu_seconds"] += cost
                by_scale[scale]["charged_training_tokens"] += entry["charged_training_tokens"]
                by_scale[scale]["charged_training_flops"] += entry["charged_training_flops"]
                if trace_error is not None:
                    by_scale[scale]["compliant"] = False
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                invalid_attempts.append({"attempt": attempt.name, "error": str(error)})
    total_flops = sum(by_scale[scale]["charged_training_flops"] for scale in SCALE_ORDER)
    total_tokens = sum(by_scale[scale]["charged_training_tokens"] for scale in SCALE_ORDER)
    total_gpu_seconds = sum(by_scale[scale]["charged_gpu_seconds"] for scale in SCALE_ORDER)
    tolerance = max(1.0, 1e-9 * VISIBLE_FLOP_BUDGET)
    within_budget = total_flops <= VISIBLE_FLOP_BUDGET + tolerance
    return {
        "version": 2,
        "budget_coordinate": "charged_visible_training_flops",
        "budget_training_flops": VISIBLE_FLOP_BUDGET,
        "charged_training_flops": total_flops,
        "remaining_training_flops": VISIBLE_FLOP_BUDGET - total_flops,
        "charged_training_tokens": total_tokens,
        "charged_gpu_seconds": total_gpu_seconds,
        "scales": by_scale,
        "invalid_attempts": invalid_attempts,
        "running_attempts": running_attempts,
        "compliant": (
            not invalid_attempts
            and not running_attempts
            and within_budget
            and all(by_scale[scale]["compliant"] for scale in SCALE_ORDER)
        ),
    }


def read_cost_trace(attempt: Path) -> list[dict[str, Any]]:
    trace_path = attempt / "parameterization_cost_trace.jsonl"
    rows: list[dict[str, Any]] = []
    previous_iteration = 0
    contract_path = attempt / "run_contract.json"
    if contract_path.is_file():
        contract = json.loads(contract_path.read_text())
        resume = contract.get("resume")
        if isinstance(resume, dict):
            previous_iteration = int(resume.get("iteration", 0))
    previous_cost = -1.0
    for line in trace_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        iteration = int(row["iteration"])
        cost = float(row["cumulative_gpu_seconds"])
        loss = row.get("lm_loss")
        if iteration != previous_iteration + 1 or cost <= previous_cost:
            raise ValueError("parameterization cost trace is non-contiguous or non-monotonic")
        if loss is None or not math.isfinite(float(loss)) or float(loss) <= 0:
            raise ValueError(f"invalid training loss at iteration {iteration}")
        if int(row.get("skipped", 0) or 0) != 0:
            raise ValueError(f"skipped parameterization update at iteration {iteration}")
        rows.append(row)
        previous_iteration = iteration
        previous_cost = cost
    if not rows:
        raise ValueError("parameterization cost trace is empty")
    return rows


def resume_source_attempt(attempt: Path) -> tuple[Path, int] | None:
    """Resolve and validate the immutable predecessor named by a resume contract."""
    if not (attempt / "run_contract.json").is_file():
        return None
    _, contract = attempt_contract(attempt)
    resume = contract.get("resume")
    if resume is None:
        return None
    if not isinstance(resume, dict):
        raise ValueError("invalid resume contract")
    iteration = int(resume.get("iteration", 0))
    if iteration <= 0:
        raise ValueError("invalid resume iteration")
    attempt_id = safe_identifier(resume.get("attempt_id"), "resume attempt_id")
    attempts_root = attempt.parent.resolve()
    source = (attempts_root / attempt_id).resolve()
    if source.parent != attempts_root or source == attempt.resolve():
        raise ValueError("unsafe or cyclic resume attempt")
    source_scale, source_contract = attempt_contract(source)
    if source_scale != contract["scale"]:
        raise ValueError("resume source scale differs from resumed attempt")
    if source_contract.get("source_inventory_sha256") != contract.get(
        "source_inventory_sha256"
    ):
        raise ValueError("resume source inventory differs from resumed attempt")
    return source, iteration


def read_scoring_cost_trace(
    attempt: Path,
    seen: set[Path] | None = None,
) -> list[dict[str, Any]]:
    """Join checkpoint-prefix and resumed-suffix traces for scientific scoring.

    Research-budget charging deliberately remains local to each attempt via
    ``charged_attempt_cost``.  This view instead represents the one logical
    training trajectory that reached the final checkpoint.
    """
    resolved = attempt.resolve()
    visited = set() if seen is None else set(seen)
    if resolved in visited:
        raise ValueError("cyclic resume chain")
    visited.add(resolved)
    current = read_cost_trace(resolved)
    predecessor = resume_source_attempt(resolved)
    if predecessor is None:
        return current
    source, resume_iteration = predecessor
    source_trace = read_scoring_cost_trace(source, visited)
    prefix = [
        dict(row)
        for row in source_trace
        if int(row["iteration"]) <= resume_iteration
    ]
    if (
        len(prefix) != resume_iteration
        or not prefix
        or int(prefix[-1]["iteration"]) != resume_iteration
    ):
        raise ValueError("resume source trace does not cover the checkpoint prefix")
    if int(current[0]["iteration"]) != resume_iteration + 1:
        raise ValueError("resumed trace does not begin after the checkpoint")
    prefix_cost = float(prefix[-1]["cumulative_gpu_seconds"])
    suffix: list[dict[str, Any]] = []
    for row in current:
        adjusted = dict(row)
        adjusted["cumulative_gpu_seconds"] = (
            prefix_cost + float(row["cumulative_gpu_seconds"])
        )
        suffix.append(adjusted)
    return prefix + suffix


def fixed_window_loss_curve(scale: str, trace: list[dict[str, Any]]) -> list[dict[str, float | int]]:
    tokens_per_update = SCALE_CONTRACTS[scale]["gbs"] * 4096
    window_steps = math.ceil(16_777_216 / tokens_per_update)
    losses: list[float] = []
    running = 0.0
    result: list[dict[str, float | int]] = []
    for row in trace:
        value = float(row["lm_loss"])
        losses.append(value)
        running += value
        if len(losses) > window_steps:
            running -= losses[-window_steps - 1]
        result.append(
            {
                "iteration": int(row["iteration"]),
                "cumulative_gpu_seconds": float(row["cumulative_gpu_seconds"]),
                "fixed_window_loss": running / min(len(losses), window_steps),
            }
        )
    return result


def read_paloma_trajectory(attempt: Path, trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cost_by_iteration = {
        int(row["iteration"]): float(row["cumulative_gpu_seconds"]) for row in trace
    }
    result: list[dict[str, Any]] = []
    for result_path in sorted((attempt / "eval_harness").glob("step_*/results.json")):
        iteration = int(result_path.parent.name.removeprefix("step_"))
        if iteration not in cost_by_iteration:
            continue
        paloma = json.loads(result_path.read_text())["paloma_aggregate"]
        result.append(
            {
                "iteration": iteration,
                "cumulative_gpu_seconds": cost_by_iteration[iteration],
                "paloma_bits_per_byte": float(paloma["bits_per_byte"]),
                "paloma_macro_bits_per_byte": float(paloma["macro_bits_per_byte"]),
            }
        )
    if not result:
        raise ValueError("attempt has no Paloma trajectory joined to its cost trace")
    return result


def read_scoring_paloma_trajectory(
    attempt: Path,
    trace: list[dict[str, Any]] | None = None,
    seen: set[Path] | None = None,
) -> list[dict[str, Any]]:
    """Join evaluation points from every retained prefix in a resume chain."""
    resolved = attempt.resolve()
    visited = set() if seen is None else set(seen)
    if resolved in visited:
        raise ValueError("cyclic resume chain")
    visited.add(resolved)
    scoring_trace = trace if trace is not None else read_scoring_cost_trace(resolved)
    try:
        local = read_paloma_trajectory(resolved, scoring_trace)
    except ValueError as error:
        if str(error) != "attempt has no Paloma trajectory joined to its cost trace":
            raise
        local = []
    predecessor = resume_source_attempt(resolved)
    prefix: list[dict[str, Any]] = []
    if predecessor is not None:
        source, resume_iteration = predecessor
        try:
            source_points = read_scoring_paloma_trajectory(source, seen=visited)
        except ValueError as error:
            if str(error) != "attempt has no Paloma trajectory joined to its cost trace":
                raise
            source_points = []
        prefix = [
            row for row in source_points if int(row["iteration"]) <= resume_iteration
        ]
    joined = {int(row["iteration"]): row for row in (*prefix, *local)}
    result = [joined[index] for index in sorted(joined)]
    if not result:
        raise ValueError("attempt has no Paloma trajectory joined to its cost trace")
    return result


def read_metrics(attempt: Path, final_iteration: int, scale: str | None = None) -> dict[str, Any]:
    harness_path = attempt / "eval_harness" / f"step_{final_iteration:08d}" / "results.json"
    harness = json.loads(harness_path.read_text())
    paloma = harness["paloma_aggregate"]
    metrics: dict[str, Any] = {
        "paloma_bits_per_byte": float(paloma["bits_per_byte"]),
        "paloma_macro_bits_per_byte": float(paloma["macro_bits_per_byte"]),
    }
    if scale is not None:
        trace = read_scoring_cost_trace(attempt)
        if int(trace[-1]["iteration"]) != final_iteration:
            raise ValueError("cost trace does not end at the final checkpoint")
        loss_curve = fixed_window_loss_curve(scale, trace)
        metrics.update(
            {
                "final_gpu_seconds": float(trace[-1]["cumulative_gpu_seconds"]),
                "final_fixed_window_loss": float(loss_curve[-1]["fixed_window_loss"]),
                "cost_trace_rows": len(trace),
            }
        )
    return metrics


def read_metric_trajectory(attempt: Path) -> list[dict[str, Any]]:
    """Read every complete intermediate Paloma result at measured cost."""
    trace = read_scoring_cost_trace(attempt)
    return read_scoring_paloma_trajectory(attempt, trace)


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
    baselines: dict[str, dict[str, Any]] | None = None,
) -> dict[str, bool]:
    """Evaluate per-run integrity; scientific quality is a six-rung aggregate."""
    del baselines
    return {
        "finite_paloma": all(
            math.isfinite(float(metrics[key])) and float(metrics[key]) > 0
            for key in ("paloma_bits_per_byte", "paloma_macro_bits_per_byte")
        ),
        "complete_cost_trace": int(metrics.get("cost_trace_rows", 0))
        == SCALE_CONTRACTS[scale]["iterations"],
        "fixed_model_parameters": parameters == SCALE_CONTRACTS[scale]["parameters"],
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
            and status.get("run_valid") is True
            and run_contract.get("source_inventory_sha256") == source_hash
        ):
            result[scale] = status_path.parent
    return result


def active_resume_budget_hold(scale: str) -> tuple[Path, dict[str, Any]] | None:
    """Return a trusted active recovery hold that blocks checkpoint continuation."""
    contract_root = Path(
        os.environ.get("PARAMETERIZATION_RUN_CONTRACT_ROOT", "/run-contract")
    )
    if not contract_root.is_dir():
        return None
    for path in sorted(contract_root.glob("CONTROL_HOLD_*_BUDGET_*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        gate = payload.get("gate")
        if (
            payload.get("type") == "trusted_budget_safety_hold"
            and payload.get("status") == "active"
            and payload.get("applies_to_scale") == scale
            and isinstance(gate, dict)
            and gate.get("status") == "hold"
        ):
            return path, payload
    return None


def prepare(args: argparse.Namespace) -> None:
    attempt = args.attempt.resolve()
    output_root = Path(os.environ.get("PARAMETERIZATION_OUTPUT_ROOT", "/app/output")).resolve()
    if output_root not in attempt.parents:
        raise SystemExit(f"attempt must be below {output_root}: {attempt}")
    if attempt.exists():
        raise SystemExit(f"attempt already exists: {attempt}")
    scale = args.scale
    if scale not in SCALE_CONTRACTS:
        raise SystemExit(f"unknown scale: {scale}")
    resume_iteration = int(getattr(args, "resume_iteration", 0) or 0)
    resume_attempt_id = getattr(args, "resume_attempt_id", None)
    if resume_iteration:
        hold = active_resume_budget_hold(scale)
        if hold is not None:
            hold_path, _ = hold
            raise SystemExit(
                f"checkpoint continuation is held by trusted budget gate: {hold_path.name}"
            )
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
    run_contract = {
        "version": 1,
        "attempt_id": args.attempt_id,
        "scale": scale,
        "created_at": utc_now(),
        "source_inventory_sha256": source_hash,
        "source_snapshot": str(source_manifest.relative_to(output_root)),
        "changed_paths": changed_paths,
        "contract": scale_contract(scale),
    }
    if resume_iteration:
        if not resume_attempt_id:
            raise SystemExit("--resume-attempt-id is required with --resume-iteration")
        run_contract["resume"] = {
            "iteration": resume_iteration,
            "attempt_id": resume_attempt_id,
        }
    atomic_json(attempt / "run_contract.json", run_contract)
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
    charged_cost, charged_updates, trace_error = charged_attempt_cost(attempt)
    status["charged_gpu_seconds"] = charged_cost
    status["charged_updates"] = charged_updates
    if trace_error is not None:
        status["cost_trace_error"] = trace_error
    final_iteration = scale_values["iterations"]
    if args.exit_code == 0 and iteration == final_iteration and status["source_matches_launch"]:
        try:
            status["metrics"] = read_metrics(attempt, final_iteration, scale)
            status["trajectory"] = read_metric_trajectory(attempt)
            status["parameters"] = read_parameter_count(attempt)
            status["guards"] = metric_guards(
                scale,
                status["metrics"],
                status["parameters"],
                load_baselines(args.baselines),
            )
            status["run_valid"] = all(status["guards"].values())
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
                    "training_tokens": charged_updates * scale_values["gbs"] * 4096,
                    "training_flops": (
                        SCALE_FULL_FLOPS[scale]
                        * charged_updates
                        / scale_values["iterations"]
                    ),
                    "target_training_tokens": scale_values["tokens"],
                    "parameterization_updates": charged_updates,
                    "checkpoint_iteration": iteration,
                    "charged_gpu_seconds": charged_cost,
                    "cost_trace_error": trace_error,
                    "source_inventory_sha256": source_hash,
                    "metrics": status.get("metrics", {}),
                    "parameters": status.get("parameters"),
                    "guards": status.get("guards", {}),
                    "run_valid": status.get("run_valid", False),
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


def cost_report(args: argparse.Namespace) -> None:
    attempt = args.attempt.resolve()
    scale, contract = attempt_contract(attempt)
    cost, updates, trace_error = charged_attempt_cost(attempt)
    output_root = Path(os.environ.get("PARAMETERIZATION_OUTPUT_ROOT", "/app/output")).resolve()
    baselines = load_baselines(args.baselines)
    budgets = research_budget_report(output_root, baselines)
    print(json.dumps({
        "attempt_id": contract["attempt_id"],
        "scale": scale,
        "charged_gpu_seconds": cost,
        "charged_updates": updates,
        "trace_error": trace_error,
        "scale_budget": budgets["scales"][scale],
        "all_budgets_compliant": budgets["compliant"],
    }, indent=2, sort_keys=True))


def print_source_hash(args: argparse.Namespace) -> None:
    """Print the deterministic inventory hash used by the rung gates."""
    source_hash, _ = project_inventory(args.project.resolve())
    print(source_hash)


def signal_training_workers(attempt: Path, process_root: Path = Path("/proc")) -> list[int]:
    """Send SIGTERM only to this attempt's local worker or broker processes."""
    checkpoint_argument = str(attempt / "checkpoints")
    attempt_argument = str(attempt)
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
            direct_worker = (
                "pretrain_gpt_marin_adamh.py" in command
                and checkpoint_argument in command
            )
            broker_worker = (
                "blaunch_proxy_client.py" in command
                and "--host-worker" in command
                and attempt_argument in command
            )
            if not direct_worker and not broker_worker:
                continue
            os.kill(int(process.name), signal.SIGTERM)
            stopped.append(int(process.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return stopped


def stop(args: argparse.Namespace) -> None:
    """Request a scoped early stop for one currently running attempt."""
    attempt = args.attempt.resolve()
    output_root = Path(os.environ.get("PARAMETERIZATION_OUTPUT_ROOT", "/app/output")).resolve()
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
    output_root = Path(os.environ.get("PARAMETERIZATION_OUTPUT_ROOT", "/app/output")).resolve()
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


def interpolate_log_cost(
    curve: list[dict[str, Any]], target_cost: float, value_key: str
) -> float:
    points = sorted(
        (
            float(item["cumulative_gpu_seconds"]),
            float(item[value_key]),
        )
        for item in curve
        if float(item["cumulative_gpu_seconds"]) > 0
        and math.isfinite(float(item[value_key]))
    )
    if len(points) < 2 or target_cost < points[0][0] or target_cost > points[-1][0]:
        raise ValueError(
            f"target cost {target_cost} is outside measured {value_key} range "
            f"{points[0][0] if points else None}..{points[-1][0] if points else None}"
        )
    for index, (cost, value) in enumerate(points):
        if math.isclose(cost, target_cost, rel_tol=1e-12, abs_tol=1e-9):
            return value
        if cost > target_cost:
            low_cost, low_value = points[index - 1]
            alpha = (math.log(target_cost) - math.log(low_cost)) / (
                math.log(cost) - math.log(low_cost)
            )
            return low_value + alpha * (value - low_value)
    raise AssertionError("unreachable interpolation state")


def measured_cost_bounds(
    curve: list[dict[str, Any]], value_key: str
) -> tuple[float, float]:
    costs = sorted(
        float(item["cumulative_gpu_seconds"])
        for item in curve
        if float(item["cumulative_gpu_seconds"]) > 0
        and math.isfinite(float(item[value_key]))
    )
    if len(costs) < 2:
        raise ValueError(f"fewer than two measured {value_key} points")
    return costs[0], costs[-1]


def mean_auc_log_cost(
    curve: list[dict[str, Any]], start_cost: float, end_cost: float, value_key: str
) -> float:
    if not (0 < start_cost < end_cost):
        raise ValueError("invalid AUC cost interval")
    points: list[tuple[float, float]] = [
        (start_cost, interpolate_log_cost(curve, start_cost, value_key))
    ]
    points.extend(
        (float(item["cumulative_gpu_seconds"]), float(item[value_key]))
        for item in curve
        if start_cost < float(item["cumulative_gpu_seconds"]) < end_cost
    )
    points.append((end_cost, interpolate_log_cost(curve, end_cost, value_key)))
    points.sort()
    integral = 0.0
    for (left_cost, left_value), (right_cost, right_value) in zip(points, points[1:]):
        integral += 0.5 * (left_value + right_value) * (
            math.log(right_cost) - math.log(left_cost)
        )
    return integral / (math.log(end_cost) - math.log(start_cost))


def evaluate_matched_cost_scale(
    scale: str,
    attempt: Path,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    trace = read_scoring_cost_trace(attempt)
    paloma = read_scoring_paloma_trajectory(attempt, trace)
    loss_curve = fixed_window_loss_curve(scale, trace)
    candidate_final_cost = float(trace[-1]["cumulative_gpu_seconds"])
    baseline_scoring_cost = float(baseline["scoring_gpu_seconds"])
    scoring_cost = min(candidate_final_cost, baseline_scoring_cost)
    fractions = (1.0 / 3.0, 2.0 / 3.0, 1.0)
    weights = (0.2, 0.3, 0.5)
    nominal_start_cost = scoring_cost * fractions[0]
    candidate_start, candidate_end = measured_cost_bounds(
        paloma, "paloma_bits_per_byte"
    )
    baseline_start, baseline_end = measured_cost_bounds(
        baseline["paloma_trajectory"], "bits_per_byte"
    )
    measured_end = min(candidate_end, baseline_end)
    if scoring_cost > measured_end and not math.isclose(
        scoring_cost, measured_end, rel_tol=1e-12, abs_tol=1e-9
    ):
        raise ValueError(
            f"scoring cost {scoring_cost} exceeds shared measured Paloma endpoint "
            f"{measured_end}"
        )
    matched_start_cost = max(nominal_start_cost, candidate_start, baseline_start)
    if matched_start_cost > scoring_cost and not math.isclose(
        matched_start_cost, scoring_cost, rel_tol=1e-12, abs_tol=1e-9
    ):
        raise ValueError(
            f"no shared measured Paloma interval through scoring cost {scoring_cost}; "
            f"shared start would be {matched_start_cost}"
        )
    matched_cost_policy = (
        "nominal_fractions"
        if math.isclose(
            matched_start_cost, nominal_start_cost, rel_tol=1e-12, abs_tol=1e-9
        )
        else "common_measured_overlap"
    )
    matched: list[dict[str, float]] = []
    score = 0.0
    for fraction, weight in zip(fractions, weights):
        position = (fraction - fractions[0]) / (fractions[-1] - fractions[0])
        target = matched_start_cost + position * (scoring_cost - matched_start_cost)
        candidate_micro = interpolate_log_cost(paloma, target, "paloma_bits_per_byte")
        baseline_micro = interpolate_log_cost(
            baseline["paloma_trajectory"], target, "bits_per_byte"
        )
        candidate_macro = interpolate_log_cost(paloma, target, "paloma_macro_bits_per_byte")
        baseline_macro = interpolate_log_cost(
            baseline["paloma_trajectory"], target, "macro_bits_per_byte"
        )
        candidate_loss = interpolate_log_cost(loss_curve, target, "fixed_window_loss")
        baseline_loss = interpolate_log_cost(
            baseline["loss_cost_curve"], target, "fixed_window_loss"
        )
        score += weight * math.log(baseline_micro / candidate_micro)
        matched.append(
            {
                "fraction": fraction,
                "effective_fraction": target / scoring_cost,
                "weight": weight,
                "gpu_seconds": target,
                "candidate_micro_bpb": candidate_micro,
                "baseline_micro_bpb": baseline_micro,
                "candidate_macro_bpb": candidate_macro,
                "baseline_macro_bpb": baseline_macro,
                "candidate_fixed_window_loss": candidate_loss,
                "baseline_fixed_window_loss": baseline_loss,
            }
        )
    start_cost = scoring_cost / 3.0
    candidate_auc = mean_auc_log_cost(loss_curve, start_cost, scoring_cost, "fixed_window_loss")
    baseline_auc = mean_auc_log_cost(
        baseline["loss_cost_curve"], start_cost, scoring_cost, "fixed_window_loss"
    )
    return {
        "scale": scale,
        "score": score,
        "relative_paloma_gain": math.exp(score),
        "candidate_final_gpu_seconds": candidate_final_cost,
        "baseline_scoring_gpu_seconds": baseline_scoring_cost,
        "matched_scoring_gpu_seconds": scoring_cost,
        "nominal_start_gpu_seconds": nominal_start_cost,
        "shared_measured_start_gpu_seconds": matched_start_cost,
        "matched_cost_policy": matched_cost_policy,
        "matched": matched,
        "candidate_loss_auc": candidate_auc,
        "baseline_loss_auc": baseline_auc,
    }


def validate_mechanism_ablation(
    output_root: Path,
    selected_attempts: dict[str, Path],
    baselines: dict[str, dict[str, Any]],
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Recompute the preregistered full-versus-reduced mechanism gate."""
    path = manifest_path or output_root / "mechanism_ablation.json"
    payload = json.loads(path.read_text())
    anchors = ("E0", "E2", "E4")
    if payload.get("version") != 1 or set(payload.get("anchors", {})) != set(anchors):
        raise ValueError("mechanism_ablation.json must contain exactly E0, E2, and E4")
    results: dict[str, dict[str, Any]] = {}
    for scale in anchors:
        specification = payload["anchors"][scale]
        if not isinstance(specification, dict):
            raise ValueError(f"invalid mechanism ablation entry for {scale}")
        full_id = safe_identifier(specification.get("full_attempt_id"), "full_attempt_id")
        reduced_id = safe_identifier(
            specification.get("reduced_attempt_id"), "reduced_attempt_id"
        )
        full_attempt = selected_attempts[scale].resolve()
        reduced_attempt = (output_root / "attempts" / reduced_id).resolve()
        attempts_root = (output_root / "attempts").resolve()
        if full_id != full_attempt.name:
            raise ValueError(f"{scale} ablation full attempt is not the staged ladder attempt")
        if full_id == reduced_id or reduced_attempt.parent != attempts_root:
            raise ValueError(f"{scale} reduced attempt is invalid")
        pair: dict[str, dict[str, Any]] = {}
        for role, candidate in (("full", full_attempt), ("reduced", reduced_attempt)):
            candidate_scale, contract = attempt_contract(candidate)
            status = json.loads((candidate / "status.json").read_text())
            final_iteration = SCALE_CONTRACTS[scale]["iterations"]
            tracker = candidate / "checkpoints" / "latest_checkpointed_iteration.txt"
            if (
                candidate_scale != scale
                or status.get("status") != "completed"
                or status.get("run_valid") is not True
                or status.get("checkpoint_iteration") != final_iteration
                or not tracker.is_file()
                or int(tracker.read_text().strip()) != final_iteration
            ):
                raise ValueError(f"{scale} {role} mechanism is not an exact-final valid run")
            trace = read_scoring_cost_trace(candidate)
            pair[role] = {
                "attempt_id": candidate.name,
                "source_inventory_sha256": contract["source_inventory_sha256"],
                "trace": trace,
                "paloma": read_scoring_paloma_trajectory(candidate, trace),
                "loss": fixed_window_loss_curve(scale, trace),
                "final_gpu_seconds": float(trace[-1]["cumulative_gpu_seconds"]),
            }
        if pair["full"]["source_inventory_sha256"] == pair["reduced"]["source_inventory_sha256"]:
            raise ValueError(f"{scale} full and reduced mechanisms have identical source")
        matched_cost = min(
            pair["full"]["final_gpu_seconds"],
            pair["reduced"]["final_gpu_seconds"],
            float(baselines[scale]["scoring_gpu_seconds"]),
        )
        full_micro = interpolate_log_cost(
            pair["full"]["paloma"], matched_cost, "paloma_bits_per_byte"
        )
        reduced_micro = interpolate_log_cost(
            pair["reduced"]["paloma"], matched_cost, "paloma_bits_per_byte"
        )
        full_loss = interpolate_log_cost(
            pair["full"]["loss"], matched_cost, "fixed_window_loss"
        )
        reduced_loss = interpolate_log_cost(
            pair["reduced"]["loss"], matched_cost, "fixed_window_loss"
        )
        geometric_gain = math.sqrt((reduced_micro / full_micro) * (reduced_loss / full_loss))
        results[scale] = {
            "full_attempt_id": full_id,
            "reduced_attempt_id": reduced_id,
            "matched_gpu_seconds": matched_cost,
            "full_micro_bpb": full_micro,
            "reduced_micro_bpb": reduced_micro,
            "full_fixed_window_loss": full_loss,
            "reduced_fixed_window_loss": reduced_loss,
            "geometric_gain": geometric_gain,
            "passes": geometric_gain > 1.0,
        }
    return {
        "version": 1,
        "manifest": str(path.relative_to(output_root)),
        "anchors": results,
        "passes": all(item["passes"] for item in results.values()),
    }


def _legacy_quality_stage(args: argparse.Namespace) -> None:
    attempt = args.attempt.resolve()
    project = args.project.resolve()
    output_root = Path(os.environ.get("PARAMETERIZATION_OUTPUT_ROOT", "/app/output")).resolve()
    if output_root not in attempt.parents:
        raise SystemExit(f"attempt must be below {output_root}: {attempt}")
    scale, run_contract = attempt_contract(attempt)
    if scale != SCALE_ORDER[-1]:
        raise SystemExit("only a completed E4 attempt may anchor this legacy visible ladder")
    final_iteration = SCALE_CONTRACTS[scale]["iterations"]
    status = json.loads((attempt / "status.json").read_text())
    if status.get("status") != "completed" or status.get("checkpoint_iteration") != final_iteration:
        raise SystemExit(f"only a completed {scale} update-{final_iteration} attempt may be staged")
    baselines = load_baselines(args.baselines)
    if not status.get("run_valid"):
        raise SystemExit(f"E4 run integrity failed: {status.get('guards', {})}")
    if not (attempt / "checkpoints").is_dir():
        raise SystemExit("selected candidate checkpoint was pruned or is missing")
    attempt_id = safe_identifier(run_contract.get("attempt_id"), "attempt_id")
    source_hash, changed_paths = project_inventory(project)
    if source_hash != run_contract.get("source_inventory_sha256"):
        raise SystemExit("source changed after the selected run; rerun the candidate before staging")
    parameterization_source_changed = any(
        path.startswith("examples/training/parameterization/runtime/parameterization_")
        or path.startswith("3rdparty/Megatron-LM/megatron/core/parameterization/")
        or path.startswith("3rdparty/Megatron-LM/megatron/core/fusions/")
        or path.startswith("3rdparty/Megatron-LM/megatron/core/extensions/")
        for path in changed_paths
    )
    if not parameterization_source_changed:
        raise SystemExit("staging requires a changed parameterization implementation, not the starter AdamH")
    ladder_attempts = passing_attempts(output_root, source_hash)
    missing = [item for item in SCALE_ORDER if item not in ladder_attempts]
    if missing:
        raise SystemExit(f"same-source ladder is incomplete or invalid; missing scales: {missing}")
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
    novelty_path = output_root / "NOVELTY.md"
    parameterization_path = output_root / "PARAMETERIZATION_SPEC.md"
    if not novelty_path.is_file() or not novelty_path.read_text().strip():
        raise SystemExit("NOVELTY.md is required before staging")
    if not parameterization_path.is_file() or not parameterization_path.read_text().strip():
        raise SystemExit("PARAMETERIZATION_SPEC.md is required before staging")
    scales: dict[str, dict[str, Any]] = {}
    scale_scores: list[float] = []
    loss_log_gains: list[float] = []
    macro_log_gains: list[float] = []
    improved_scales = 0
    improved_loss_auc_scales = 0
    for ladder_scale in SCALE_ORDER:
        ladder_attempt = ladder_attempts[ladder_scale]
        ladder_status = json.loads((ladder_attempt / "status.json").read_text())
        ladder_contract = SCALE_CONTRACTS[ladder_scale]
        ladder_iteration = ladder_contract["iterations"]
        ladder_metrics = read_metrics(ladder_attempt, ladder_iteration, ladder_scale)
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
        evaluation = evaluate_matched_cost_scale(
            ladder_scale, ladder_attempt, baselines[ladder_scale]
        )
        scale_scores.append(float(evaluation["score"]))
        if float(evaluation["score"]) > 0:
            improved_scales += 1
        endpoint = evaluation["matched"][-1]
        macro_log_gains.append(
            math.log(endpoint["baseline_macro_bpb"] / endpoint["candidate_macro_bpb"])
        )
        loss_log_gains.append(
            math.log(evaluation["baseline_loss_auc"] / evaluation["candidate_loss_auc"])
        )
        if evaluation["candidate_loss_auc"] < evaluation["baseline_loss_auc"]:
            improved_loss_auc_scales += 1
        scales[ladder_scale] = {
            "attempt_id": ladder_attempt.name,
            "checkpoint_dir": str(relative_attempt / "checkpoints"),
            "training_eval_results": str(
                relative_attempt / "eval_harness" / f"step_{ladder_iteration:08d}" / "results.json"
            ),
            "cost_trace": str(relative_attempt / "parameterization_cost_trace.jsonl"),
            "metrics": ladder_metrics,
            "parameters": ladder_parameters,
            "guards": ladder_guards,
            "matched_cost_evaluation": evaluation,
        }

    budget_report = research_budget_report(output_root, baselines)
    selected_attempts = {
        scale_name: ladder_attempts[scale_name] for scale_name in SCALE_ORDER
    }
    try:
        ablation_report = validate_mechanism_ablation(
            output_root, selected_attempts, baselines
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid mechanism-on/off evidence: {error}") from error

    reward = math.exp(sum(scale_scores) / len(scale_scores))
    macro_gain = math.exp(sum(macro_log_gains) / len(macro_log_gains))
    loss_auc_gain = math.exp(sum(loss_log_gains) / len(loss_log_gains))
    endpoint_micro_noninferiority = all(
        item["matched"][-1]["candidate_micro_bpb"]
        <= 1.005 * item["matched"][-1]["baseline_micro_bpb"]
        for item in (scales[scale]["matched_cost_evaluation"] for scale in SCALE_ORDER)
    )
    intermediate_micro_noninferiority = all(
        point["candidate_micro_bpb"] <= 1.01 * point["baseline_micro_bpb"]
        for item in (scales[scale]["matched_cost_evaluation"] for scale in SCALE_ORDER)
        for point in item["matched"][:2]
    )
    macro_noninferiority = all(
        item["matched"][-1]["candidate_macro_bpb"]
        <= 1.01 * item["matched"][-1]["baseline_macro_bpb"]
        for item in (scales[scale]["matched_cost_evaluation"] for scale in SCALE_ORDER)
    )
    large_scale_gain = math.exp((scale_scores[3] + scale_scores[4]) / 2.0)
    quality_guards = {
        "aggregate_paloma_gain": reward >= 1.002,
        "improved_scale_count": improved_scales >= 4,
        "large_scale_transfer": large_scale_gain > 1.0,
        "endpoint_micro_noninferiority": endpoint_micro_noninferiority,
        "intermediate_micro_noninferiority": intermediate_micro_noninferiority,
        "aggregate_macro_noninferiority": macro_gain >= 1.0,
        "per_scale_macro_noninferiority": macro_noninferiority,
        "aggregate_loss_auc_improvement": loss_auc_gain > 1.0,
        "improved_loss_auc_scale_count": improved_loss_auc_scales >= 4,
        "mechanism_ablation": bool(ablation_report["passes"]),
        "research_budget": bool(budget_report["compliant"]),
        "parameterization_source_changed": parameterization_source_changed,
    }
    if not all(quality_guards.values()):
        raise SystemExit(f"complete ladder fails aggregate quality gates: {quality_guards}")

    existing_submission_path = output_root / "submission.json"
    if existing_submission_path.is_file():
        incumbent = json.loads(existing_submission_path.read_text())
        if reward <= float(incumbent.get("reward", float("-inf"))):
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
        "reward": reward,
        "aggregate_macro_gain": macro_gain,
        "aggregate_loss_auc_gain": loss_auc_gain,
        "improved_scales": improved_scales,
        "improved_loss_auc_scales": improved_loss_auc_scales,
        "large_scale_gain": large_scale_gain,
        "quality_guards": quality_guards,
        "mechanism_ablation_file": "mechanism_ablation.json",
        "mechanism_ablation": ablation_report,
        "budget_report": budget_report,
        "submission_eligible": True,
    }
    atomic_json(output_root / "COST_REPORT.json", {
        "version": 1,
        "cost_coordinate": "synchronized_active_training_gpu_seconds",
        "selected_scales": {
            scale_name: {
                "attempt_id": scales[scale_name]["attempt_id"],
                "final_gpu_seconds": scales[scale_name]["metrics"]["final_gpu_seconds"],
                "matched_scoring_gpu_seconds": scales[scale_name]["matched_cost_evaluation"]["matched_scoring_gpu_seconds"],
                "updates": SCALE_CONTRACTS[scale_name]["iterations"],
                "tokens": SCALE_CONTRACTS[scale_name]["tokens"],
            }
            for scale_name in SCALE_ORDER
        },
        "research_budget": budget_report,
    })
    atomic_json(output_root / "submission.json", submission)
    print(json.dumps(submission, indent=2, sort_keys=True))


def _finite_positive(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise SystemExit(f"{label} must be numeric") from error
    if not math.isfinite(result) or result <= 0.0:
        raise SystemExit(f"{label} must be finite and positive, got {result}")
    return result


def materialize_transfer_recipe(project: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the candidate recipe and deterministically materialize every known shape."""
    source_path = project / "parameterization_transfer.py"
    if not source_path.is_file() or source_path.is_symlink():
        raise SystemExit(f"candidate recipe is missing or unsafe: {source_path}")
    source = source_path.read_text()
    forbidden_patterns = {
        "exact scale label": r"['\"]E[0-5]['\"]",
        "candidate scale environment": r"\bPARAMETERIZATION_SCALE\b|\bSCALE_KEY\b",
        "training-horizon input": r"\bTARGET_TOKENS\b|\bTRAIN_ITERS\b|\bGLOBAL_BATCH_SIZE\b",
        "target metadata access": r"hidden_target(?:\.json)?",
        "environment-dependent recipe": r"os\.environ|os\.getenv|getenv\(",
    }
    violations = [label for label, pattern in forbidden_patterns.items() if re.search(pattern, source)]
    if violations:
        raise SystemExit(f"recipe violates scale-general boundary: {violations}")

    module_name = f"parameterization_transfer_candidate_{hashlib.sha256(source.encode()).hexdigest()[:16]}"
    module_spec = importlib.util.spec_from_file_location(
        module_name,
        source_path,
    )
    if module_spec is None or module_spec.loader is None:
        raise SystemExit("unable to load candidate recipe")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    if not callable(getattr(module, "build_recipe", None)):
        raise SystemExit("parameterization_transfer.py must define build_recipe()")
    recipe = module.build_recipe()
    for method in ("init_scale", "forward_scale", "lr_scale"):
        if not callable(getattr(recipe, method, None)):
            raise SystemExit(f"recipe must define {method}()")
    base_lr = _finite_positive(getattr(recipe, "base_lr", None), "recipe.base_lr")

    base = SimpleNamespace(layers=12, hidden_size=1152, ffn_hidden_size=4608)
    models: dict[str, SimpleNamespace] = {
        scale: SimpleNamespace(
            layers=contract["layers"],
            hidden_size=contract["hidden"],
            ffn_hidden_size=contract["ffn"],
        )
        for scale, contract in SCALE_CONTRACTS.items()
    }
    hidden = json.loads((project / "hidden_target.json").read_text())["model"]
    models["E5"] = SimpleNamespace(
        layers=int(hidden["layers"]),
        hidden_size=int(hidden["hidden_size"]),
        ffn_hidden_size=int(hidden["ffn_hidden_size"]),
    )
    init_roles = ("embedding", "hidden_matrix", "normalization", "readout")
    lr_roles = ("mha_qkv", "attention_output", "mlp_gate_up", "mlp_output", "readout")
    forward_sites = ("embedding", "attention_output", "mlp_output", "residual", "readout")
    materialized: dict[str, Any] = {}
    for scale, model in models.items():
        role_shapes = {
            "embedding": SimpleNamespace(dimensions=(128256, model.hidden_size), fan_in=model.hidden_size, fan_out=128256),
            "hidden_matrix": SimpleNamespace(dimensions=(model.hidden_size, model.hidden_size), fan_in=model.hidden_size, fan_out=model.hidden_size),
            "normalization": SimpleNamespace(dimensions=(model.hidden_size,), fan_in=model.hidden_size, fan_out=model.hidden_size),
            "readout": SimpleNamespace(dimensions=(128256, model.hidden_size), fan_in=model.hidden_size, fan_out=128256),
            "mha_qkv": SimpleNamespace(dimensions=(3 * model.hidden_size, model.hidden_size), fan_in=model.hidden_size, fan_out=3 * model.hidden_size),
            "attention_output": SimpleNamespace(dimensions=(model.hidden_size, model.hidden_size), fan_in=model.hidden_size, fan_out=model.hidden_size),
            "mlp_gate_up": SimpleNamespace(dimensions=(2 * model.ffn_hidden_size, model.hidden_size), fan_in=model.hidden_size, fan_out=2 * model.ffn_hidden_size),
            "mlp_output": SimpleNamespace(dimensions=(model.hidden_size, model.ffn_hidden_size), fan_in=model.ffn_hidden_size, fan_out=model.hidden_size),
        }
        initializers: dict[str, Any] = {}
        for role in init_roles:
            first = recipe.init_scale(role, role_shapes[role], model, base)
            second = recipe.init_scale(role, role_shapes[role], model, base)
            if first != second:
                raise SystemExit(f"non-deterministic init_scale for {scale}/{role}")
            if isinstance(first, dict):
                normalized = dict(first)
                normalized.setdefault("distribution", "truncated_normal")
                if normalized["distribution"] == "constant":
                    try:
                        normalized["value"] = float(normalized.get("value", 1.0))
                    except (TypeError, ValueError) as error:
                        raise SystemExit(f"constant initializer for {scale}/{role} must be numeric") from error
                    if not math.isfinite(normalized["value"]):
                        raise SystemExit(f"constant initializer for {scale}/{role} must be finite")
                else:
                    if "std" not in normalized:
                        raise SystemExit(f"init_scale for {scale}/{role} must include std")
                    normalized["std"] = _finite_positive(normalized["std"], f"{scale}/{role} init std")
                initializers[role] = normalized
            else:
                initializers[role] = {
                    "distribution": "truncated_normal",
                    "std": _finite_positive(first, f"{scale}/{role} init std"),
                }
        lr_multipliers: dict[str, float] = {}
        for role in lr_roles:
            shape = role_shapes.get(role, role_shapes["readout"])
            first = _finite_positive(recipe.lr_scale(role, shape, model, base), f"{scale}/{role} LR multiplier")
            second = _finite_positive(recipe.lr_scale(role, shape, model, base), f"{scale}/{role} LR multiplier")
            if first != second:
                raise SystemExit(f"non-deterministic lr_scale for {scale}/{role}")
            lr_multipliers[role] = first
        forward_multipliers: dict[str, float] = {}
        for site in forward_sites:
            first = _finite_positive(recipe.forward_scale(site, model, base), f"{scale}/{site} forward multiplier")
            second = _finite_positive(recipe.forward_scale(site, model, base), f"{scale}/{site} forward multiplier")
            if first != second:
                raise SystemExit(f"non-deterministic forward_scale for {scale}/{site}")
            forward_multipliers[site] = first
        materialized[scale] = {
            "model": {
                "layers": model.layers,
                "hidden_size": model.hidden_size,
                "ffn_hidden_size": model.ffn_hidden_size,
            },
            "initializers": initializers,
            "forward_multipliers": forward_multipliers,
            "adamh_lr_multipliers": lr_multipliers,
            "base_lr": base_lr,
            "auxiliary_adam_lr": 0.000304311605183897,
        }
    spec = {
        "version": 1,
        "base_shape": {"layers": 12, "hidden_size": 1152, "ffn_hidden_size": 4608},
        "base_lr": base_lr,
        "proxy_scale": "E0",
        "visible_scales": list(SCALE_ORDER),
        "held_out_scale": "E5",
        "auxiliary_adam_lr": 0.000304311605183897,
        "materialized_recipe": materialized,
    }
    checks = {
        "version": 1,
        "recipe_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "models_materialized": len(materialized),
        "visible_widths_checked": len(SCALE_ORDER),
        "held_out_shape_materialized_without_metrics": True,
        "finite_initializers": True,
        "finite_forward_multipliers": True,
        "finite_adamh_lr_multipliers": True,
        "one_base_lr": True,
        "locked_auxiliary_adam_lr": True,
        "deterministic": True,
        "passes": True,
    }
    return spec, checks


def freeze_parameterization(args: argparse.Namespace) -> None:
    project = args.project.resolve()
    output_root = args.output_root.resolve()
    if (output_root / "submission.json").exists():
        raise SystemExit("submission is already staged and immutable")
    spec, checks = materialize_transfer_recipe(project)
    source_hash, changed_paths = project_inventory(project)
    if "parameterization_transfer.py" not in changed_paths:
        raise SystemExit("the starter recipe is not eligible; edit parameterization_transfer.py")
    atomic_json(output_root / "PARAMETERIZATION_SPEC.json", spec)
    atomic_json(output_root / "materialized_role_report.json", spec["materialized_recipe"])
    atomic_json(output_root / "coordinate_check_report.json", checks)
    atomic_json(output_root / "FREEZE_CANDIDATE.json", {
        "version": 1,
        "created_at": utc_now(),
        "source_inventory_sha256": source_hash,
        "recipe_source_sha256": checks["recipe_source_sha256"],
        "changed_paths": changed_paths,
        "status": "materialized_not_staged",
    })
    print(json.dumps({"source_inventory_sha256": source_hash, "spec": spec, "checks": checks}, indent=2, sort_keys=True))


def stage(args: argparse.Namespace) -> None:
    """Freeze visible evidence exactly once; held-out execution remains trusted-only."""
    attempt = args.attempt.resolve()
    project = args.project.resolve()
    output_root = Path(os.environ.get("PARAMETERIZATION_OUTPUT_ROOT", "/app/output")).resolve()
    require_resolved_below(output_root / "attempts", attempt, "anchor attempt")
    if (output_root / "submission.json").exists():
        raise SystemExit("submission has already been staged; post-freeze edits are forbidden")
    anchor_scale, run_contract = attempt_contract(attempt)
    if anchor_scale == "E0":
        raise SystemExit("stage with a completed larger visible confirmation (E1-E4), not E0")
    anchor_status = json.loads((attempt / "status.json").read_text())
    if anchor_status.get("status") != "completed" or anchor_status.get("run_valid") is not True:
        raise SystemExit(f"anchor {attempt.name} is not a valid completed run")
    if anchor_status.get("checkpoint_iteration") != SCALE_CONTRACTS[anchor_scale]["iterations"]:
        raise SystemExit(f"anchor {attempt.name} is not a full {anchor_scale} run")
    if not (attempt / "checkpoints").is_dir():
        raise SystemExit("anchor checkpoint is missing or pruned")

    source_hash, changed_paths = project_inventory(project)
    if source_hash != run_contract.get("source_inventory_sha256"):
        raise SystemExit("source changed after the anchor run; rerun before staging")
    if "parameterization_transfer.py" not in changed_paths:
        raise SystemExit("staging requires a changed root parameterization recipe")
    allowed_changed = all(
        path == "parameterization_transfer.py"
        or path.startswith("3rdparty/Megatron-LM/megatron/core/parameterization/")
        or path.startswith("3rdparty/Megatron-LM/megatron/core/fusions/")
        or path.startswith("3rdparty/Megatron-LM/megatron/core/extensions/")
        for path in changed_paths
    )
    if not allowed_changed:
        raise SystemExit(f"source changes escape the parameterization boundary: {changed_paths}")

    passing = passing_attempts(output_root, source_hash)
    if "E0" not in passing:
        raise SystemExit("a valid full E0 proxy run from the frozen source is required")
    if anchor_scale not in passing:
        raise SystemExit("the selected larger confirmation is not retained as passing evidence")
    selected_scales = [scale for scale in SCALE_ORDER if scale in passing]
    if not any(scale != "E0" for scale in selected_scales):
        raise SystemExit("at least one larger full visible confirmation is required")

    hypothesis = args.hypothesis_file.resolve()
    if not hypothesis.is_file() or not hypothesis.read_text().strip():
        raise SystemExit("a non-empty HYPOTHESIS.md is required")
    hypothesis_target = output_root / "HYPOTHESIS.md"
    if hypothesis != hypothesis_target:
        hypothesis_target.write_bytes(hypothesis.read_bytes())
    rule_path = output_root / "PARAMETERIZATION_RULE.md"
    if not rule_path.is_file() or not rule_path.read_text().strip():
        raise SystemExit("PARAMETERIZATION_RULE.md is required before staging")
    ledger = output_root / "experiments.jsonl"
    if not ledger.is_file() or not ledger.read_text().strip():
        raise SystemExit("experiments.jsonl is missing or empty")

    spec, coordinate_checks = materialize_transfer_recipe(project)
    budget_report = research_budget_report(output_root, load_baselines(args.baselines))
    if not budget_report["compliant"]:
        raise SystemExit(f"visible research budget is not compliant: {budget_report}")
    source_manifest = snapshot_source(
        project,
        output_root,
        safe_identifier(run_contract.get("attempt_id"), "attempt_id"),
        changed_paths,
        source_hash,
    )
    scales: dict[str, Any] = {}
    for scale in selected_scales:
        selected_attempt = passing[scale]
        selected_status = json.loads((selected_attempt / "status.json").read_text())
        relative = selected_attempt.relative_to(output_root)
        scales[scale] = {
            "attempt_id": selected_attempt.name,
            "checkpoint_dir": str(relative / "checkpoints"),
            "metrics": selected_status.get("metrics", {}),
            "parameters": selected_status.get("parameters"),
            "guards": selected_status.get("guards", {}),
            "training_flops": SCALE_FULL_FLOPS[scale],
            "full_run": True,
        }

    policy_report = {
        "version": 1,
        "candidate_visible_scales": list(SCALE_ORDER),
        "held_out_launcher_absent": True,
        "allowed_source_changes": allowed_changed,
        "single_base_lr": True,
        "locked_auxiliary_adam_lr": True,
        "passes": True,
    }
    reproducibility_report = {
        "version": 1,
        "source_inventory_sha256": source_hash,
        "same_source_visible_evidence": True,
        "selected_scales": selected_scales,
        "proxy_complete": "E0" in selected_scales,
        "larger_confirmation_complete": any(scale != "E0" for scale in selected_scales),
        "passes": True,
    }
    leakage_report = {
        "version": 1,
        "target_metrics_available_to_candidate": False,
        "target_launcher_available_to_candidate": False,
        "target_oracle_available_to_candidate": False,
        "sanitized_target_shape_only": True,
        "recipe_static_boundary_passes": True,
        "passes": True,
    }
    atomic_json(output_root / "PARAMETERIZATION_SPEC.json", spec)
    atomic_json(output_root / "materialized_role_report.json", spec["materialized_recipe"])
    atomic_json(output_root / "coordinate_check_report.json", coordinate_checks)
    atomic_json(output_root / "POLICY_REPORT.json", policy_report)
    atomic_json(output_root / "REPRODUCIBILITY_REPORT.json", reproducibility_report)
    atomic_json(output_root / "LEAKAGE_REPORT.json", leakage_report)
    atomic_json(output_root / "COST_REPORT.json", budget_report)
    submission = {
        "version": 3,
        "submission_type": "frozen_visible_parameterization",
        "attempt_id": attempt.name,
        "anchor_scale": anchor_scale,
        "selected_at": utc_now(),
        "source_inventory_sha256": source_hash,
        "source_snapshot": str(source_manifest.relative_to(output_root)),
        "recipe_source_sha256": coordinate_checks["recipe_source_sha256"],
        "changed_paths": changed_paths,
        "base_lr": spec["base_lr"],
        "visible_scale_order": list(SCALE_ORDER),
        "selected_visible_scales": selected_scales,
        "scales": scales,
        "budget_report": budget_report,
        "quality_level": "valid_visible_freeze",
        "held_out_evaluation": {"scale": "E5", "status": "pending_trusted_evaluator"},
        "candidate_claimed_reward": None,
        "submission_eligible": True,
    }
    atomic_json(output_root / "submission.json", submission)
    atomic_json(output_root / "FROZEN.json", {
        "version": 1,
        "frozen_at": submission["selected_at"],
        "source_inventory_sha256": source_hash,
        "submission_sha256": hashlib.sha256((output_root / "submission.json").read_bytes()).hexdigest(),
    })
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
    output_root = Path(os.environ.get("PARAMETERIZATION_OUTPUT_ROOT", "/app/output")).resolve()
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
    prepare_parser.add_argument("--resume-iteration", type=int, default=0)
    prepare_parser.add_argument("--resume-attempt-id")
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
    cost_parser = subparsers.add_parser("cost-report")
    cost_parser.add_argument("--attempt", type=Path, required=True)
    cost_parser.add_argument("--baselines", type=Path)
    cost_parser.set_defaults(handler=cost_report)
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
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--project", type=Path, default=Path("/app/project"))
    freeze_parser.add_argument("--output-root", type=Path, default=Path("/app/output"))
    freeze_parser.set_defaults(handler=freeze_parameterization)
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
