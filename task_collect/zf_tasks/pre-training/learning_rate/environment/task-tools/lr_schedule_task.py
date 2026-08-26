#!/usr/bin/env python3
"""Prepare, gate, summarize, and stage learning-rate ladder attempts."""

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
DEVELOPMENT_SCALES = ("E0", "E2")
CONFIRMATION_SCALES = tuple(
    scale for scale in SCALE_ORDER if scale not in DEVELOPMENT_SCALES
)
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
HYPOTHESIS_SECTIONS = (
    "mechanism",
    "exact schedule",
    "predicted trajectory",
    "rejection criterion",
    "discriminating ablation",
    "planned experiments",
)
TUNING_ONLY_PATTERNS = (
    "grid search",
    "bayesian optimization",
    "choose the best parameter",
    "best official score",
    "neighboring parameter",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_hypothesis(path: Path) -> dict[str, str]:
    """Require a falsifiable mechanism proposal rather than a tuning recipe."""

    if not path.is_file():
        raise ValueError(f"hypothesis file is missing: {path}")
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in path.read_text().splitlines():
        match = re.fullmatch(r"##\s+(.+?)\s*", line)
        if match:
            current = match.group(1).strip().lower()
            sections.setdefault(current, [])
        elif current is not None:
            sections[current].append(line)
    missing = [name for name in HYPOTHESIS_SECTIONS if name not in sections]
    if missing:
        raise ValueError(f"hypothesis is missing required sections: {missing}")
    normalized = {
        name: "\n".join(sections[name]).strip() for name in HYPOTHESIS_SECTIONS
    }
    empty = [name for name, content in normalized.items() if len(content) < 20]
    if empty:
        raise ValueError(f"hypothesis sections need substantive content: {empty}")
    lowered = "\n".join(normalized.values()).lower()
    if any(pattern in lowered for pattern in TUNING_ONLY_PATTERNS):
        raise ValueError(
            "tuning-only hypothesis: state a mechanism, prediction, and removal ablation"
        )
    return normalized


def bind_confirmation_source(
    output_root: Path,
    *,
    scale: str,
    source_hash: str,
    hypothesis_hash: str,
) -> dict[str, Any] | None:
    """Freeze the science claim before exposing any confirmation-scale result."""

    if scale in DEVELOPMENT_SCALES:
        return None
    if scale not in CONFIRMATION_SCALES:
        raise ValueError(f"unknown scale: {scale}")
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".science-freeze.lock"
    freeze_path = output_root / "science_freeze.json"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if freeze_path.is_file():
            payload = json.loads(freeze_path.read_text())
            if (
                payload.get("source_inventory_sha256") != source_hash
                or payload.get("hypothesis_sha256") != hypothesis_hash
            ):
                raise ValueError(
                    "confirmation source and hypothesis are already frozen"
                )
            return payload
        payload = {
            "version": 1,
            "development_scales": list(DEVELOPMENT_SCALES),
            "confirmation_scales": list(CONFIRMATION_SCALES),
            "frozen_before_scale": scale,
            "source_inventory_sha256": source_hash,
            "hypothesis_sha256": hypothesis_hash,
            "created_at": utc_now(),
        }
        atomic_json(freeze_path, payload)
        return payload


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


def candidate_source_changed(changed_paths: list[str]) -> bool:
    """Return whether the one participant-owned schedule module changed."""

    return "examples/training/lr_schedule/runtime/lr_schedule_candidate.py" in changed_paths


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


def _finite_positive(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def validate_baseline_manifest(
    payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Validate the release-time exact-final WSD evidence contract."""

    if payload.get("version") != 3:
        raise ValueError("learning-rate baseline manifest must have version 3")
    if payload.get("evidence_scope") != "locked_wsd_exact_final":
        raise ValueError("baseline evidence_scope must be locked_wsd_exact_final")
    if payload.get("cost_definition") != "active_training_gpu_seconds_excluding_evaluation":
        raise ValueError("baseline cost definition is not the locked active-training cost")
    scales = payload.get("scales")
    if not isinstance(scales, dict) or set(scales) != set(SCALE_ORDER):
        raise ValueError("baseline manifest must contain exactly E0-E5")

    for scale in SCALE_ORDER:
        raw = scales[scale]
        contract = SCALE_CONTRACTS[scale]
        if not isinstance(raw, dict):
            raise ValueError(f"{scale} baseline record must be an object")
        if int(raw.get("final_update", -1)) != contract["iterations"]:
            raise ValueError(f"{scale} final update differs from the locked horizon")
        if int(raw.get("scoring_update", -1)) != contract["iterations"]:
            raise ValueError(f"{scale} baseline is not exact-final")
        if raw.get("reference_status") != "exact_final":
            raise ValueError(f"{scale} baseline must carry exact-final status")
        for key, expected in (
            ("parameters", contract["parameters"]),
            ("tokens", contract["tokens"]),
            ("global_batch_size", contract["gbs"]),
            ("gpus", contract["gpus"]),
        ):
            if int(raw.get(key, -1)) != expected:
                raise ValueError(f"{scale} {key} differs from the locked contract")
        _finite_positive(raw.get("paloma_bits_per_byte"), f"{scale} final micro BPB")
        _finite_positive(
            raw.get("paloma_macro_bits_per_byte"), f"{scale} final macro BPB"
        )
        _finite_positive(raw.get("scoring_gpu_seconds"), f"{scale} final cost")
        _finite_positive(
            raw.get("scoring_fixed_window_loss"), f"{scale} final fixed-window loss"
        )

        trajectory = raw.get("paloma_trajectory")
        if not isinstance(trajectory, list) or len(trajectory) < 2:
            raise ValueError(f"{scale} must contain at least two Paloma evaluations")
        previous_iteration = 0
        previous_cost = 0.0
        for point in trajectory:
            iteration = int(point.get("iteration", -1))
            cost = _finite_positive(
                point.get("cumulative_gpu_seconds"), f"{scale} Paloma cost"
            )
            _finite_positive(point.get("bits_per_byte"), f"{scale} Paloma micro BPB")
            _finite_positive(
                point.get("macro_bits_per_byte"), f"{scale} Paloma macro BPB"
            )
            if iteration <= previous_iteration or cost <= previous_cost:
                raise ValueError(f"{scale} Paloma trajectory is not strictly ordered")
            previous_iteration, previous_cost = iteration, cost
        if previous_iteration != contract["iterations"]:
            raise ValueError(f"{scale} Paloma trajectory lacks its exact-final point")
        final_point = trajectory[-1]
        if not math.isclose(
            float(final_point["bits_per_byte"]),
            float(raw["paloma_bits_per_byte"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            float(final_point["macro_bits_per_byte"]),
            float(raw["paloma_macro_bits_per_byte"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{scale} final Paloma summary differs from its trajectory")

        curve = raw.get("loss_cost_curve")
        if not isinstance(curve, list) or len(curve) < 2:
            raise ValueError(f"{scale} loss/cost curve is incomplete")
        curve_iterations = [int(point.get("iteration", -1)) for point in curve]
        curve_costs = [
            _finite_positive(point.get("cumulative_gpu_seconds"), f"{scale} loss cost")
            for point in curve
        ]
        for point in curve:
            _finite_positive(point.get("fixed_window_loss"), f"{scale} fixed-window loss")
        if any(right <= left for left, right in zip(curve_iterations, curve_iterations[1:])):
            raise ValueError(f"{scale} loss iterations are not strictly ordered")
        if any(right <= left for left, right in zip(curve_costs, curve_costs[1:])):
            raise ValueError(f"{scale} loss costs are not strictly ordered")
        if curve_iterations[-1] != contract["iterations"]:
            raise ValueError(f"{scale} loss/cost curve lacks its exact-final point")

        checkpoint = raw.get("checkpoint")
        if not isinstance(checkpoint, dict) or int(checkpoint.get("iteration", -1)) != contract["iterations"]:
            raise ValueError(f"{scale} checkpoint provenance is not exact-final")
        if len(str(checkpoint.get("path_sha256", ""))) != 64:
            raise ValueError(f"{scale} checkpoint provenance lacks a SHA-256")
        if len(str(raw.get("source_inventory_sha256", ""))) != 64:
            raise ValueError(f"{scale} source provenance lacks a SHA-256")
        if not raw.get("wandb_project") or not raw.get("wandb_run_id"):
            raise ValueError(f"{scale} offline run provenance is missing")
        run_contract = raw.get("run_contract")
        expected_run_contract = {
            "seed": 0,
            "sequence_length": 4096,
            "tensor_parallel": 1,
            "pipeline_parallel": 1,
            "context_parallel": 1,
        }
        if not isinstance(run_contract, dict) or any(
            run_contract.get(key) != value for key, value in expected_run_contract.items()
        ):
            raise ValueError(f"{scale} run-contract provenance is incomplete")
    return scales


def load_baselines(path: Path | None = None) -> dict[str, dict[str, Any]]:
    baseline_path = path or Path(
        os.environ.get("LR_SCHEDULE_BASELINES", "/task-data/lr_schedule_baselines.json")
    )
    payload = json.loads(baseline_path.read_text())
    return validate_baseline_manifest(payload)


def _trajectory_area(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]], final_update: int
) -> tuple[float, list[dict[str, float]]]:
    if len(baseline) != len(candidate) or len(baseline) < 2:
        raise ValueError("candidate trajectory does not match the baseline update set")
    points: list[dict[str, float]] = []
    for reference, measured in zip(baseline, candidate, strict=True):
        iteration = int(reference["iteration"])
        if int(measured.get("iteration", -1)) != iteration:
            raise ValueError("candidate trajectory updates do not match the baseline")
        baseline_micro = _finite_positive(
            reference["bits_per_byte"], "baseline trajectory micro BPB"
        )
        candidate_micro = _finite_positive(
            measured.get("bits_per_byte"), "candidate trajectory micro BPB"
        )
        baseline_macro = _finite_positive(
            reference["macro_bits_per_byte"], "baseline trajectory macro BPB"
        )
        candidate_macro = _finite_positive(
            measured.get("macro_bits_per_byte"), "candidate trajectory macro BPB"
        )
        points.append(
            {
                "iteration": float(iteration),
                "progress": float(iteration) / float(final_update),
                "baseline_micro_bpb": baseline_micro,
                "candidate_micro_bpb": candidate_micro,
                "baseline_macro_bpb": baseline_macro,
                "candidate_macro_bpb": candidate_macro,
                "log_micro_gain": math.log(baseline_micro / candidate_micro),
            }
        )
    width = points[-1]["progress"] - points[0]["progress"]
    if width <= 0.0:
        raise ValueError("trajectory has no normalized update interval")
    integral = sum(
        0.5
        * (right["progress"] - left["progress"])
        * (left["log_micro_gain"] + right["log_micro_gain"])
        for left, right in zip(points, points[1:])
    )
    return integral / width, points


def score_ladder(
    baselines: dict[str, dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Apply the public trajectory/final gates and compute the trusted reward."""

    if set(baselines) != set(SCALE_ORDER) or set(candidates) != set(SCALE_ORDER):
        raise ValueError("scoring requires exactly E0-E5")
    scale_results: dict[str, dict[str, Any]] = {}
    areas: list[float] = []
    final_micro_logs: list[float] = []
    final_macro_logs: list[float] = []
    source_hashes: set[str] = set()
    improved_scales = 0
    per_scale_final = True
    trajectory_positive = True
    intermediate_micro = True
    matched_macro = True
    cost_compliant = True
    exact_contract = True

    for scale in SCALE_ORDER:
        reference = baselines[scale]
        candidate = candidates[scale]
        contract = SCALE_CONTRACTS[scale]
        if int(candidate.get("final_update", -1)) != contract["iterations"]:
            exact_contract = False
        source_hash = str(candidate.get("source_inventory_sha256", ""))
        if len(source_hash) != 64:
            exact_contract = False
        source_hashes.add(source_hash)
        area, points = _trajectory_area(
            reference["paloma_trajectory"],
            candidate.get("paloma_trajectory", []),
            contract["iterations"],
        )
        areas.append(area)
        final = points[-1]
        final_micro_gain = final["baseline_micro_bpb"] / final["candidate_micro_bpb"]
        final_macro_gain = final["baseline_macro_bpb"] / final["candidate_macro_bpb"]
        final_micro_logs.append(math.log(final_micro_gain))
        final_macro_logs.append(math.log(final_macro_gain))
        per_scale_final &= final["candidate_micro_bpb"] < final["baseline_micro_bpb"]
        trajectory_positive &= area > 0.0
        intermediate_micro &= all(
            point["candidate_micro_bpb"] <= point["baseline_micro_bpb"] + 0.005
            for point in points[:-1]
        )
        matched_macro &= all(
            point["candidate_macro_bpb"] <= 1.005 * point["baseline_macro_bpb"]
            for point in points
        )
        absolute_final_improvement = (
            final["baseline_micro_bpb"] - final["candidate_micro_bpb"]
        )
        if absolute_final_improvement >= 0.002:
            improved_scales += 1
        candidate_cost = _finite_positive(
            candidate.get("scoring_gpu_seconds"), f"{scale} candidate cost"
        )
        baseline_cost = _finite_positive(
            reference["scoring_gpu_seconds"], f"{scale} baseline cost"
        )
        cost_ratio = candidate_cost / baseline_cost
        cost_compliant &= cost_ratio <= 1.01
        scale_results[scale] = {
            "trajectory_area": area,
            "final_micro_gain": final_micro_gain,
            "final_macro_gain": final_macro_gain,
            "absolute_final_micro_improvement": absolute_final_improvement,
            "cost_ratio": cost_ratio,
            "matched": points,
        }

    final_micro_gain = math.exp(sum(final_micro_logs) / len(final_micro_logs))
    final_macro_gain = math.exp(sum(final_macro_logs) / len(final_macro_logs))
    trajectory_gain = math.exp(sum(areas) / len(areas))
    guards = {
        "exact_contract": exact_contract and len(source_hashes) == 1,
        "per_scale_final_micro": per_scale_final,
        "trajectory_positive": trajectory_positive,
        "intermediate_micro": intermediate_micro,
        "aggregate_final_micro": final_micro_gain >= 1.002,
        "minimum_improved_scales": improved_scales >= 4,
        "matched_macro": matched_macro,
        "aggregate_final_macro": final_macro_gain >= 1.0,
        "cost": cost_compliant,
    }
    passes = all(guards.values())
    reward = math.sqrt(final_micro_gain * trajectory_gain) if passes else 0.0
    return {
        "passes": passes,
        "reward": reward,
        "final_micro_gain": final_micro_gain,
        "final_macro_gain": final_macro_gain,
        "trajectory_gain": trajectory_gain,
        "improved_scales": improved_scales,
        "guards": guards,
        "scales": scale_results,
    }


def charged_attempt_cost(attempt: Path) -> tuple[float, int, str | None]:
    """Recover charged work even when an attempt failed before finish()."""
    trace_path = attempt / "optimizer_cost_trace.jsonl"
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
    """Charge every candidate attempt to its rung-specific 3C research budget."""
    attempts_root = output_root / "attempts"
    by_scale: dict[str, dict[str, Any]] = {
        scale: {
            "baseline_scoring_gpu_seconds": float(baselines[scale]["scoring_gpu_seconds"]),
            "budget_gpu_seconds": 6.0 * float(baselines[scale]["scoring_gpu_seconds"]),
            "charged_gpu_seconds": 0.0,
            "remaining_gpu_seconds": 0.0,
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
                    "trace_error": trace_error,
                }
                by_scale[scale]["attempts"].append(entry)
                by_scale[scale]["charged_gpu_seconds"] += cost
                if trace_error is not None:
                    by_scale[scale]["compliant"] = False
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                invalid_attempts.append({"attempt": attempt.name, "error": str(error)})
    for scale in SCALE_ORDER:
        report = by_scale[scale]
        report["remaining_gpu_seconds"] = (
            report["budget_gpu_seconds"] - report["charged_gpu_seconds"]
        )
        tolerance = max(1e-6, 1e-9 * report["budget_gpu_seconds"])
        report["compliant"] = bool(report["compliant"]) and (
            report["charged_gpu_seconds"] <= report["budget_gpu_seconds"] + tolerance
        )
    return {
        "version": 1,
        "budget_multiplier": 6.0,
        "scales": by_scale,
        "invalid_attempts": invalid_attempts,
        "running_attempts": running_attempts,
        "compliant": (
            not invalid_attempts
            and not running_attempts
            and all(by_scale[scale]["compliant"] for scale in SCALE_ORDER)
        ),
    }


def read_cost_trace(attempt: Path) -> list[dict[str, Any]]:
    trace_path = attempt / "optimizer_cost_trace.jsonl"
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
            raise ValueError("optimizer cost trace is non-contiguous or non-monotonic")
        if loss is None or not math.isfinite(float(loss)) or float(loss) <= 0:
            raise ValueError(f"invalid training loss at iteration {iteration}")
        if int(row.get("skipped", 0) or 0) != 0:
            raise ValueError(f"skipped optimizer update at iteration {iteration}")
        rows.append(row)
        previous_iteration = iteration
        previous_cost = cost
    if not rows:
        raise ValueError("optimizer cost trace is empty")
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
        schedule_path = attempt / "lr_schedule_trace.jsonl"
        schedule_rows = [
            json.loads(line)
            for line in schedule_path.read_text().splitlines()
            if line.strip()
        ]
        launch_hash = json.loads((attempt / "run_contract.json").read_text())[
            "source_inventory_sha256"
        ]
        metrics.update(
            {
                "final_gpu_seconds": float(trace[-1]["cumulative_gpu_seconds"]),
                "final_fixed_window_loss": float(loss_curve[-1]["fixed_window_loss"]),
                "cost_trace_rows": len(trace),
                "schedule_trace_rows": len(schedule_rows),
                "schedule_source_consistent": bool(schedule_rows)
                and all(row.get("source_sha256") == launch_hash for row in schedule_rows),
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
        "complete_schedule_trace": int(metrics.get("schedule_trace_rows", 0))
        == SCALE_CONTRACTS[scale]["iterations"],
        "schedule_source_consistent": metrics.get("schedule_source_consistent") is True,
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
        os.environ.get("LR_SCHEDULE_RUN_CONTRACT_ROOT", "/run-contract")
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
    output_root = Path(os.environ.get("LR_SCHEDULE_OUTPUT_ROOT", "/app/output")).resolve()
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
    try:
        validate_hypothesis(args.hypothesis_file)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    hypothesis_sha256 = source_sha256(args.hypothesis_file)
    source_hash, changed_paths = project_inventory(args.project.resolve())
    try:
        bind_confirmation_source(
            output_root,
            scale=scale,
            source_hash=source_hash,
            hypothesis_hash=hypothesis_sha256,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
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
        "hypothesis_sha256": hypothesis_sha256,
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


def print_validated_hypothesis(args: argparse.Namespace) -> None:
    print(json.dumps(validate_hypothesis(args.hypothesis_file), sort_keys=True))


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
                    "target_training_tokens": scale_values["tokens"],
                    "optimizer_updates": charged_updates,
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
    output_root = Path(os.environ.get("LR_SCHEDULE_OUTPUT_ROOT", "/app/output")).resolve()
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
    output_root = Path(os.environ.get("LR_SCHEDULE_OUTPUT_ROOT", "/app/output")).resolve()
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
    output_root = Path(os.environ.get("LR_SCHEDULE_OUTPUT_ROOT", "/app/output")).resolve()
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
    anchors = DEVELOPMENT_SCALES
    if payload.get("version") != 1 or set(payload.get("anchors", {})) != set(anchors):
        raise ValueError("mechanism_ablation.json must contain exactly E0 and E2")
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


def stage(args: argparse.Namespace) -> None:
    attempt = args.attempt.resolve()
    project = args.project.resolve()
    output_root = Path(os.environ.get("LR_SCHEDULE_OUTPUT_ROOT", "/app/output")).resolve()
    if output_root not in attempt.parents:
        raise SystemExit(f"attempt must be below {output_root}: {attempt}")
    scale, run_contract = attempt_contract(attempt)
    if scale != SCALE_ORDER[-1]:
        raise SystemExit("only a completed E5 attempt may anchor a staged ladder")
    final_iteration = SCALE_CONTRACTS[scale]["iterations"]
    status = json.loads((attempt / "status.json").read_text())
    if status.get("status") != "completed" or status.get("checkpoint_iteration") != final_iteration:
        raise SystemExit(f"only a completed {scale} update-{final_iteration} attempt may be staged")
    baselines = load_baselines(args.baselines)
    if not status.get("run_valid"):
        raise SystemExit(f"E5 run integrity failed: {status.get('guards', {})}")
    if not (attempt / "checkpoints").is_dir():
        raise SystemExit("selected candidate checkpoint was pruned or is missing")
    attempt_id = safe_identifier(run_contract.get("attempt_id"), "attempt_id")
    source_hash, changed_paths = project_inventory(project)
    if source_hash != run_contract.get("source_inventory_sha256"):
        raise SystemExit("source changed after the selected run; rerun the candidate before staging")
    lr_source_changed = candidate_source_changed(changed_paths)
    if not lr_source_changed:
        raise SystemExit("staging requires a changed lr_schedule_candidate.py")
    ladder_attempts = passing_attempts(output_root, source_hash)
    missing = [item for item in SCALE_ORDER if item not in ladder_attempts]
    if missing:
        raise SystemExit(f"same-source ladder is incomplete or invalid; missing scales: {missing}")
    try:
        validate_hypothesis(args.hypothesis_file)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    hypothesis_sha256 = source_sha256(args.hypothesis_file)
    freeze_path = output_root / "science_freeze.json"
    if not freeze_path.is_file():
        raise SystemExit("confirmation source was never frozen")
    confirmation_freeze = json.loads(freeze_path.read_text())
    if (
        confirmation_freeze.get("source_inventory_sha256") != source_hash
        or confirmation_freeze.get("hypothesis_sha256") != hypothesis_sha256
    ):
        raise SystemExit("staged source or hypothesis differs from the confirmation freeze")
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
    schedule_path = output_root / "LR_SCHEDULE.md"
    if not schedule_path.is_file() or not schedule_path.read_text().strip():
        raise SystemExit("LR_SCHEDULE.md is required before staging")
    scales: dict[str, dict[str, Any]] = {}
    candidate_scales: dict[str, dict[str, Any]] = {}
    for ladder_scale in SCALE_ORDER:
        ladder_attempt = ladder_attempts[ladder_scale]
        _, ladder_run_contract = attempt_contract(ladder_attempt)
        if ladder_run_contract.get("hypothesis_sha256") != hypothesis_sha256:
            raise SystemExit(
                f"retained {ladder_scale} was not run under the staged hypothesis"
            )
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
        trace = read_scoring_cost_trace(ladder_attempt)
        trajectory = read_scoring_paloma_trajectory(ladder_attempt, trace)
        expected_updates = [
            int(point["iteration"])
            for point in baselines[ladder_scale]["paloma_trajectory"]
        ]
        by_iteration = {int(point["iteration"]): point for point in trajectory}
        if set(by_iteration) != set(expected_updates):
            raise SystemExit(
                f"{ladder_scale} candidate Paloma updates differ from the baseline: "
                f"{sorted(by_iteration)} != {expected_updates}"
            )
        schedule_trace = ladder_attempt / "lr_schedule_trace.jsonl"
        if not schedule_trace.is_file():
            raise SystemExit(f"retained {ladder_scale} schedule trace is missing")
        schedule_rows = [
            json.loads(line)
            for line in schedule_trace.read_text().splitlines()
            if line.strip()
        ]
        if (
            len(schedule_rows) != ladder_iteration
            or int(schedule_rows[-1].get("iteration", -1)) != ladder_iteration
            or any(row.get("source_sha256") != source_hash for row in schedule_rows)
        ):
            raise SystemExit(f"retained {ladder_scale} schedule trace is incomplete")
        candidate_scales[ladder_scale] = {
            "final_update": ladder_iteration,
            "scoring_gpu_seconds": float(trace[-1]["cumulative_gpu_seconds"]),
            "source_inventory_sha256": source_hash,
            "paloma_trajectory": [
                {
                    "iteration": iteration,
                    "bits_per_byte": float(by_iteration[iteration]["paloma_bits_per_byte"]),
                    "macro_bits_per_byte": float(
                        by_iteration[iteration]["paloma_macro_bits_per_byte"]
                    ),
                }
                for iteration in expected_updates
            ],
        }
        scales[ladder_scale] = {
            "attempt_id": ladder_attempt.name,
            "checkpoint_dir": str(relative_attempt / "checkpoints"),
            "training_eval_results": str(
                relative_attempt / "eval_harness" / f"step_{ladder_iteration:08d}" / "results.json"
            ),
            "cost_trace": str(relative_attempt / "optimizer_cost_trace.jsonl"),
            "schedule_trace": str(relative_attempt / "lr_schedule_trace.jsonl"),
            "metrics": ladder_metrics,
            "parameters": ladder_parameters,
            "guards": ladder_guards,
        }

    try:
        mechanism_ablation = validate_mechanism_ablation(
            output_root, ladder_attempts, baselines
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid mechanism ablation evidence: {error}") from error

    budget_report = research_budget_report(output_root, baselines)
    scoring = score_ladder(baselines, candidate_scales)
    reward = float(scoring["reward"])
    quality_guards = {
        **scoring["guards"],
        "research_budget": bool(budget_report["compliant"]),
        "lr_source_changed": lr_source_changed,
        "mechanism_ablation": bool(mechanism_ablation["passes"]),
    }
    if not all(quality_guards.values()):
        raise SystemExit(f"complete ladder fails aggregate quality gates: {quality_guards}")

    existing_submission_path = output_root / "submission.json"
    if existing_submission_path.is_file():
        incumbent = json.loads(existing_submission_path.read_text())
        if reward <= float(incumbent.get("reward", float("-inf"))):
            raise SystemExit("complete ladder passes but does not improve the staged incumbent")

    submission = {
        "version": 3,
        "attempt_id": attempt_id,
        "selected_at": utc_now(),
        "anchor_scale": scale,
        "hypothesis_file": "HYPOTHESIS.md",
        "hypothesis_sha256": hypothesis_sha256,
        "mechanism_ablation": mechanism_ablation,
        "confirmation_freeze": confirmation_freeze,
        "source_inventory_sha256": source_hash,
        "source_snapshot": str(source_manifest.relative_to(output_root)),
        "changed_paths": changed_paths,
        "scale_order": list(SCALE_ORDER),
        "scales": scales,
        "reward": reward,
        "scoring": scoring,
        "quality_guards": quality_guards,
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
                "baseline_gpu_seconds": baselines[scale_name]["scoring_gpu_seconds"],
                "cost_ratio": scoring["scales"][scale_name]["cost_ratio"],
                "updates": SCALE_CONTRACTS[scale_name]["iterations"],
                "tokens": SCALE_CONTRACTS[scale_name]["tokens"],
            }
            for scale_name in SCALE_ORDER
        },
        "research_budget": budget_report,
    })
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
    output_root = Path(os.environ.get("LR_SCHEDULE_OUTPUT_ROOT", "/app/output")).resolve()
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
    hypothesis_parser = subparsers.add_parser("validate-hypothesis")
    hypothesis_parser.add_argument("--hypothesis-file", type=Path, required=True)
    hypothesis_parser.set_defaults(handler=print_validated_hypothesis)
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
