#!/usr/bin/env python3
"""Build one read-only RSI-Harness Blue Vela authorization envelope."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tomllib
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from validate_task import Validator, _read_run_options


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _load_harness(harness_root: Path) -> dict[str, Any]:
    root = harness_root.resolve()
    source = root / "src"
    if not (source / "rsi_harness/task/compiler.py").is_file():
        raise ValueError(f"RSI-Harness source not found under {root}")
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

    from rsi_harness.cluster.bluevela.resources import derive_resource_plan
    from rsi_harness.cluster.config import load_cluster_profile
    from rsi_harness.models import CompileOptions
    from rsi_harness.task.compiler import HarborTaskCompiler

    return {
        "root": root,
        "source": source,
        "CompileOptions": CompileOptions,
        "HarborTaskCompiler": HarborTaskCompiler,
        "derive_resource_plan": derive_resource_plan,
        "load_cluster_profile": load_cluster_profile,
    }


def _profile_path(harness_root: Path, cluster: str | Path) -> Path:
    candidate = Path(cluster).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if str(cluster) == "bluevela":
        return (
            harness_root
            / "src/rsi_harness/cluster/bluevela/profile.toml"
        ).resolve()
    raise ValueError(f"unknown Blue Vela profile: {cluster}")


def _covered(path: PurePosixPath, target: PurePosixPath) -> bool:
    return path == target or target in path.parents


def _validate_asset_binds(definition: Any, profile: Any) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    for asset in definition.assets:
        binds = (
            profile.apptainer.work_binds
            if asset.phase == "work"
            else profile.apptainer.judge_binds
        )
        if not any(_covered(asset.path, item.target) for item in binds):
            missing.append({"phase": asset.phase, "path": str(asset.path)})
    return missing


def _scheduler_policy(
    profile: Any,
    *,
    policy: str,
    allocation_evidence: str | None,
) -> dict[str, object]:
    queue = profile.scheduler.queue
    exclusive = profile.scheduler.exclusive
    if policy == "default":
        if queue != "normal" or not exclusive:
            raise ValueError(
                "default scheduler policy requires the normal queue and an "
                "exclusive profile"
            )
        if allocation_evidence:
            raise ValueError(
                "allocation evidence is valid only for allocation-fallback"
            )
        evidence = None
    elif policy == "allocation-fallback":
        if queue != "priority" or exclusive:
            raise ValueError(
                "allocation-fallback requires a run-private priority queue "
                "profile with exclusive = false"
            )
        raw_evidence = (allocation_evidence or "").strip()
        if not raw_evidence:
            raise ValueError(
                "allocation-fallback requires observed allocation evidence"
            )
        evidence = _parse_allocation_evidence(raw_evidence)
    else:
        raise ValueError(f"unknown scheduler policy: {policy}")
    return {
        "name": policy,
        "allocation_evidence": evidence,
        "full_task_resources_required": True,
        "reduced_resources_require_explicit_user_instruction": True,
        "completed_run_is_final": True,
        "normal_rerun_required": False,
        "shared_profile_mutation_forbidden": True,
        "cancel_normal_before_fallback": policy == "allocation-fallback",
        "allocation_evidence_requirements": (
            "same full-size normal job in two timestamped observations "
            "separated by the declared wait window, or an allocation-specific "
            "terminal scheduler reason"
            if policy == "allocation-fallback"
            else None
        ),
    }


def _parse_observed_at(value: object) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("allocation evidence observed_at must be an ISO timestamp")
    normalized = value.strip().replace("Z", "+00:00")
    try:
        observed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(
            "allocation evidence observed_at must be an ISO timestamp"
        ) from error
    if observed.tzinfo is None:
        raise ValueError("allocation evidence timestamps must include a timezone")
    return observed


def _parse_allocation_evidence(raw: str) -> dict[str, object]:
    try:
        evidence = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("allocation evidence must be a JSON object") from error
    if not isinstance(evidence, dict):
        raise ValueError("allocation evidence must be a JSON object")
    job_id = evidence.get("normal_job_id")
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("allocation evidence requires normal_job_id")

    observations = evidence.get("observations")
    terminal = evidence.get("terminal")
    if isinstance(observations, list):
        wait_seconds = evidence.get("declared_wait_seconds")
        if (
            isinstance(wait_seconds, bool)
            or not isinstance(wait_seconds, (int, float))
            or wait_seconds <= 0
        ):
            raise ValueError(
                "pending allocation evidence requires positive "
                "declared_wait_seconds"
            )
        if len(observations) < 2:
            raise ValueError(
                "pending allocation evidence requires at least two observations"
            )
        timestamps: list[datetime] = []
        for item in observations:
            if not isinstance(item, dict):
                raise ValueError("allocation observations must be JSON objects")
            if str(item.get("status", "")).upper() != "PEND":
                raise ValueError("allocation observations must have status PEND")
            if not str(item.get("reason", "")).strip():
                raise ValueError("allocation observations require scheduler reasons")
            timestamps.append(_parse_observed_at(item.get("observed_at")))
        if timestamps != sorted(timestamps) or (
            timestamps[-1] - timestamps[0]
        ).total_seconds() < float(wait_seconds):
            raise ValueError(
                "allocation observations must span the declared wait window"
            )
    elif isinstance(terminal, dict):
        if terminal.get("allocation_specific") is not True:
            raise ValueError(
                "terminal allocation evidence must be allocation-specific"
            )
        status = str(terminal.get("status", "")).upper()
        if not status or status == "PEND":
            raise ValueError(
                "terminal allocation evidence requires terminal status"
            )
        if not str(terminal.get("reason", "")).strip():
            raise ValueError(
                "terminal allocation evidence requires scheduler reason"
            )
        _parse_observed_at(terminal.get("observed_at"))
    else:
        raise ValueError(
            "allocation evidence requires observations or terminal outcome"
        )
    return evidence


def _harness_contract_sha256(harness_root: Path) -> str:
    roots = (
        harness_root / "src/rsi_harness/task",
        harness_root / "src/rsi_harness/cluster/bluevela",
    )
    digest = hashlib.sha256()
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"missing Harness contract directory: {root}")
        subtree = _tree_sha256(root).encode()
        digest.update(root.name.encode())
        digest.update(subtree)
    for path in (
        harness_root / "src/rsi_harness/cluster/config.py",
        harness_root / "src/rsi_harness/cluster/schedulers/lsf.py",
        harness_root / "src/rsi_harness/models.py",
    ):
        digest.update(path.name.encode())
        digest.update(_sha256(path).encode())
    return digest.hexdigest()


def _resource_envelope(resource_plan: Any) -> dict[str, Any]:
    if resource_plan.multi_node is not None:
        resources = resource_plan.multi_node
        return {
            "mode": "multi_node",
            "gpus_per_node": resources.gpus_per_node,
            "work_gpus": resources.work.gpu_count,
            "work_nodes": resources.work.node_count,
            "judge_gpus": resources.verifier.gpu_count,
            "judge_nodes": resources.verifier.node_count,
            "total_nodes": resources.total_nodes,
            "total_gpus": (
                resources.work.gpu_count + resources.verifier.gpu_count
            ),
            "cpu_slots_per_node": resources.cpu_slots_per_node,
            "memory_mb_per_node": resources.memory_mb_per_node,
            "node_tmp_mb": resources.node_tmp_mb,
            "shared_workspace_mb": resources.shared_workspace_mb,
            "build_walltime": resources.build_walltime,
            "run_walltime": resources.run_walltime,
            "pool_policy": "ordered disjoint Work prefix and Judge suffix",
        }
    resources = resource_plan.single_node
    assert resources is not None
    return {
        "mode": "single_node",
        "gpus_per_node": None,
        "work_gpus": resources.work_gpus,
        "work_nodes": 1 if resources.work_gpus else 0,
        "judge_gpus": resources.verifier_gpus,
        "judge_nodes": 1 if resources.verifier_gpus else 0,
        "total_nodes": 1,
        "total_gpus": resources.total_gpus,
        "cpu_slots_per_node": resources.cpu_slots,
        "memory_mb_per_node": resources.memory_mb,
        "node_tmp_mb": resources.local_tmp_mb,
        "shared_workspace_mb": resources.local_tmp_mb,
        "build_walltime": resources.build_walltime,
        "run_walltime": resources.run_walltime,
        "pool_policy": "legacy single-node Harness behavior",
    }


def build_plan(
    task_dir: Path,
    *,
    harness_root: Path,
    cluster: str | Path = "bluevela",
    scheduler_policy: str = "default",
    allocation_evidence: str | None = None,
) -> dict[str, object]:
    """Validate and plan without allocating, building, or mutating state."""

    task = task_dir.resolve()
    if (task / "cluster").exists():
        raise ValueError(
            "portable RSI-Harness task contains a task-owned cluster control plane"
        )
    validator = Validator(task)
    findings = validator.validate()
    errors = [item for item in findings if item.severity == "ERROR"]
    if errors:
        detail = "; ".join(f"{item.code}:{item.path}" for item in errors)
        raise ValueError(f"task has static contract errors: {detail}")

    harness = _load_harness(harness_root)
    primary, direction, submissions = _read_run_options(task)
    options = harness["CompileOptions"](
        primary_reward=primary,
        score_direction=direction,
        max_submissions=submissions,
    )
    try:
        definition = harness["HarborTaskCompiler"]().compile(task, options)
        profile_file = _profile_path(harness["root"], cluster)
        profile = harness["load_cluster_profile"](
            profile_file,
            environ={**os.environ, "USER": os.environ.get("USER", "operator")},
        )
        resource_plan = harness["derive_resource_plan"](definition, profile)
    except Exception as error:
        raise ValueError(str(error)) from error

    missing_assets = _validate_asset_binds(definition, profile)
    if missing_assets:
        detail = ", ".join(
            f"{item['phase']}:{item['path']}" for item in missing_assets
        )
        raise ValueError(f"profile does not bind declared task assets: {detail}")

    scheduler_contract = _scheduler_policy(
        profile,
        policy=scheduler_policy,
        allocation_evidence=allocation_evidence,
    )

    with (task / "task.toml").open("rb") as stream:
        config = tomllib.load(stream)
    metadata = config.get("metadata", {})
    baseline_status = (
        metadata.get("baseline_status") if isinstance(metadata, dict) else None
    )
    external_baseline_authority_required = (
        isinstance(baseline_status, str)
        and "certif" in baseline_status.lower()
        and "required" in baseline_status.lower()
    )
    task_gpu_count = definition.gpu_requirement.count
    if not isinstance(task_gpu_count, int):
        raise ValueError("Blue Vela preflight requires a numeric Work GPU count")

    executable = harness["root"] / ".venv/bin/rsi-harness"
    if not executable.is_file():
        executable = harness["root"] / ".venv/bin/rsi-harness"
    cluster_argument = (
        "bluevela" if str(cluster) == "bluevela" else str(profile_file)
    )
    run_command = [
        str(executable),
        "run",
        str(task),
        "--cluster",
        cluster_argument,
    ]
    resource_envelope = _resource_envelope(resource_plan)
    warnings = [
        {
            "code": item.code,
            "path": item.path,
            "message": item.message,
        }
        for item in findings
        if item.severity == "WARNING"
    ]
    return {
        "task": str(task),
        "workflow": "approved_proposal_to_end_to_end_reward",
        "fingerprints": {
            "task_sha256": _tree_sha256(task),
            "harness_contract_sha256": _harness_contract_sha256(harness["root"]),
            "profile_sha256": _sha256(profile_file),
        },
        "task_contract": {
            "schema_version": config.get("schema_version"),
            "task_id": definition.task_id,
            "work_gpus": task_gpu_count,
            "judge_gpus": definition.verifier.gpu_count,
            "agent_timeout_sec": definition.agent.timeout_seconds,
            "verifier_timeout_sec": definition.verifier.timeout_seconds,
            "primary_reward": primary,
            "score_direction": direction,
            "max_submissions": submissions,
            "baseline_status": baseline_status,
            "external_baseline_authority_required": (
                external_baseline_authority_required
            ),
            "assets": [
                item.model_dump(mode="json") for item in definition.assets
            ],
        },
        "profile": {
            "name": profile.name,
            "source": str(profile_file),
            "queue": profile.scheduler.queue,
            "group": profile.scheduler.group,
            "exclusive": profile.scheduler.exclusive,
            "excluded_hosts": list(profile.scheduler.excluded_hosts),
            "gpus_per_node": profile.resources.gpus_per_node,
            "run_root": str(profile.storage.run_root),
            "image_cache": str(profile.storage.image_cache),
            "logs_root": str(profile.storage.logs_root),
        },
        "resource_plan": resource_envelope,
        "scheduler_policy": scheduler_contract,
        "static_validation": {
            "errors": 0,
            "warnings": warnings,
            "live_ready": not warnings,
            "warning_disposition_required": bool(warnings),
            "compiler": "passed",
            "profile_asset_coverage": "passed",
        },
        "commands": {
            "static": [
                str(harness["root"] / ".venv/bin/python"),
                str(Path(__file__).with_name("validate_task.py")),
                str(task),
                "--harness-root",
                str(harness["root"]),
                "--json",
            ],
            "dry_run": [*run_command, "--dry-run"],
            "run": run_command,
        },
        "live_gates": [
            "PROFILE_READY",
            "IMAGE_READY",
            "ALLOCATION_READY",
            "WORK_READY",
            "SUBMISSION_READY",
            "JUDGE_READY",
            "END_TO_END_VALIDATED",
        ],
        "conditional_baseline_authority": {
            "required": external_baseline_authority_required,
            "acceptance": (
                "fresh uninterrupted rewardless certification followed by one "
                "normal finite-reward baseline run"
                if external_baseline_authority_required
                else "one normal scoreable baseline run"
            ),
        },
        "execution_policy": {
            "canonical_profile_mutation_forbidden": True,
            "task_mutation_after_freeze_forbidden": True,
            "fallback_must_keep_task_resource_totals": True,
            "resource_reduction_requires_explicit_user_instruction": True,
            "completed_fallback_requires_no_duplicate_normal_run": True,
            "interrupted_certification_stitching_forbidden": True,
        },
        "approval_scope": (
            "named fingerprints, data preparation, SIF build/cache, exact final "
            "allocation, in-envelope source-controlled repairs/fresh retries, and "
            "cleanup of exact resources created by this validation"
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path)
    parser.add_argument("--harness-root", type=Path, required=True)
    parser.add_argument("--cluster", default="bluevela")
    parser.add_argument(
        "--scheduler-policy",
        choices=("default", "allocation-fallback"),
        default="default",
    )
    parser.add_argument("--allocation-evidence")
    parser.add_argument("--json", action="store_true", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan = build_plan(
            args.task,
            harness_root=args.harness_root,
            cluster=args.cluster,
            scheduler_policy=args.scheduler_policy,
            allocation_evidence=args.allocation_evidence,
        )
    except (FileNotFoundError, OSError, ValueError, tomllib.TOMLDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps(plan, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
