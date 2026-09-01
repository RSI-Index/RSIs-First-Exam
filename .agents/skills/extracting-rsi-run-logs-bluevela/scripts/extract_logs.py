#!/usr/bin/env python3
"""Extract the newest completed RSI-Harness log tree for one task."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import uuid
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9 on the publication host.
    tomllib = None  # type: ignore[assignment]


class ExtractionError(RuntimeError):
    """A user-actionable extraction failure."""


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PROFILE = (
    REPOSITORY_ROOT
    / "RSI-Harness"
    / "src"
    / "rsi_harness"
    / "cluster"
    / "bluevela"
    / "profile.toml"
)
DEFAULT_DESTINATION_ROOT = REPOSITORY_ROOT / "rsi-logs"
PUBLICATION_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the newest completed native RSI-Harness logs for an exact task ID."
        )
    )
    parser.add_argument("task_id", help="Exact Harbor task_id directory name")
    parser.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_PROFILE,
        help="Blue Vela profile TOML used to resolve source roots",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        help="Override profile storage.run_root",
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        help="Override profile storage.logs_root",
    )
    parser.add_argument(
        "--destination-root",
        type=Path,
        default=DEFAULT_DESTINATION_ROOT,
        help="Export root (default: repository rsi-logs)",
    )
    return parser.parse_args(argv)


def _validate_task_id(task_id: str) -> None:
    if (
        not task_id
        or task_id in {".", ".."}
        or "/" in task_id
        or "\\" in task_id
        or "\x00" in task_id
    ):
        raise ExtractionError("task ID must be one nonempty path component")


def _fallback_profile_storage(content: str) -> dict[str, object]:
    """Parse the two string paths needed from [storage] on Python 3.9."""
    section = ""
    storage: dict[str, object] = {}
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if line.startswith("[") and "]" in line:
            section = line[1 : line.index("]")].strip()
            continue
        if section != "storage" or "=" not in line:
            continue
        key, raw_value = (part.strip() for part in line.split("=", 1))
        if key not in {"run_root", "logs_root"}:
            continue
        try:
            storage[key] = ast.literal_eval(raw_value)
        except (SyntaxError, ValueError) as error:
            raise ExtractionError(
                f"cannot parse profile storage.{key}: {raw_value!r}"
            ) from error
    return {"storage": storage}


def _profile_roots(profile: Path) -> tuple[Path, Path]:
    try:
        content = profile.read_text()
    except OSError as error:
        raise ExtractionError(
            f"cannot read Blue Vela profile {profile}: {error}"
        ) from error
    try:
        payload = (
            _fallback_profile_storage(content)
            if tomllib is None
            else tomllib.loads(content)
        )
        storage = payload["storage"]
        if not isinstance(storage, dict):
            raise TypeError("storage is not a table")
        run_root = Path(storage["run_root"]).expanduser()
        logs_root = Path(storage["logs_root"]).expanduser()
    except (KeyError, TypeError, ValueError) as error:
        raise ExtractionError(
            f"cannot read Blue Vela profile {profile}: {error}"
        ) from error
    if not run_root.is_absolute() or not logs_root.is_absolute():
        raise ExtractionError("profile run_root and logs_root must be absolute")
    return run_root, logs_root


def _resolve_roots(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.run_root is not None and args.logs_root is not None:
        return (
            args.run_root.expanduser().resolve(),
            args.logs_root.expanduser().resolve(),
        )
    profile_run_root, profile_logs_root = _profile_roots(args.profile.expanduser())
    run_root = profile_run_root if args.run_root is None else args.run_root.expanduser()
    logs_root = (
        profile_logs_root if args.logs_root is None else args.logs_root.expanduser()
    )
    return run_root.resolve(), logs_root.resolve()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _nonempty_regular_file(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(metadata.st_mode) and metadata.st_size > 0


def _safe_regular_tree(root: Path) -> bool:
    try:
        if root.is_symlink() or not root.is_dir():
            return False
        for path in root.rglob("*"):
            metadata = path.lstat()
            if not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)):
                return False
    except OSError:
        return False
    return True


def _has_completed_round(leaf: Path) -> bool:
    submissions = leaf / "submissions"
    feedback = leaf / "feedback"
    if not submissions.is_dir() or not feedback.is_dir():
        return False
    for report_path in sorted(submissions.glob("*/report.json")):
        report = _read_json(report_path)
        round_id = report_path.parent.name
        if (
            report is not None
            and report.get("status") == "completed"
            and _nonempty_regular_file(feedback / f"{round_id}.log")
        ):
            return True
    return False


def _is_completed_candidate(
    *,
    run_root: Path,
    logs_root: Path,
    run_id: str,
    task_id: str,
) -> bool:
    manifest = _read_json(run_root / run_id / "RUN_INFO.json")
    if manifest is None:
        return False
    if manifest.get("run_id") != run_id or manifest.get("state") != "completed":
        return False
    declared_task = manifest.get("task_id")
    if declared_task is None:
        declared_task = str(manifest.get("purpose", "")).removeprefix(
            "RSI Harness task "
        )
    if declared_task != task_id:
        return False

    leaf = logs_root / "runs" / run_id / task_id
    final_result = _read_json(leaf / "final_result.json")
    if final_result is None or final_result.get("status") != "completed":
        return False
    required = (
        leaf / "agent_output.txt",
        leaf / "run_agent.log",
        leaf / "run-plan.json",
    )
    return (
        all(_nonempty_regular_file(path) for path in required)
        and _has_completed_round(leaf)
        and _safe_regular_tree(leaf)
    )


def _select_latest(run_root: Path, logs_root: Path, task_id: str) -> tuple[str, Path]:
    logs_runs = logs_root / "runs"
    if not run_root.is_dir():
        raise ExtractionError(f"run root is unavailable: {run_root}")
    if not logs_runs.is_dir():
        raise ExtractionError(f"logs runs directory is unavailable: {logs_runs}")

    for log_run in sorted(
        logs_runs.iterdir(), key=lambda path: path.name, reverse=True
    ):
        if log_run.is_symlink() or not log_run.is_dir():
            continue
        run_id = log_run.name
        if _is_completed_candidate(
            run_root=run_root,
            logs_root=logs_root,
            run_id=run_id,
            task_id=task_id,
        ):
            return run_id, log_run / task_id
    raise ExtractionError(f"no completed run found for task {task_id!r}")


def _publication_name(source: Path) -> str:
    plan = _read_json(source / "run-plan.json")
    try:
        agent = plan["task"]["agent"]  # type: ignore[index]
        name = agent["name"]
        model = agent["model"]
        reasoning_effort = agent.get("reasoning_effort")
    except (KeyError, TypeError) as error:
        raise ExtractionError(
            "run-plan.json does not declare task.agent name and model"
        ) from error

    values = (("agent name", name), ("model", model))
    components: list[str] = []
    for field, value in values:
        if not isinstance(value, str) or not PUBLICATION_COMPONENT.fullmatch(value):
            raise ExtractionError(
                f"run-plan.json {field} is not a safe publication component: {value!r}"
            )
        components.append(value)
    if reasoning_effort not in (None, ""):
        if not isinstance(reasoning_effort, str) or not PUBLICATION_COMPONENT.fullmatch(
            reasoning_effort
        ):
            raise ExtractionError(
                "run-plan.json reasoning effort is not a safe publication component: "
                f"{reasoning_effort!r}"
            )
        components.append(reasoning_effort)
    return "-".join(components)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_manifest(root: Path) -> dict[str, tuple[str, int, str]]:
    manifest: dict[str, tuple[str, int, str]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        if stat.S_ISDIR(metadata.st_mode):
            manifest[relative] = ("directory", mode, "")
        elif stat.S_ISREG(metadata.st_mode):
            manifest[relative] = ("file", mode, _file_digest(path))
        else:
            raise ExtractionError(f"unsupported artifact type: {path}")
    return manifest


def _same_tree(left: Path, right: Path) -> bool:
    try:
        return _tree_manifest(left) == _tree_manifest(right)
    except (OSError, ExtractionError):
        return False


def _copy_verified(
    source: Path, destination_root: Path, task_id: str, publication_name: str
) -> Path:
    destination_root.mkdir(parents=True, exist_ok=True)
    destination_task = destination_root / task_id
    destination = destination_task / publication_name

    if destination.exists() or destination.is_symlink():
        if not destination.is_symlink() and _same_tree(source, destination):
            print(f"Destination already matches source: {destination}")
            return destination
        raise ExtractionError(
            f"destination already exists with different contents: {destination}"
        )

    if destination_task.exists() or destination_task.is_symlink():
        if destination_task.is_symlink() or not destination_task.is_dir():
            raise ExtractionError(
                f"destination task path is not a regular directory: {destination_task}"
            )
    else:
        destination_task.mkdir()

    temporary_destination = destination_task / (
        f".{publication_name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        shutil.copytree(source, temporary_destination, copy_function=shutil.copy2)
        if not _same_tree(source, temporary_destination):
            raise ExtractionError("copied log verification failed")
        os.rename(temporary_destination, destination)
    finally:
        if temporary_destination.exists():
            shutil.rmtree(temporary_destination)

    if not _same_tree(source, destination):
        raise ExtractionError("destination verification failed after extraction")
    return destination


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        _validate_task_id(args.task_id)
        run_root, logs_root = _resolve_roots(args)
        run_id, source = _select_latest(run_root, logs_root, args.task_id)
        publication_name = _publication_name(source)
        destination = _copy_verified(
            source,
            args.destination_root.expanduser().resolve(),
            args.task_id,
            publication_name,
        )
    except (ExtractionError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Selected completed run: {run_id}")
    print(f"Source: {source}")
    print(f"Destination: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
