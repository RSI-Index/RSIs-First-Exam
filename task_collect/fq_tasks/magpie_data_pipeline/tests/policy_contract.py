"""Fail-closed source-tree and selected-artifact policy for Magpie Harbor."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from .dataset_contract import DatasetValidationError, validate_dataset
    from .task_contract import ASSET_REVISIONS
except ImportError:  # pragma: no cover - flat /tests verifier execution
    from dataset_contract import DatasetValidationError, validate_dataset
    from task_contract import ASSET_REVISIONS


WRITABLE_PREFIX = "data_research"
MAX_SOURCE_FILE_BYTES = 10 * 1024 * 1024
REQUIRED_OUTPUT_FILES = (
    "dataset.jsonl", "dataset_manifest.json", "pipeline_config.toml", "provenance.json", "experiments.jsonl",
    "submission-selection.json", "control-complete.json",
)
PROTECTED_PATHS = ("/models", "/task-assets", "/task-tools", "/opt/project", "/tests")
_IGNORED_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ATTEMPT = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _has_symlink_parent(path: Path, root: Path) -> bool:
    current = path
    while current != root:
        if current.is_symlink():
            return True
        current = current.parent
    return root.is_symlink()


def _inventory(root: Path) -> tuple[dict[str, tuple[str, str]], list[str]]:
    entries: dict[str, tuple[str, str]] = {}
    unsafe: list[str] = []
    if not root.is_dir() or root.is_symlink():
        return entries, ["."]
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if any(part in _IGNORED_PARTS for part in path.relative_to(root).parts):
            continue
        if path.is_symlink() or _has_symlink_parent(path, root):
            unsafe.append(relative)
        elif path.is_dir():
            entries[relative] = ("directory", "")
        elif path.is_file():
            entries[relative] = ("file", _sha256(path))
        else:
            unsafe.append(relative)
    return entries, unsafe


def _allowed(relative: str) -> bool:
    return relative == WRITABLE_PREFIX or relative.startswith(f"{WRITABLE_PREFIX}/")


def _safe_regular_file(path: Path, root: Path) -> bool:
    return path.is_file() and not path.is_symlink() and not _has_symlink_parent(path, root)


def _safe_directory(path: Path, root: Path) -> bool:
    return path.is_dir() and not path.is_symlink() and not _has_symlink_parent(path, root)


def _load_json(path: Path, name: str, errors: list[str], root: Path) -> dict[str, Any] | None:
    if not _safe_regular_file(path, root):
        errors.append(f"unsafe_output:{name}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        errors.append(f"invalid_output:{name}")
        return None
    if not isinstance(value, dict):
        errors.append(f"invalid_output:{name}")
        return None
    return value


def _sha(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256.fullmatch(value))


def _canonical_source_manifest(root: Path) -> tuple[dict[str, object], str, str, str]:
    if root.is_symlink() or not root.is_dir() or _has_symlink_parent(root, root):
        raise ValueError("data_research must be an ordinary directory")
    files: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or _has_symlink_parent(path, root):
            raise ValueError("data_research contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("data_research contains a non-regular file")
        files.append({"bytes": path.stat().st_size, "path": path.relative_to(root).as_posix(), "sha256": _sha256(path)})
    manifest: dict[str, object] = {"files": files, "version": 1}
    by_path = {str(item["path"]): str(item["sha256"]) for item in files}
    if "pipeline.py" not in by_path or "pipeline_config.toml" not in by_path:
        raise ValueError("data_research pipeline and config are required")
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return manifest, hashlib.sha256(payload).hexdigest(), by_path["pipeline.py"], by_path["pipeline_config.toml"]


def _unsafe_output_entries(root: Path) -> list[str]:
    if root.is_symlink() or not root.is_dir():
        return ["."]
    unsafe: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink() or _has_symlink_parent(path, root) or (not path.is_dir() and not path.is_file()):
            unsafe.append(relative)
    return unsafe


def _validate_submission(output: Path, tokenizer: Any, baseline_token_cap: int, hidden_prompts: list[str], chat_template: str, source_manifest: dict[str, object], source_tree_sha: str, source_sha: str, config_sha: str, errors: list[str]) -> None:
    dataset = output / "dataset.jsonl"
    manifest = _load_json(output / "dataset_manifest.json", "dataset_manifest.json", errors, output)
    provenance = _load_json(output / "provenance.json", "provenance.json", errors, output)
    selection = _load_json(output / "submission-selection.json", "submission-selection.json", errors, output)
    control = _load_json(output / "control-complete.json", "control-complete.json", errors, output)
    if not _safe_regular_file(dataset, output):
        errors.append("unsafe_output:dataset.jsonl")
        return
    try:
        summary = validate_dataset(
            dataset,
            tokenizer,
            baseline_token_cap,
            hidden_prompts,
            chat_template=chat_template,
        )
    except (DatasetValidationError, OSError, ValueError) as exc:
        errors.append(f"invalid_output:dataset.jsonl:{exc}")
        return
    expected = {"records": summary.records, "tokens": summary.tokens, "dataset_sha256": summary.sha256}
    if not isinstance(manifest, dict) or any(manifest.get(key) != value for key, value in expected.items()) or manifest.get("pipeline_source_sha256") != source_sha or manifest.get("pipeline_config_sha256") != config_sha or manifest.get("data_research_manifest") != source_manifest or manifest.get("data_research_sha256") != source_tree_sha:
        errors.append("invalid_output:dataset_manifest.json")
    if not isinstance(provenance, dict):
        errors.append("invalid_output:provenance.json")
    else:
        provenance_expected = {**expected, "asset_revisions": dict(ASSET_REVISIONS)}
        if any(provenance.get(key) != value for key, value in provenance_expected.items()):
            errors.append("invalid_output:provenance.json")
        if provenance.get("pipeline_source_sha256") != source_sha or provenance.get("pipeline_config_sha256") != config_sha or provenance.get("data_research_manifest") != source_manifest or provenance.get("data_research_sha256") != source_tree_sha or not isinstance(provenance.get("generator_calls"), list):
            errors.append("invalid_output:provenance.json")
        command = provenance.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(part, str) and part for part in command):
            errors.append("invalid_output:provenance.json")
    selected = selection.get("selected_attempt") if isinstance(selection, dict) else None
    if not isinstance(selected, str) or not _ATTEMPT.fullmatch(selected) or selection.get("status") != "selected" or selection.get("dataset_sha256") != summary.sha256:
        errors.append("invalid_output:submission-selection.json")
        selected = None
    if not isinstance(control, dict) or control.get("status") != "complete" or control.get("selected_attempt") != selected or control.get("dataset_sha256") != summary.sha256:
        errors.append("invalid_output:control-complete.json")
    if isinstance(provenance, dict) and provenance.get("attempt_id") != selected:
        errors.append("invalid_output:provenance.json")
    selected_config = output / "pipeline_config.toml"
    if not _safe_regular_file(selected_config, output) or _sha256(selected_config) != config_sha:
        errors.append("invalid_output:pipeline_config.toml")
    _validate_lifecycle(output, selected, summary.sha256, errors)


def _validate_lifecycle(output: Path, selected: str | None, dataset_sha256: str, errors: list[str]) -> None:
    ledger = output / "experiments.jsonl"
    if not _safe_regular_file(ledger, output):
        errors.append("unsafe_output:experiments.jsonl")
        return
    terminal_events: dict[str, list[dict[str, Any]]] = {}
    selected_events: list[dict[str, Any]] = []
    try:
        lines = ledger.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise ValueError
        for line in lines:
            if not line:
                raise ValueError
            event = json.loads(line)
            if not isinstance(event, dict) or not isinstance(event.get("attempt_id"), str) or not _ATTEMPT.fullmatch(event["attempt_id"]):
                raise ValueError
            outcome = event.get("outcome")
            if outcome in {"completed", "failed"} and event.get("selected") is False:
                terminal_events.setdefault(event["attempt_id"], []).append(event)
            elif outcome == "selected" and event.get("selected") is True:
                selected_events.append(event)
            else:
                raise ValueError
    except (OSError, ValueError, json.JSONDecodeError):
        errors.append("invalid_output:experiments.jsonl")
        return
    attempts_root = output / "attempts"
    if not _safe_directory(attempts_root, output):
        errors.append("invalid_output:experiments.jsonl")
        return
    try:
        attempt_directories = {path.name: path for path in attempts_root.iterdir() if _safe_directory(path, output)}
        if len(attempt_directories) != len(list(attempts_root.iterdir())) or any(not _ATTEMPT.fullmatch(name) for name in attempt_directories):
            raise ValueError
    except (OSError, ValueError):
        errors.append("invalid_output:experiments.jsonl")
        return
    if set(attempt_directories) != set(terminal_events) or any(len(events) != 1 for events in terminal_events.values()):
        errors.append("invalid_output:experiments.jsonl")
        return
    statuses: dict[str, dict[str, Any]] = {}
    for attempt_id, attempt_dir in attempt_directories.items():
        status = _load_json(attempt_dir / "status.json", f"attempts/{attempt_id}/status.json", errors, output)
        event = terminal_events[attempt_id][0]
        if not isinstance(status, dict) or status.get("attempt_id") != attempt_id or status.get("status") != event.get("outcome"):
            errors.append("invalid_output:experiments.jsonl")
            return
        if event.get("outcome") == "completed":
            digest = event.get("dataset_sha256")
            if not _sha(digest) or status.get("dataset_sha256") != digest:
                errors.append("invalid_output:experiments.jsonl")
                return
        statuses[attempt_id] = status
    matching_selected = [event for event in selected_events if event.get("attempt_id") == selected and event.get("dataset_sha256") == dataset_sha256]
    if len(selected_events) != 1 or len(matching_selected) != 1 or selected not in statuses or statuses[selected].get("status") != "completed" or statuses[selected].get("dataset_sha256") != dataset_sha256:
        errors.append("invalid_output:experiments.jsonl")
        return
    selected_dir = attempt_directories[selected]
    for name in ("dataset.jsonl", "dataset_manifest.json", "provenance.json", "pipeline_config.toml"):
        attempt_file, final_file = selected_dir / name, output / name
        if not _safe_regular_file(attempt_file, output) or not _safe_regular_file(final_file, output) or _sha256(attempt_file) != _sha256(final_file):
            errors.append("invalid_output:experiments.jsonl")
            return


def audit_candidate(clean: Path, candidate: Path, output_dir: Path, *, tokenizer: Any | None = None, baseline_token_cap: int | None = None, hidden_prompts: list[str] | None = None, chat_template: str | None = None) -> list[str]:
    """Return deterministic hard-policy errors; missing evidence is a hard failure."""

    clean, candidate, output = Path(clean), Path(candidate), Path(output_dir)
    before, clean_unsafe = _inventory(clean)
    after, candidate_unsafe = _inventory(candidate)
    errors = [*(f"unsafe_clean:{item}" for item in clean_unsafe), *(f"unsafe_source:{item}" for item in candidate_unsafe)]
    errors.extend(f"unsafe_output:{item}" for item in _unsafe_output_entries(output))
    for relative in sorted(before.keys() | after.keys()):
        old, new = before.get(relative), after.get(relative)
        if old is None:
            if not _allowed(relative): errors.append(f"forbidden_added:{relative}")
        elif new is None:
            if not _allowed(relative): errors.append(f"forbidden_deleted:{relative}")
        elif old[0] != new[0]:
            if not _allowed(relative): errors.append(f"forbidden_type_changed:{relative}")
        elif old[1] != new[1] and not _allowed(relative):
            errors.append(f"forbidden_modified:{relative}")
    for relative, (kind, _) in after.items():
        if kind != "file": continue
        source = candidate / relative
        if source.name == "sitecustomize.py": errors.append(f"forbidden_source:{relative}")
        if source.stat().st_size > MAX_SOURCE_FILE_BYTES: errors.append(f"oversized_source:{relative}")
    editable = candidate / WRITABLE_PREFIX
    source, config = editable / "pipeline.py", editable / "pipeline_config.toml"
    if not _safe_regular_file(source, candidate) or not _safe_regular_file(config, candidate):
        errors.append("unverifiable:editable_source")
    elif tokenizer is None or not isinstance(baseline_token_cap, int) or isinstance(baseline_token_cap, bool) or baseline_token_cap < 0 or not isinstance(chat_template, str):
        errors.append("unverifiable:dataset_contract")
    else:
        try:
            source_manifest, source_tree_sha, source_sha, config_sha = _canonical_source_manifest(editable)
        except (OSError, ValueError):
            errors.append("unverifiable:editable_source")
        else:
            _validate_submission(output, tokenizer, baseline_token_cap, list(hidden_prompts or ()), chat_template, source_manifest, source_tree_sha, source_sha, config_sha, errors)
    return sorted(set(errors))
