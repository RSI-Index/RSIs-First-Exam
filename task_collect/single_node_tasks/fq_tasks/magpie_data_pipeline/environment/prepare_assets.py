#!/usr/bin/env python3
"""Prepare root-only Magpie baseline-budget and evaluation-manifest assets offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:  # Package execution in CPU contract tests.
    from ..shared.dataset_contract import write_canonical_jsonl
    from ..shared.evaluation_contract import select_prompt_ids
    from ..shared.task_contract import ASSET_REVISIONS, RUNTIME_PATHS, load_llama3_chat_template
except ImportError:  # Flat /task-tools execution in the Harbor image.
    from dataset_contract import write_canonical_jsonl
    from evaluation_contract import select_prompt_ids
    from task_contract import ASSET_REVISIONS, RUNTIME_PATHS, load_llama3_chat_template


EVALUATION_SOURCES = ("alpaca_eval_2", "arena_hard", "wildbench")
_SOURCE_ASSET_KEYS = {
    "alpaca_eval_2": ("AlpacaEval code and prompts", "AlpacaEval 2 data"),
    "arena_hard": ("Arena-Hard code", "Arena-Hard data"),
    "wildbench": ("WildBench code", "WildBench data"),
}
_BASELINE_ASSET_KEY = "Magpie-Align/Magpie-Pro-MT-300K-v0.1"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink() and not any(parent.is_symlink() for parent in path.parents)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not _regular_file(path):
        raise ValueError(f"pinned snapshot must be an ordinary file: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"pinned snapshot has invalid JSON at line {line_number}: {path}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"pinned snapshot row must be an object at line {line_number}: {path}")
            records.append(item)
    return records


def _canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _write_asset(path: Path, payload: bytes, mode: int) -> None:
    """Write an immutable build artifact, rejecting divergent prior state."""

    if path.exists() or path.is_symlink():
        if not _regular_file(path) or path.read_bytes() != payload:
            raise ValueError(f"existing asset has mismatched content: {path}")
        os.chmod(path, mode)
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    os.chmod(path, mode)


def _prompt_entry(raw: Mapping[str, Any], source: str) -> dict[str, Any]:
    stable_id = raw.get("id")
    if not isinstance(stable_id, str) or not stable_id:
        raise ValueError(f"{source} snapshot row has no stable id")
    if source == "wildbench":
        messages = raw.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{source} snapshot row {stable_id!r} has no message history")
        normalized: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, Mapping) or message.get("role") not in {"user", "assistant"}:
                raise ValueError(f"{source} snapshot row {stable_id!r} has invalid message history")
            content = message.get("content")
            if not isinstance(content, str) or not content:
                raise ValueError(f"{source} snapshot row {stable_id!r} has invalid message history")
            normalized.append({"role": str(message["role"]), "content": content})
        if not any(message["role"] == "user" for message in normalized):
            raise ValueError(f"{source} snapshot row {stable_id!r} has no user message")
        return {"id": stable_id, "source": source, "messages": normalized}
    prompt = raw.get("prompt", raw.get("instruction", raw.get("input", "")))
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"{source} snapshot row {stable_id!r} has no prompt")
    return {"id": stable_id, "source": source, "prompt": prompt}


def _selected_source_entries(source: str, path: Path) -> list[dict[str, Any]]:
    raw_rows = _read_jsonl(path)
    eligible: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for raw in raw_rows:
        stable_id = raw.get("id")
        language = raw.get("language", "en")
        if not isinstance(stable_id, str) or not stable_id:
            raise ValueError(f"{source} snapshot row has no stable id")
        if not isinstance(language, str):
            raise ValueError(f"{source} snapshot row {stable_id!r} has invalid language")
        eligible.append({"id": stable_id, "source": source, "language": language})
        by_id[stable_id] = _prompt_entry(raw, source)
    selected_ids = select_prompt_ids(eligible, source)
    return [by_id[stable_id] for stable_id in selected_ids]


def _baseline_summary(baseline_path: Path, tokenizer: Any, chat_template: str) -> tuple[int, int, str]:
    rows = _read_jsonl(baseline_path)
    with tempfile.NamedTemporaryFile(prefix="magpie-baseline-", suffix=".jsonl", delete=False) as handle:
        canonical_path = Path(handle.name)
    try:
        summary = write_canonical_jsonl(
            canonical_path,
            rows,
            tokenizer,
            baseline_token_cap=2**63 - 1,
            hidden_prompts=(),
            chat_template=chat_template,
        )
    finally:
        canonical_path.unlink(missing_ok=True)
    return summary.records, summary.tokens, summary.sha256


def prepare_assets(
    output_dir: Path,
    baseline_path: Path,
    source_paths: Mapping[str, Path],
    tokenizer: Any,
    chat_template: str,
) -> dict[str, object]:
    """Materialize canonical immutable asset metadata from only pinned local snapshots."""

    output_dir = Path(output_dir)
    if output_dir.exists() and (output_dir.is_symlink() or not output_dir.is_dir()):
        raise ValueError("asset output must be an ordinary directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    if set(source_paths) != set(EVALUATION_SOURCES):
        raise ValueError("source paths must provide exactly the frozen evaluation sources")

    records, tokens, canonical_sha256 = _baseline_summary(Path(baseline_path), tokenizer, chat_template)
    budget = {
        "baseline_revision": ASSET_REVISIONS[_BASELINE_ASSET_KEY],
        "canonical_dataset_sha256": canonical_sha256,
        "canonical_records": records,
        "canonical_tokens": tokens,
        "snapshot_sha256": _sha256_file(Path(baseline_path)),
    }
    selected = {source: _selected_source_entries(source, Path(source_paths[source])) for source in EVALUATION_SOURCES}
    source_hashes = {source: _sha256_file(Path(source_paths[source])) for source in EVALUATION_SOURCES}
    source_revisions = {source: {key: ASSET_REVISIONS[key] for key in _SOURCE_ASSET_KEYS[source]} for source in EVALUATION_SOURCES}
    manifest = {
        "prompt_count": sum(len(items) for items in selected.values()),
        "source_revisions": source_revisions,
        "source_snapshot_sha256": source_hashes,
        "sources": selected,
    }
    budget_payload = _canonical_json(budget)
    manifest_payload = _canonical_json(manifest)
    # The worker must be able to enforce its public token budget. Evaluation
    # identities and contents remain verifier-only.
    _write_asset(output_dir / "baseline_budget.json", budget_payload, 0o444)
    _write_asset(output_dir / "eval_manifest.json", manifest_payload, 0o600)
    return {**budget, "eval_manifest_sha256": _sha256_bytes(manifest_payload)}


def _load_tokenizer() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(RUNTIME_PATHS["student_model"], local_files_only=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/task-assets"))
    parser.add_argument("--baseline", type=Path, default=Path("/task-assets/snapshots/magpie-baseline.jsonl"))
    for source in EVALUATION_SOURCES:
        parser.add_argument(f"--{source.replace('_', '-')}", type=Path, default=Path(f"/task-assets/snapshots/{source}.jsonl"))
    args = parser.parse_args()
    sources = {source: getattr(args, source) for source in EVALUATION_SOURCES}
    prepare_assets(args.output, args.baseline, sources, _load_tokenizer(), load_llama3_chat_template())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
