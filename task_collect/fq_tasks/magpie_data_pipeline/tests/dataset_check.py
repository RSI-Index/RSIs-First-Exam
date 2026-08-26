#!/usr/bin/env python3
"""Canonicalize the untrusted dataset before frozen candidate training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Sequence

try:  # Package imports make the CPU contract tests exercise this exact verifier.
    from .dataset_contract import DatasetSummary, DatasetValidationError, validate_dataset, write_canonical_jsonl
    from .task_contract import RUNTIME_PATHS, load_llama3_chat_template
except ImportError:  # Flat /tests execution inside the Harbor container.
    from dataset_contract import DatasetSummary, DatasetValidationError, validate_dataset, write_canonical_jsonl
    from task_contract import RUNTIME_PATHS, load_llama3_chat_template


def _ordinary_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink() and not any(parent.is_symlink() for parent in path.parents)


def _ensure_private_directory(path: Path) -> None:
    path = Path(path)
    if path.exists():
        if path.is_symlink() or not path.is_dir() or any(parent.is_symlink() for parent in path.parents):
            raise DatasetValidationError("verifier output directory must be an ordinary directory")
        metadata = path.stat()
        if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
            raise DatasetValidationError("verifier output directory must be private to the verifier")
        return
    if path.parent.is_symlink() or not path.parent.is_dir() or any(parent.is_symlink() for parent in path.parent.parents):
        raise DatasetValidationError("verifier output directory parent must be an ordinary directory")
    path.mkdir(mode=0o700)
    os.chmod(path, 0o700)
    _ensure_private_directory(path)


def _load_tokenizer() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(RUNTIME_PATHS["student_model"], local_files_only=True)


def _load_budget() -> int:
    budget_path = Path(RUNTIME_PATHS["baseline_budget"])
    if not _ordinary_file(budget_path):
        raise ValueError("baseline_budget.json must be an ordinary file")
    payload = json.loads(budget_path.read_text(encoding="utf-8"))
    cap = payload.get("canonical_tokens") if isinstance(payload, dict) else None
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 0:
        raise ValueError("baseline_budget.json must contain canonical_tokens")
    return cap


def _hidden_prompts() -> list[str]:
    manifest_path = Path("/task-assets/eval_manifest.json")
    if not _ordinary_file(manifest_path):
        raise ValueError("eval_manifest.json must be an ordinary file")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = manifest.get("sources") if isinstance(manifest, dict) else None
    if not isinstance(sources, dict):
        raise ValueError("eval_manifest.json must contain sources")
    prompts: list[str] = []
    for entries in sources.values():
        if not isinstance(entries, list):
            raise ValueError("eval_manifest.json has invalid source entries")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("eval_manifest.json has invalid prompt entry")
            if isinstance(entry.get("prompt"), str):
                prompts.append(entry["prompt"])
            else:
                for message in entry.get("messages", []):
                    if isinstance(message, dict) and message.get("role") == "user" and isinstance(message.get("content"), str):
                        prompts.append(message["content"])
    return prompts


def _rows(path: Path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))
            handle.write("\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def canonicalize_dataset(
    input_path: Path,
    output_path: Path,
    *,
    tokenizer: Any | None = None,
    baseline_token_cap: int | None = None,
    hidden_prompts: Sequence[str] | None = None,
    chat_template: str | None = None,
) -> DatasetSummary:
    """Validate untrusted JSONL, then atomically write its sorted canonical form."""

    source, destination = Path(input_path), Path(output_path)
    if not _ordinary_file(source):
        raise DatasetValidationError("dataset must be an ordinary file")
    _ensure_private_directory(destination.parent)
    if destination.exists() and (destination.is_symlink() or not destination.is_file()):
        raise DatasetValidationError("canonical dataset destination must be an ordinary file")
    active_tokenizer = tokenizer if tokenizer is not None else _load_tokenizer()
    cap = baseline_token_cap if baseline_token_cap is not None else _load_budget()
    prompts = list(hidden_prompts) if hidden_prompts is not None else _hidden_prompts()
    active_template = chat_template if chat_template is not None else load_llama3_chat_template()
    expected = validate_dataset(source, active_tokenizer, cap, prompts, chat_template=active_template)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".canonical.", suffix=".jsonl", dir=destination.parent)
    temporary = Path(temporary_name)
    os.close(descriptor)
    try:
        actual = write_canonical_jsonl(
            temporary,
            _rows(source),
            active_tokenizer,
            cap,
            prompts,
            chat_template=active_template,
        )
        if actual != expected:
            raise DatasetValidationError("canonical dataset summary changed during validation")
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
        os.chmod(destination, 0o600)
    finally:
        temporary.unlink(missing_ok=True)
    _atomic_json(
        destination.parent / "canonical_summary.json",
        {"dataset_sha256": expected.sha256, "records": expected.records, "tokens": expected.tokens},
    )
    return expected


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = canonicalize_dataset(args.input, args.output)
    print(json.dumps({"records": summary.records, "tokens": summary.tokens, "dataset_sha256": summary.sha256}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
