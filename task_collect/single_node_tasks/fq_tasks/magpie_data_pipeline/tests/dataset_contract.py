"""Strict, dependency-free handling for untrusted Magpie training JSONL."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any
import unicodedata

try:
    from .task_contract import LLAMA3_CHAT_TEMPLATE_SHA256
except ImportError:  # pragma: no cover - flat verifier/task-tools execution
    from task_contract import LLAMA3_CHAT_TEMPLATE_SHA256


MAX_RECORDS = 300_000
SEQUENCE_LENGTH = 8_192
_RECORD_FIELDS = frozenset({"id", "conversations"})
_MESSAGE_FIELDS = frozenset({"from", "value"})
_ROLES = ("human", "gpt")


class DatasetValidationError(ValueError):
    """Raised when an untrusted dataset cannot satisfy the frozen contract."""


@dataclass(frozen=True)
class DatasetSummary:
    records: int
    tokens: int
    sha256: str
    duplicate_records: int


def _normalized_string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise DatasetValidationError(f"{field} must be a string")
    normalized = unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    if not normalized:
        raise DatasetValidationError(f"{field} must be non-empty")
    if "\0" in normalized:
        raise DatasetValidationError(f"{field} must not contain NUL")
    try:
        normalized.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise DatasetValidationError(f"{field} must be valid UTF-8") from exc
    return normalized


def canonicalize_record(raw: object) -> dict[str, object]:
    """Normalize one ShareGPT row while rejecting all non-contract fields."""

    if not isinstance(raw, dict):
        raise DatasetValidationError("record must be an object")
    if set(raw) != _RECORD_FIELDS:
        raise DatasetValidationError("record has unknown or missing fields")
    record_id = _normalized_string(raw["id"], "record id")
    conversations = raw["conversations"]
    if not isinstance(conversations, list) or not conversations:
        raise DatasetValidationError("conversations must be a non-empty list")
    if len(conversations) % 2:
        raise DatasetValidationError("conversations must end with final gpt response")
    canonical_messages: list[dict[str, str]] = []
    for index, message in enumerate(conversations):
        if not isinstance(message, dict) or set(message) != _MESSAGE_FIELDS:
            raise DatasetValidationError("message has unknown or missing fields")
        expected_role = _ROLES[index % len(_ROLES)]
        if message["from"] != expected_role:
            raise DatasetValidationError("conversations must alternate human and gpt")
        canonical_messages.append(
            {
                "from": expected_role,
                "value": _normalized_string(message["value"], "message value"),
            }
        )
    return {"id": record_id, "conversations": canonical_messages}


def compact_json(record: dict[str, object]) -> str:
    """Encode a canonical record in the byte representation used for hashing."""

    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DatasetValidationError("duplicate JSON key")
        result[key] = value
    return result


def _contains_symlink(path: Path) -> bool:
    return any(component.is_symlink() for component in (path, *path.parents))


def _token_count(record: dict[str, object], tokenizer: Any, chat_template: str) -> int:
    messages = [
        {
            "role": "user" if message["from"] == "human" else "assistant",
            "content": message["value"],
        }
        for message in record["conversations"]  # type: ignore[index]
    ]
    if not isinstance(chat_template, str) or hashlib.sha256(chat_template.encode("utf-8")).hexdigest() != LLAMA3_CHAT_TEMPLATE_SHA256:
        raise DatasetValidationError("chat template must match the frozen Llama-3 template")
    try:
        tokens = tokenizer.apply_chat_template(
            messages,
            chat_template=chat_template,
            tokenize=True,
            add_generation_prompt=False,
        )
    except Exception as exc:
        raise DatasetValidationError("tokenizer failed to apply the frozen chat template") from exc
    if not isinstance(tokens, (list, tuple)) or not all(isinstance(token, int) for token in tokens):
        raise DatasetValidationError("tokenizer must return token id integers")
    return len(tokens)


def _contamination_error(record: dict[str, object], hidden_prompts: Sequence[str]) -> str | None:
    normalized_hidden = [_comparison_text(prompt) for prompt in hidden_prompts]
    for message in record["conversations"]:  # type: ignore[index]
        candidate = _comparison_text(message["value"])
        if candidate in normalized_hidden:
            return "record exactly matches a hidden prompt"
        candidate_grams = _thirteen_grams(candidate)
        if not candidate_grams:
            continue
        for prompt in normalized_hidden:
            prompt_grams = _thirteen_grams(prompt)
            if prompt_grams and len(candidate_grams & prompt_grams) / len(prompt_grams) > 0.8:
                return "record has hidden prompt 13-gram overlap above 0.8"
    return None


def _comparison_text(value: object) -> str:
    if not isinstance(value, str):
        raise DatasetValidationError("hidden prompts must be strings")
    normalized = unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    return " ".join(normalized.casefold().split())


def _thirteen_grams(value: str) -> set[tuple[str, ...]]:
    words = value.split()
    return {tuple(words[index : index + 13]) for index in range(len(words) - 12)}


def _summary(
    records: Iterable[dict[str, object]],
    tokenizer: Any,
    baseline_token_cap: int,
    hidden_prompts: Sequence[str],
    chat_template: str,
) -> tuple[list[dict[str, object]], DatasetSummary]:
    if not isinstance(baseline_token_cap, int) or baseline_token_cap < 0:
        raise DatasetValidationError("baseline token cap must be a non-negative integer")
    if any(not isinstance(prompt, str) for prompt in hidden_prompts):
        raise DatasetValidationError("hidden prompts must be strings")
    seen_ids: set[str] = set()
    canonical_records: list[dict[str, object]] = []
    total_tokens = 0
    for raw in records:
        canonical = canonicalize_record(raw)
        record_id = canonical["id"]
        if record_id in seen_ids:
            raise DatasetValidationError("duplicate record id")
        seen_ids.add(record_id)  # type: ignore[arg-type]
        if len(seen_ids) > MAX_RECORDS:
            raise DatasetValidationError("dataset has more than 300000 records")
        contamination = _contamination_error(canonical, hidden_prompts)
        if contamination:
            raise DatasetValidationError(contamination)
        tokens = _token_count(canonical, tokenizer, chat_template)
        if tokens > SEQUENCE_LENGTH:
            raise DatasetValidationError("record exceeds frozen sequence length 8192")
        total_tokens += tokens
        if total_tokens > baseline_token_cap:
            raise DatasetValidationError("dataset exceeds baseline token budget")
        canonical_records.append(canonical)
    if not canonical_records:
        raise DatasetValidationError("dataset must contain at least one record")
    canonical_records.sort(key=lambda record: record["id"].encode("utf-8"))  # type: ignore[union-attr]
    digest = hashlib.sha256()
    for record in canonical_records:
        digest.update(compact_json(record).encode("utf-8"))
        digest.update(b"\n")
    return canonical_records, DatasetSummary(len(canonical_records), total_tokens, digest.hexdigest(), 0)


def validate_dataset(
    path: Path,
    tokenizer: Any,
    baseline_token_cap: int,
    hidden_prompts: Sequence[str],
    *,
    chat_template: str,
) -> DatasetSummary:
    """Validate a JSONL file one line at a time without importing candidate code."""

    dataset_path = Path(path)
    if _contains_symlink(dataset_path) or not dataset_path.is_file():
        raise DatasetValidationError("dataset must be an ordinary file")

    def parsed_records() -> Iterable[dict[str, object]]:
        try:
            with dataset_path.open("r", encoding="utf-8", newline="") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        raise DatasetValidationError(f"line {line_number} must not be empty")
                    try:
                        raw = json.loads(line, object_pairs_hook=_strict_json_object)
                    except json.JSONDecodeError as exc:
                        raise DatasetValidationError(f"line {line_number} is not valid JSON") from exc
                    yield raw
        except UnicodeDecodeError as exc:
            raise DatasetValidationError("dataset must be valid UTF-8") from exc

    _, summary = _summary(parsed_records(), tokenizer, baseline_token_cap, hidden_prompts, chat_template)
    return summary


def write_canonical_jsonl(
    path: Path,
    rows: Iterable[object],
    tokenizer: Any,
    baseline_token_cap: int,
    hidden_prompts: Sequence[str],
    *,
    chat_template: str,
) -> DatasetSummary:
    """Validate rows, write their canonical sorted JSONL form, and return its digest."""

    canonical, summary = _summary(rows, tokenizer, baseline_token_cap, hidden_prompts, chat_template)  # type: ignore[arg-type]
    destination = Path(path)
    if destination.exists() and (destination.is_symlink() or not destination.is_file()):
        raise DatasetValidationError("canonical dataset destination must be an ordinary file")
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in canonical:
            handle.write(compact_json(record))
            handle.write("\n")
    return summary
