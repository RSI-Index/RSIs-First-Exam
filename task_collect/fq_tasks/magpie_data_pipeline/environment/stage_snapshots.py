#!/usr/bin/env python3
"""Convert pinned raw Magpie and benchmark snapshots into local JSONL inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

try:
    from ..shared.dataset_contract import canonicalize_record, compact_json
except ImportError:  # pragma: no cover - flat /task-tools execution
    from dataset_contract import canonicalize_record, compact_json


SNAPSHOT_NAMES = {
    "baseline": "magpie-baseline.jsonl",
    "alpaca_eval_2": "alpaca_eval_2.jsonl",
    "arena_hard": "arena_hard.jsonl",
    "wildbench": "wildbench.jsonl",
}


def _stable_id(source: str, *parts: str) -> str:
    payload = "\0".join((source, *parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a non-empty string")
    return value


def _messages(raw: object) -> list[dict[str, str]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("conversation must be a non-empty list")
    normalized: list[dict[str, str]] = []
    for message in raw:
        if not isinstance(message, Mapping):
            raise ValueError("conversation messages must be objects")
        role = message.get("role", message.get("from"))
        content = message.get("content", message.get("value"))
        if role in {"user", "human"}:
            role = "user"
        elif role in {"assistant", "gpt"}:
            role = "assistant"
        else:
            raise ValueError("conversation roles must be user or assistant")
        normalized.append({"role": role, "content": _string(content, "conversation content")})
    if not any(message["role"] == "user" for message in normalized):
        raise ValueError("conversation must contain a user message")
    return normalized


def _baseline_row(raw: Mapping[str, object]) -> dict[str, object]:
    identifier = raw.get("id", raw.get("uuid"))
    identifier = _string(identifier, "baseline id")
    conversations = raw.get("conversations")
    if isinstance(conversations, list):
        candidate = {"id": identifier, "conversations": conversations}
    else:
        candidate = {
            "id": identifier,
            "conversations": [
                {"from": "human" if item["role"] == "user" else "gpt", "value": item["content"]}
                for item in _messages(raw.get("messages"))
            ],
        }
    return canonicalize_record(candidate)


def _alpaca_row(raw: Mapping[str, object]) -> dict[str, str]:
    instruction = _string(raw.get("instruction"), "AlpacaEval instruction")
    supplied_input = raw.get("input", "")
    if supplied_input is None:
        supplied_input = ""
    if not isinstance(supplied_input, str):
        raise ValueError("AlpacaEval input must be a string")
    prompt = instruction if not supplied_input else f"{instruction}\n\n{supplied_input}"
    return {"id": str(raw.get("id") or _stable_id("alpaca_eval_2", instruction, supplied_input)), "language": str(raw.get("language", "en")), "prompt": prompt}


def _arena_row(raw: Mapping[str, object]) -> dict[str, str]:
    uid = _string(raw.get("uid", raw.get("id")), "Arena-Hard uid")
    return {"id": uid, "language": str(raw.get("language", "en")), "prompt": _string(raw.get("prompt"), "Arena-Hard prompt")}


def _wildbench_row(raw: Mapping[str, object]) -> dict[str, object]:
    messages = _messages(raw.get("conversation_input", raw.get("conversation", raw.get("messages"))))
    identifier = raw.get("session_id", raw.get("id"))
    identifier = str(identifier) if identifier is not None else _stable_id("wildbench", *(item["content"] for item in messages))
    return {"id": _string(identifier, "WildBench id"), "language": str(raw.get("language", "en")), "messages": messages}


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    payload = b"".join((json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8") for row in rows)
    path.write_bytes(payload)


def stage_rows(raw: Mapping[str, Iterable[Mapping[str, object]]], destination: Path) -> None:
    """Stage small fixtures or parsed real rows into the four immutable snapshot forms."""

    destination = Path(destination)
    snapshots = destination / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    baseline = [_baseline_row(item) for item in raw["baseline"]]
    _write_jsonl(snapshots / SNAPSHOT_NAMES["baseline"], baseline)
    _write_jsonl(destination / "Magpie-Pro-MT-300K-v0.1.jsonl", baseline)
    _write_jsonl(snapshots / SNAPSHOT_NAMES["alpaca_eval_2"], (_alpaca_row(item) for item in raw["alpaca_eval_2"]))
    _write_jsonl(snapshots / SNAPSHOT_NAMES["arena_hard"], (_arena_row(item) for item in raw["arena_hard"]))
    _write_jsonl(snapshots / SNAPSHOT_NAMES["wildbench"], (_wildbench_row(item) for item in raw["wildbench"]))


def read_rows_file(path: Path) -> list[dict[str, object]]:
    """Read exactly one declared pinned data file; repository roots are unsafe."""
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("staging requires one explicit file, never a repository directory")
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError("pyarrow is required to stage the pinned Magpie parquet snapshot") from exc
        return parquet.read_table(path).to_pylist()
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise ValueError("JSON snapshot must be an array of objects")
        return value
    if path.suffix != ".jsonl":
        raise ValueError("staging input must be an explicit .parquet, .json, or .jsonl file")
    files = [path]
    rows: list[dict[str, object]] = []
    for file in files:
        with file.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        raise ValueError(f"no JSONL or parquet rows found in pinned snapshot {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/task-assets"))
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--alpaca-eval-2", type=Path, required=True)
    parser.add_argument("--arena-hard", type=Path, required=True)
    parser.add_argument("--wildbench", type=Path, required=True)
    args = parser.parse_args()
    stage_rows(
        {
            "baseline": read_rows_file(args.baseline),
            "alpaca_eval_2": read_rows_file(args.alpaca_eval_2),
            "arena_hard": read_rows_file(args.arena_hard),
            "wildbench": read_rows_file(args.wildbench),
        },
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
