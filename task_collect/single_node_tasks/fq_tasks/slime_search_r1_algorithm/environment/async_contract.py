"""Atomic attempt-state primitives for the Slime Search-R1 task."""

from __future__ import annotations

import fcntl
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ATTEMPT_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
LEGAL_TRANSITIONS = {
    ("queued", "running"),
    ("running", "completed"),
    ("running", "failed"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_attempt_id(value: str) -> str:
    if not isinstance(value, str) or not ATTEMPT_PATTERN.fullmatch(value):
        raise ValueError("attempt id must be a lowercase slug of at most 64 characters")
    return value


def validate_transition(old: str, new: str) -> str:
    if (old, new) not in LEGAL_TRANSITIONS:
        raise ValueError(f"illegal attempt status transition: {old} -> {new}")
    return new


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def transition_status(path: str | Path, new_status: str, **updates: Any) -> dict[str, Any]:
    status_path = Path(path)
    current = read_json(status_path)
    validate_transition(str(current.get("status")), new_status)
    current.update(updates)
    current["status"] = new_status
    current["updated_at"] = utc_now()
    atomic_write_json(status_path, current)
    return current


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    with target.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_events(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return []
    events = []
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"ledger line {number} is not an object")
        events.append(payload)
    return events
