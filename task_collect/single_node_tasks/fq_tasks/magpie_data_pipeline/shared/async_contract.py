"""Atomic lifecycle primitives for same-session Magpie dataset attempts."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TERMINAL_STATES = frozenset({"completed", "failed"})
_LEGAL_TRANSITIONS = {
    "queued": frozenset({"running"}),
    "running": TERMINAL_STATES,
    "completed": frozenset(),
    "failed": frozenset(),
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(path: str | Path, payload: dict[str, Any]) -> None:
    """Publish JSON atomically with a sibling temporary file and ``os.replace``."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_status(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("status"), str):
        raise ValueError("status payload must contain a status")
    return payload


def transition_status(path: str | Path, status: str, **changes: Any) -> dict[str, Any]:
    """Apply exactly one legal state transition and return the published payload."""

    current = read_status(path)
    previous = current["status"]
    if status not in _LEGAL_TRANSITIONS.get(previous, frozenset()):
        if previous in TERMINAL_STATES:
            raise RuntimeError(f"terminal status cannot transition from {previous}")
        raise RuntimeError(f"illegal status transition {previous} -> {status}")
    payload = {**current, **changes, "status": status}
    if status == "running":
        payload.setdefault("started_at", now_utc())
    if status in TERMINAL_STATES:
        payload.setdefault("finished_at", now_utc())
    write_status(path, payload)
    return payload


def run_command_with_status(
    command: list[str],
    *,
    status_path: str | Path,
    log_path: str | Path,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> int:
    """Run a real worker command while atomically publishing terminal state."""

    write_status(status_path, {**(metadata or {}), "command": command, "status": "queued"})
    transition_status(status_path, "running")
    destination = Path(log_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        transition_status(status_path, "completed" if completed.returncode == 0 else "failed", return_code=completed.returncode)
        return completed.returncode
    except Exception as exc:
        transition_status(status_path, "failed", return_code=127, error=str(exc))
        return 127
