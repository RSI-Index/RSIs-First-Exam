#!/usr/bin/env python3
"""Contained client for the trusted TMAX blaunch broker."""

from __future__ import annotations

import json
import os
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Any

BLOCKED_EXACT_NAMES = {"HOME", "LOGNAME", "OLDPWD", "PATH", "PWD", "SHELL", "USER"}
BLOCKED_PREFIXES = ("APPTAINER_", "SINGULARITY_", "LSB_", "LSF_")
BLOCKED_NAME_PARTS = (
    "ANTHROPIC",
    "COOKIE",
    "CREDENTIAL",
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "OPENAI",
    "PASSWORD",
    "PRIVATE_KEY",
    "SECRET",
    "SSH_",
)


def safe_client_environment(raw: dict[str, str]) -> dict[str, str]:
    result = {}
    for name, value in raw.items():
        upper = name.upper()
        if upper in BLOCKED_EXACT_NAMES or upper.startswith(BLOCKED_PREFIXES):
            continue
        if any(part in upper for part in BLOCKED_NAME_PARTS):
            continue
        result[name] = value
    return result


def atomic_request(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f".tmp.{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def main() -> int:
    if len(sys.argv) != 5:
        sys.stderr.write("usage: blaunch_proxy_client.py -z HOST /task-tools/run_tmax_attempt.sh CONTRACT\n")
        return 2
    root = Path(os.environ.get("BLAUNCH_PROXY_ROOT", "/blaunch-proxy"))
    if not (root / "READY.json").is_file():
        sys.stderr.write(f"blaunch proxy is not ready: {root / 'READY.json'}\n")
        return 125
    request_id = uuid.uuid4().hex
    request = root / "requests" / f"{request_id}.json"
    output = root / "results" / f"{request_id}.output"
    status = root / "results" / f"{request_id}.status.json"
    heartbeat = root / "heartbeats" / request_id
    cancel = root / "cancel" / request_id
    cancelled = False

    def request_cancel(_signum: int, _frame: Any) -> None:
        nonlocal cancelled
        cancelled = True
        cancel.touch()

    signal.signal(signal.SIGINT, request_cancel)
    signal.signal(signal.SIGTERM, request_cancel)
    heartbeat.touch()
    atomic_request(request, {"argv": sys.argv[1:], "env": safe_client_environment(dict(os.environ))})

    offset = 0
    last_heartbeat = 0.0
    while not status.is_file():
        now = time.time()
        if now - last_heartbeat >= 2:
            heartbeat.touch()
            last_heartbeat = now
        if output.is_file():
            with output.open("rb") as stream:
                stream.seek(offset)
                chunk = stream.read()
            if chunk:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                offset += len(chunk)
        time.sleep(0.2)

    if output.is_file():
        with output.open("rb") as stream:
            stream.seek(offset)
            chunk = stream.read()
        if chunk:
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
    result = json.loads(status.read_text())
    for path in (heartbeat, cancel, output, status):
        path.unlink(missing_ok=True)
    return 143 if cancelled else int(result["returncode"])


if __name__ == "__main__":
    raise SystemExit(main())
