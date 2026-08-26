#!/usr/bin/env python3

"""Container-side client for the trusted host LSF blaunch broker."""

from __future__ import annotations

import json
import os
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Any


SAFE_SECRET_NAMES = {
    "APPTAINERENV_HF_TOKEN",
    "APPTAINERENV_WANDB_API_KEY",
    "HF_TOKEN",
    "WANDB_API_KEY",
}
# These are task-profile inputs consumed by the trusted worker launcher, not
# LSF scheduler identity.  Keep the broader LSB_*/LSF_* deny rule below.
SAFE_LSF_PROFILE_NAMES = {
    "LSF_MEMORY",
    "LSF_SLOTS",
    "LSF_SPAN",
    "LSF_WALLTIME",
}
# Do not leak the current Harbor container's identity or Apptainer runtime
# metadata into the trusted host process that launches the training container.
# APPTAINER itself and explicitly approved APPTAINERENV_* credentials are
# handled separately and remain forwardable.
BLOCKED_EXACT_NAMES = {
    "HOME",
    "LOGNAME",
    "OLDPWD",
    "PATH",
    "PWD",
    "SHELL",
    "USER",
}
BLOCKED_PREFIXES = ("APPTAINER_", "SINGULARITY_")
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


def forwarded_environment() -> dict[str, str]:
    """Return runtime state needed by the trusted remote worker launcher."""

    result: dict[str, str] = {}
    for name, value in os.environ.items():
        upper = name.upper()
        if upper in SAFE_LSF_PROFILE_NAMES:
            result[name] = value
            continue
        if upper.startswith(("LSB_", "LSF_")):
            continue
        if upper in SAFE_SECRET_NAMES:
            result[name] = value
            continue
        if upper in BLOCKED_EXACT_NAMES or upper.startswith(BLOCKED_PREFIXES):
            continue
        if any(part in upper for part in BLOCKED_NAME_PARTS):
            continue
        result[name] = value
    return result


def atomic_request(path: Path, value: dict[str, Any]) -> None:
    """Publish one request atomically with owner-only permissions."""

    temporary = path.with_suffix(f".tmp.{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def main() -> int:
    """Submit one blaunch call, stream its output, and return its status."""

    if len(sys.argv) < 4:
        sys.stderr.write("usage: blaunch_proxy_client.py -z HOST COMMAND ...\n")
        return 2
    root = Path(os.environ.get("BLAUNCH_PROXY_ROOT", "/blaunch-proxy"))
    ready = root / "READY.json"
    if not ready.is_file():
        sys.stderr.write(f"blaunch proxy is not ready: {ready}\n")
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
    atomic_request(request, {"argv": sys.argv[1:], "env": forwarded_environment()})

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
    if cancelled:
        return 143
    return int(result["returncode"])


if __name__ == "__main__":
    raise SystemExit(main())
