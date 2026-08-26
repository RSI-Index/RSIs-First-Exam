#!/usr/bin/env python3

"""Run trusted LSF blaunch requests for the contained Harbor environment."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
# These describe the *calling* Harbor container.  Reusing them while starting
# the host-side training container makes Apptainer try to reuse paths such as
# /app/home and stale bind/session metadata that do not exist on the host.
# The broker's trusted host values must win for process identity and launch
# state.  APPTAINER (the executable path) and the two explicitly approved
# APPTAINERENV_* credentials remain allowed.
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


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Write one JSON object atomically."""

    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n")
    os.replace(temporary, path)


def allocated_hosts(raw_hosts: str) -> set[str]:
    """Return the unique hosts in an LSF HOST/SLOTS allocation string."""

    fields = raw_hosts.split()
    if not fields or len(fields) % 2:
        raise ValueError("malformed LSB_MCPU_HOSTS")
    return {fields[index].split(".", 1)[0] for index in range(0, len(fields), 2)}


def safe_client_environment(raw: Any) -> dict[str, str]:
    """Filter candidate environment state before passing it to host blaunch."""

    if not isinstance(raw, dict):
        raise ValueError("request environment must be an object")
    result: dict[str, str] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise ValueError("request environment must contain strings")
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


class Broker:
    """Poll a private request directory and execute validated blaunch calls."""

    def __init__(self, root: Path, blaunch: Path) -> None:
        self.root = root
        self.blaunch = blaunch
        self.requests = root / "requests"
        self.working = root / "working"
        self.results = root / "results"
        self.heartbeats = root / "heartbeats"
        self.cancel = root / "cancel"
        self.stop_event = threading.Event()
        self.processes: dict[str, subprocess.Popen[bytes]] = {}
        self.process_lock = threading.Lock()
        self.hosts = allocated_hosts(os.environ["LSB_MCPU_HOSTS"])
        self.base_environment = {
            name: value
            for name, value in os.environ.items()
            if name.startswith(("LSB_", "LSF_"))
            or name in {"HOME", "LANG", "LOGNAME", "PATH", "SHELL", "USER"}
        }

    def initialize(self) -> None:
        """Create an empty private protocol directory."""

        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        for directory in (
            self.requests,
            self.working,
            self.results,
            self.heartbeats,
            self.cancel,
        ):
            directory.mkdir(mode=0o700, exist_ok=True)
        atomic_json(
            self.root / "READY.json",
            {"job_id": os.environ["LSB_JOBID"], "hosts": sorted(self.hosts)},
        )

    def validate_arguments(self, raw: Any) -> list[str]:
        """Allow a single allocated target host and an explicit command."""

        if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
            raise ValueError("request argv must be a string array")
        if len(raw) < 3 or raw[0] != "-z":
            raise ValueError("blaunch proxy requires: -z HOST COMMAND ...")
        requested = raw[1].split()
        if len(requested) != 1:
            raise ValueError("blaunch proxy accepts exactly one host per request")
        host = requested[0].split(".", 1)[0]
        if host not in self.hosts:
            raise ValueError(f"host is outside this allocation: {requested[0]}")
        return raw

    def terminate_process(self, request_id: str) -> None:
        """Terminate one active blaunch process and then force-kill it."""

        with self.process_lock:
            process = self.processes.get(request_id)
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()

    def handle(self, request_path: Path) -> None:
        """Execute one request and publish its output and return code."""

        request_id = request_path.stem
        output_path = self.results / f"{request_id}.output"
        status_path = self.results / f"{request_id}.status.json"
        try:
            payload = json.loads(request_path.read_text())
            argv = self.validate_arguments(payload.get("argv"))
            environment = self.base_environment.copy()
            environment.update(safe_client_environment(payload.get("env")))
            with output_path.open("wb", buffering=0) as output:
                process = subprocess.Popen(
                    [str(self.blaunch), *argv],
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                )
                with self.process_lock:
                    self.processes[request_id] = process
                while process.poll() is None:
                    heartbeat = self.heartbeats / request_id
                    stale = not heartbeat.exists() or time.time() - heartbeat.stat().st_mtime > 45
                    if self.stop_event.is_set() or (self.cancel / request_id).exists() or stale:
                        self.terminate_process(request_id)
                        break
                    time.sleep(0.25)
                returncode = process.wait()
            atomic_json(status_path, {"returncode": returncode})
        except Exception as error:  # The status file is the client-facing error boundary.
            with output_path.open("ab") as output:
                output.write(f"blaunch proxy error: {error}\n".encode())
            atomic_json(status_path, {"returncode": 125})
        finally:
            with self.process_lock:
                self.processes.pop(request_id, None)
            request_path.unlink(missing_ok=True)

    def run(self) -> None:
        """Serve requests until signalled by the outer allocation driver."""

        self.initialize()
        with ThreadPoolExecutor(max_workers=64, thread_name_prefix="blaunch") as executor:
            while not self.stop_event.is_set():
                for request in self.requests.glob("*.json"):
                    claimed = self.working / request.name
                    try:
                        os.replace(request, claimed)
                    except FileNotFoundError:
                        continue
                    executor.submit(self.handle, claimed)
                time.sleep(0.1)
            with self.process_lock:
                active = list(self.processes)
            for request_id in active:
                self.terminate_process(request_id)


def main() -> None:
    """Parse arguments and run the broker."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--blaunch", required=True, type=Path)
    args = parser.parse_args()
    if not args.blaunch.is_absolute() or not args.blaunch.is_file():
        raise SystemExit("--blaunch must be an absolute executable path")
    broker = Broker(args.root, args.blaunch)

    def stop(_signum: int, _frame: Any) -> None:
        broker.stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    broker.run()


if __name__ == "__main__":
    main()
