#!/usr/bin/env python3
"""Sudo allowlist entry: enter a private network namespace, then supervise."""

from __future__ import annotations

import argparse
import os
import re


ATTEMPT_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args()
    if os.geteuid() != 0:
        raise PermissionError("supervisor entry must run as root")
    if not ATTEMPT_PATTERN.fullmatch(args.attempt_id):
        raise ValueError("invalid attempt id")
    environment = {
        "HOME": "/root",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PYTHONPATH": "/task-tools:/opt/slime:/opt/Megatron-LM",
    }
    os.execve(
        "/usr/bin/unshare",
        [
            "unshare",
            "--net",
            "--",
            "/task-tools/root_supervisor.py",
            "--attempt-id",
            args.attempt_id,
        ],
        environment,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
