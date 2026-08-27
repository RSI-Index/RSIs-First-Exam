#!/usr/bin/env python3
"""Validate the fixed 32-node, 256-GPU outer LSF allocation."""

from __future__ import annotations

import sys


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_outer_allocation.py LSB_MCPU_HOSTS")

    fields = sys.argv[1].split()
    if not fields or len(fields) % 2:
        raise SystemExit("error: malformed LSB_MCPU_HOSTS in 256-GPU allocation")

    hosts: dict[str, int] = {}
    order: list[str] = []
    for index in range(0, len(fields), 2):
        host, slots = fields[index], int(fields[index + 1])
        if host not in hosts:
            order.append(host)
            hosts[host] = 0
        hosts[host] += slots

    if len(order) != 32 or any(hosts[host] < 8 for host in order):
        raise SystemExit(
            "error: expected 32 hosts with at least 8 slots each, got "
            f"{[(host, hosts[host]) for host in order]}"
        )

    print("outer allocation: 32 nodes x 8 GPUs = 256 GPUs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
