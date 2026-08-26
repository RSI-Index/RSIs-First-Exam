"""Attempt-scoped port reservation for local vLLM API servers."""

from __future__ import annotations

import os
import socket


def _port_block() -> tuple[int, int]:
    base_value = os.getenv("TMAX_VLLM_PORT_BASE")
    if base_value is None:
        return 0, 1
    count_value = os.getenv("TMAX_VLLM_PORT_COUNT", "64")
    try:
        base = int(base_value)
        count = int(count_value)
    except ValueError as error:
        raise ValueError("TMAX vLLM port settings must be integers") from error
    if base < 1024 or count < 1 or base + count > 65536:
        raise ValueError("TMAX vLLM port block is outside the usable TCP range")
    return base, count


def reserve_vllm_server_socket() -> socket.socket:
    """Bind and retain a listener, removing the find-free-port race."""
    base, count = _port_block()
    for port in range(base, base + count):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.bind(("127.0.0.1", port))
            listener.listen(socket.SOMAXCONN)
            listener.setblocking(False)
            return listener
        except OSError:
            listener.close()
    raise RuntimeError(f"no free vLLM server port in block {base}:{base + count - 1}")
