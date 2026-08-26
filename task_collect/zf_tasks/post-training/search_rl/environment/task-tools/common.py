#!/usr/bin/env python3
"""Shared validation and JSON helpers for search-RL Task Tools."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict


ATTEMPT_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")


def validate_attempt_id(value: str) -> str:
    if not ATTEMPT_PATTERN.fullmatch(value) or ".." in value:
        raise ValueError("attempt ID must be 1-64 safe lowercase characters without '..'")
    return value


def output_root(raw: str | None) -> Path:
    return Path(raw or os.environ.get("TASK_OUTPUT_ROOT", "/app/output")).resolve()


def require_within(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} escapes {root}")
    return resolved


def read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
