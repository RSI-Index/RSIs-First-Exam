"""Trusted runtime for stateless, normalized-progress LR schedules."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Protocol

try:
    from megatron.bridge.training.callbacks import Callback
except ImportError:  # Unit tests load this module outside the training image.
    class Callback:  # type: ignore[no-redef]
        """Minimal import-time fallback; the real image supplies Callback."""


class Schedule(Protocol):
    """Participant surface: one deterministic function of normalized progress."""

    def multiplier(self, progress: float) -> float: ...


def evaluate_schedule(schedule: Schedule, progress: float) -> float:
    """Evaluate participant code and fail closed outside the locked LR envelope."""

    normalized = float(progress)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError("schedule progress must be finite and in [0, 1]")
    multiplier = float(schedule.multiplier(normalized))
    if not math.isfinite(multiplier):
        raise ValueError("schedule multiplier must be finite")
    if not 0.0 <= multiplier <= 1.0:
        raise ValueError("schedule multiplier must be in [0, 1]")
    return multiplier


def _assign_lr(group: dict[str, Any], value: float) -> None:
    current = group.get("lr")
    if hasattr(current, "fill_"):
        current.fill_(value)
    else:
        group["lr"] = value


class ScheduleHarness:
    """Apply a stateless schedule to fresh locked scheduler LRs."""

    def __init__(self, schedule: Schedule) -> None:
        self.schedule = schedule

    def apply(
        self,
        optimizer: Any,
        *,
        progress: float,
    ) -> dict[str, Any]:
        multiplier = evaluate_schedule(self.schedule, progress)
        try:
            peak_lrs = [float(group["max_lr"]) for group in optimizer.param_groups]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("every locked optimizer group must expose max_lr") from error
        if any(not math.isfinite(value) or value <= 0.0 for value in peak_lrs):
            raise ValueError("locked peak learning rates must be finite and positive")
        effective_lrs = [value * multiplier for value in peak_lrs]
        for group, value in zip(optimizer.param_groups, effective_lrs, strict=True):
            _assign_lr(group, value)
        return {
            "progress": float(progress),
            "peak_lrs": peak_lrs,
            "multiplier": multiplier,
            "effective_lrs": effective_lrs,
        }


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trusted_source_inventory_sha256() -> str:
    """Return the immutable full-project hash supplied by the trusted runner."""

    value = os.environ.get("LR_SCHEDULE_SOURCE_INVENTORY_SHA256", "")
    if not value:
        raise RuntimeError("trusted source inventory hash is missing")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError("trusted source inventory hash must be lowercase SHA-256")
    return value


def validate_stateless_resume(
    *,
    checkpoint_iteration: int,
    expected_iteration: int,
    checkpoint_source_sha256: str,
    expected_source_sha256: str,
) -> None:
    """Validate resume provenance; a stateless schedule has no sidecar to restore."""

    if int(checkpoint_iteration) != int(expected_iteration):
        raise ValueError("schedule resume iteration mismatch")
    if checkpoint_source_sha256 != expected_source_sha256:
        raise ValueError("schedule resume source hash mismatch")


def append_schedule_trace(path: Path, row: dict[str, Any]) -> None:
    required = {
        "iteration",
        "progress",
        "peak_lrs",
        "multiplier",
        "effective_lrs",
        "source_sha256",
    }
    if not required.issubset(row):
        raise ValueError(f"schedule trace is missing {sorted(required - set(row))}")
    if "state" in row or "diagnostics" in row:
        raise ValueError("stateless schedule traces may not contain state or diagnostics")
    multiplier = float(row["multiplier"])
    if not math.isfinite(multiplier) or not 0.0 <= multiplier <= 1.0:
        raise ValueError("schedule trace multiplier must be in [0, 1]")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _nested(source: Any, *names: str) -> Any:
    result = source
    for name in names:
        result = getattr(result, name, None)
        if result is None:
            break
    return result


class LearningRateScheduleCallback(Callback):
    """Apply one frozen schedule before each optimizer update."""

    def __init__(
        self,
        schedule: Schedule,
        *,
        candidate_path: Path,
        output_dir: Path,
        train_iters: int,
    ) -> None:
        self.candidate_sha256 = source_sha256(candidate_path.resolve())
        self.source_inventory_sha256 = trusted_source_inventory_sha256()
        self.output_dir = output_dir.resolve()
        self.train_iters = int(train_iters)
        if self.train_iters <= 0:
            raise ValueError("train_iters must be positive")
        self.harness = ScheduleHarness(schedule)

    def _iteration(self, context: Any) -> int:
        return int(_nested(context, "state", "train_state", "step") or 0)

    def on_train_step_start(self, context: Any) -> None:
        iteration = self._iteration(context)
        progress = min(1.0, max(0.0, float(iteration) / float(self.train_iters)))
        row = self.harness.apply(
            context.optimizer,
            progress=progress,
        )
        row.update(
            {
                "iteration": iteration + 1,
                "source_sha256": self.source_inventory_sha256,
                "candidate_source_sha256": self.candidate_sha256,
            }
        )
        append_schedule_trace(self.output_dir / "lr_schedule_trace.jsonl", row)


__all__ = [
    "LearningRateScheduleCallback",
    "Schedule",
    "ScheduleHarness",
    "append_schedule_trace",
    "evaluate_schedule",
    "source_sha256",
    "trusted_source_inventory_sha256",
    "validate_stateless_resume",
]
