"""Participant-owned scale-general learning-rate schedule.

The starter exactly reproduces the locked WSD control. Participants may edit
only this file.
"""

from __future__ import annotations


class WsdSchedule:
    """The executable 10%-warmup, 70%-stable, 20%-decay control."""

    def multiplier(self, progress: float) -> float:
        if progress < 0.1:
            return progress / 0.1
        if progress <= 0.8:
            return 1.0
        return max(0.0, (1.0 - progress) / 0.2)


def build_schedule() -> WsdSchedule:
    return WsdSchedule()


__all__ = ["build_schedule"]
