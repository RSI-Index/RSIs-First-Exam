"""Deterministic prompt selection and fail-closed Magpie reward helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import math


PROMPTS_PER_SOURCE = 256
TOTAL_PROMPTS = 768
_SCORES = frozenset({0.0, 0.5, 1.0})


def select_prompt_ids(rows: Iterable[object], source: str, count: int = PROMPTS_PER_SOURCE) -> list[str]:
    """Select exactly ``count`` unique English prompt identifiers by stable hash."""

    if not isinstance(source, str) or not source:
        raise ValueError("source must be a non-empty string")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("count must be a positive integer")
    eligible: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("prompt row must be a mapping")
        if row.get("source") != source:
            continue
        language = row.get("language")
        if not isinstance(language, str) or language.casefold() not in {"en", "english"}:
            continue
        stable_id = row.get("id")
        if not isinstance(stable_id, str) or not stable_id:
            raise ValueError("eligible prompt id must be a non-empty string")
        if stable_id in seen:
            raise ValueError("eligible prompt ids must be unique")
        seen.add(stable_id)
        eligible.append(stable_id)
    ranked = sorted(
        eligible,
        key=lambda stable_id: hashlib.sha256(f"{source}\0{stable_id}".encode("utf-8")).hexdigest(),
    )
    if len(ranked) < count:
        raise ValueError(f"source exhausted before selecting {count} unique eligible prompt ids")
    return ranked[:count]


def normalize_verdict(raw: object, candidate_label: str) -> float:
    """Map a schema-validated A/B/tie verdict to the candidate's preference score."""

    if not isinstance(candidate_label, str) or candidate_label.upper() not in {"A", "B"}:
        raise ValueError("candidate label must be A or B")
    if isinstance(raw, Mapping):
        if "winner" in raw:
            raw = raw["winner"]
        elif "verdict" in raw:
            raw = raw["verdict"]
        else:
            raise ValueError("verdict object must contain winner or verdict")
    if not isinstance(raw, str):
        raise ValueError("verdict must be A, B, or tie")
    verdict = raw.strip().casefold()
    if verdict in {"tie", "draw", "equal"}:
        return 0.5
    if verdict not in {"a", "b"}:
        raise ValueError("verdict must be A, B, or tie")
    return 1.0 if verdict.upper() == candidate_label.upper() else 0.0


def combine_orientations(first: float, second: float) -> float:
    """Keep an orientation result only when both position swaps agree."""

    try:
        normalized = (float(first), float(second))
    except (TypeError, ValueError) as exc:
        raise ValueError("orientation scores must be 0, 0.5, or 1") from exc
    if not all(math.isfinite(score) and score in _SCORES for score in normalized):
        raise ValueError("orientation scores must be 0, 0.5, or 1")
    return normalized[0] if normalized[0] == normalized[1] else 0.5


def aggregate_reward(scores: Sequence[float]) -> float:
    """Return the mean candidate score only for the complete 768-prompt suite."""

    if isinstance(scores, (str, bytes)) or len(scores) != TOTAL_PROMPTS:
        raise ValueError("aggregate reward requires exactly 768 scores")
    try:
        values = [float(score) for score in scores]
    except (TypeError, ValueError) as exc:
        raise ValueError("aggregate reward scores must be finite") from exc
    if not all(math.isfinite(score) and score in _SCORES for score in values):
        raise ValueError("aggregate reward scores must be finite win/tie/loss scores")
    return sum(values) / TOTAL_PROMPTS
