"""Editable deterministic hook for building one Magpie training dataset."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
from typing import Protocol


class GeneratorProtocol(Protocol):
    def generate(self, prompts: list[str], sampling: Mapping[str, object]) -> list[str]:
        """Generate one answer for every supplied prompt."""


def _pipeline_options(config: Mapping[str, object]) -> tuple[int, bool, bool, int]:
    if not isinstance(config.get("strategy", "baseline_prefix"), str) or config.get("strategy", "baseline_prefix") != "baseline_prefix":
        raise ValueError("pipeline strategy must be baseline_prefix")
    max_records = config.get("max_records", 50_000)
    if not isinstance(max_records, int) or isinstance(max_records, bool) or not 0 < max_records <= 300_000:
        raise ValueError("max_records must be an integer between 1 and 300000")
    include_baseline = config.get("include_baseline", True)
    generate_new = config.get("generate_new", False)
    seed = config.get("seed", 42)
    if not isinstance(include_baseline, bool) or not isinstance(generate_new, bool):
        raise ValueError("include_baseline and generate_new must be booleans")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not include_baseline and not generate_new:
        raise ValueError("pipeline must include baseline rows or generate new rows")
    return max_records, include_baseline, generate_new, seed


def _ranked_baseline(rows: Iterable[dict], max_records: int, seed: int) -> list[dict]:
    materialized = list(rows)
    for item in materialized:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError("baseline rows must be objects with string ids")
    return sorted(
        materialized,
        key=lambda item: hashlib.sha256(item["id"].encode("utf-8")).hexdigest(),
    )[:max_records]


def _prompt(record: dict) -> str:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations or not isinstance(conversations[0], dict):
        raise ValueError("baseline row must contain conversations")
    prompt = conversations[0].get("value")
    if not isinstance(prompt, str):
        raise ValueError("baseline prompt must be a string")
    return prompt


def build_dataset(
    config: Mapping[str, object], baseline_rows: Iterable[dict], generator: GeneratorProtocol
) -> Iterable[dict]:
    """Return the participant-editable candidate rows for a single attempt.

    The frozen starter only ranks and truncates baseline rows.  A local generator is
    called only when the participant opts into ``generate_new``.
    """

    if not isinstance(config, Mapping):
        raise ValueError("pipeline config must be a mapping")
    max_records, include_baseline, generate_new, seed = _pipeline_options(config)
    ranked = _ranked_baseline(baseline_rows, max_records, seed)
    selected: list[dict] = list(ranked) if include_baseline else []
    if generate_new:
        prompts = [_prompt(item) for item in ranked]
        sampling: dict[str, object] = {"seed": seed, "max_tokens": 2048}
        responses = generator.generate(prompts, sampling)
        if not isinstance(responses, list) or len(responses) != len(prompts) or not all(isinstance(item, str) for item in responses):
            raise ValueError("generator must return one string response per prompt")
        for prompt, response in zip(prompts, responses):
            identifier = hashlib.sha256((prompt + "\0" + response).encode("utf-8")).hexdigest()
            selected.append(
                {
                    "id": f"generated-{identifier}",
                    "conversations": [{"from": "human", "value": prompt}, {"from": "gpt", "value": response}],
                }
            )
    return selected[:max_records]
