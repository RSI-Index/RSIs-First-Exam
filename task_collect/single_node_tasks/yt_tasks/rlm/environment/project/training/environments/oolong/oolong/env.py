"""OOLONG synth long-context QA, wired through RLMTrainEnv."""

from __future__ import annotations

import ast
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import dateutil.parser
import rlm_train
import verifiers as vf
from datasets import Dataset, load_dataset

COMPARISON_PHRASES = ("more common than", "less common than", "same frequency as")
OOLONG_DATASET_PATH = os.environ.get(
    "OOLONG_DATASET_PATH", "oolongbench/oolong-synth"
)


def _load_oolong_validation(*, streaming: bool):
    if OOLONG_DATASET_PATH == "oolongbench/oolong-synth":
        return load_dataset(OOLONG_DATASET_PATH, split="validation", streaming=streaming)
    validation_glob = str(Path(OOLONG_DATASET_PATH) / "data" / "validation-*.parquet")
    return load_dataset(
        "parquet",
        data_files={"validation": validation_glob},
        split="validation",
        streaming=streaming,
    )


def _find_comparison_phrase(output: str) -> str | None:
    out_low = output.lower()
    hits = [(out_low.rfind(p), p) for p in COMPARISON_PHRASES if p in out_low]
    return max(hits)[1] if hits else None


def _attempt_answer_parse(answer: str) -> tuple[str, str]:
    cmp = _find_comparison_phrase(answer)
    if cmp is not None:
        return cmp, "high"
    if ":" not in answer:
        if len(answer) < 20:
            return answer, "low"
        return answer.split()[-1], "low"
    cand = answer.split(":")[-1].strip().replace("*", "").replace("[", "").replace("]", "")
    if len(cand) < 20:
        return cand, "vhigh"
    return cand, "med"


def _synth_score(datapoint: dict, output: str) -> float:
    answer = str(datapoint.get("answer", ""))
    try:
        if "datetime" in answer:
            gold: Any = datetime.strptime(answer, "[datetime.date(%Y, %m, %d)]")
        else:
            gold = ast.literal_eval(answer)[0]
    except Exception:
        gold = answer

    trimmed, _ = _attempt_answer_parse(output)
    gold_s = str(gold)

    if str(trimmed) == gold_s:
        return 1.0
    if str(trimmed).lower() == gold_s.lower():
        return 1.0

    atype = datapoint.get("answer_type", "")
    if atype == "ANSWER_TYPE.NUMERIC":
        try:
            return 0.75 ** abs(int(gold) - int(trimmed))
        except Exception:
            return 0.0
    if atype == "ANSWER_TYPE.DATE":
        try:
            return 1.0 if dateutil.parser.parse(trimmed) == gold else 0.0
        except Exception:
            return 0.0

    if gold_s and gold_s.lower() not in [p.lower() for p in COMPARISON_PHRASES]:
        if gold_s.lower() in output.lower():
            return 1.0

    return 0.0


async def _score(info, state: vf.State, **_kw: Any) -> float:
    final = state.get("rlm_final_answer") or state.get("final_answer") or ""
    meta = json.loads(info) if isinstance(info, str) else info
    return _synth_score(meta, final)


_QUESTION_INSTRUCTION = (
    "The context contains thousands of general-knowledge questions, one per "
    "line. Each line has a User ID and a question, and each question's answer "
    "falls into one of 6 categories: 'numeric value', 'entity', 'location', "
    "'description and abstract concept', 'abbreviation', 'human being'. "
    "Answer the following aggregate question."
)


def _build_dataset(
    *,
    dataset_name: str,
    min_ctx: int,
    max_ctx: int,
    num_examples: int,
    seed: int,
    exclude_numeric: bool,
) -> Dataset:
    def _keep(ex):
        if ex.get("dataset") != dataset_name:
            return False
        cl = ex.get("context_len", 0)
        if not (min_ctx <= cl <= max_ctx):
            return False
        if exclude_numeric and ex.get("answer_type") == "ANSWER_TYPE.NUMERIC":
            return False
        return True

    if num_examples > 0:
        stream = _load_oolong_validation(streaming=True)
        if seed is not None:
            stream = stream.shuffle(seed=seed, buffer_size=10_000)
        samples: list[dict] = []
        for ex in stream:
            if _keep(ex):
                samples.append(ex)
                if len(samples) >= num_examples:
                    break
    else:
        ds = _load_oolong_validation(streaming=False)
        samples = [ex for ex in ds if _keep(ex)]
        if seed is not None:
            random.Random(seed).shuffle(samples)

    rows: list[dict] = []
    for i, s in enumerate(samples):
        question = s["question"]
        context = s.get("context_window_text", s.get("context", ""))
        meta = {
            "id": s.get("id"),
            "dataset": s.get("dataset", ""),
            "answer_type": s.get("answer_type", ""),
            "answer": str(s.get("answer", "")),
            "context": context,
            "root_prompt": f"{_QUESTION_INSTRUCTION}\n\nQuestion: {question}",
        }
        rows.append(
            {
                "example_id": i,
                "prompt": [{"role": "user", "content": question}],
                "answer": str(s.get("answer", "")),
                "info": json.dumps(meta),
            }
        )
    return Dataset.from_list(rows)


def load_environment(
    *,
    dataset_name: str = "trec_coarse",
    min_ctx: int = 1024,
    max_ctx: int = 4096,
    num_examples: int = -1,
    seed: int = 42,
    exclude_numeric: bool = False,
    max_iterations: int = 12,
    sub_max_tokens: int = 4096,
    min_iterations: int = 2,
    min_subcall: int = 1,
    **kwargs: Any,
) -> vf.Environment:
    dataset = _build_dataset(
        dataset_name=dataset_name,
        min_ctx=min_ctx,
        max_ctx=max_ctx,
        num_examples=num_examples,
        seed=seed,
        exclude_numeric=exclude_numeric,
    )
    rubric = rlm_train.RLMTrainRubric(
        correctness=_score,
        weight=1.0,
        min_iterations=min_iterations,
        min_subcall=min_subcall,
    )
    return rlm_train.RLMTrainEnv(
        dataset=dataset,
        max_iterations=max_iterations,
        sub_sampling_args={"max_tokens": sub_max_tokens},
        rubric=rubric,
        **kwargs,
    )


__all__ = ["load_environment"]
