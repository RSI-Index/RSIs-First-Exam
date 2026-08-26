#!/usr/bin/env python3
"""Evaluate base, candidate, and released GRPO with frozen Search-R1 semantics."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.request
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from transformers import AutoTokenizer

sys.path.insert(0, "/task-tools")
sys.path.insert(0, "/opt/slime/examples/search-r1")

from async_contract import validate_attempt_id
from evaluation_contract import (
    aggregate_scores,
    build_bm25_command,
    build_sglang_command,
    reward_label,
    stage_candidate_checkpoint,
    validate_checkpoint_tree,
)
from generate_with_search import generate as search_r1_generate
from generate_with_search import reward_func as search_r1_reward
import slime.utils.http_utils as slime_http_utils
from slime.utils.http_utils import init_http_client
from slime.utils.types import Sample
from task_contract import (
    EVAL_DATA_PATH,
    EVAL_RECORDS,
    MODEL_PATH,
    ROLLOUT_MAX_RESPONSE_LEN,
    ROLLOUT_TEMPERATURE,
    SEED,
)


SERVER_PORT = 30000
RETRIEVER_URL = "http://127.0.0.1:8000/retrieve"


def _prompt_text(tokenizer, prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list) and prompt and all(isinstance(message, dict) for message in prompt):
        return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    raise ValueError("evaluation prompt has an unsupported shape")


def _runtime_args() -> SimpleNamespace:
    return SimpleNamespace(
        partial_rollout=False,
        hf_checkpoint=MODEL_PATH,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=SERVER_PORT,
        sglang_server_concurrency=64,
        rollout_num_engines=1,
        rollout_num_gpus=2,
        rollout_num_gpus_per_engine=2,
        rollout_temperature=ROLLOUT_TEMPERATURE,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_max_response_len=ROLLOUT_MAX_RESPONSE_LEN,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False,
        sglang_dp_size=1,
        use_distributed_post=False,
    )


async def close_http_client() -> None:
    client = slime_http_utils._http_client
    if client is not None:
        await client.aclose()
    slime_http_utils._http_client = None
    slime_http_utils._client_concurrency = 0


async def _evaluate_row(
    args: SimpleNamespace,
    semaphore: asyncio.Semaphore,
    prompt: str,
    label: dict[str, Any],
    row_index: int,
) -> float:
    sampling_params = {
        "temperature": ROLLOUT_TEMPERATURE,
        "top_p": 1.0,
        "top_k": -1,
        "max_new_tokens": ROLLOUT_MAX_RESPONSE_LEN,
        "skip_special_tokens": False,
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
        "sampling_seed": SEED,
    }
    sample = Sample(index=row_index, rollout_id=row_index, prompt=prompt, label=label)
    async with semaphore:
        sample = await search_r1_generate(args, sample, sampling_params)
        score = await search_r1_reward(args, sample)
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("Search-R1 reward is not numeric")
    return float(score)


async def _evaluate_model(records: list[dict[str, Any]], tokenizer) -> float:
    prompts = [(_prompt_text(tokenizer, record["prompt"]), reward_label(record)) for record in records]
    args = _runtime_args()
    init_http_client(args)
    semaphore = asyncio.Semaphore(64)
    try:
        scores = await asyncio.gather(
            *(
                _evaluate_row(args, semaphore, prompt, label, index)
                for index, (prompt, label) in enumerate(prompts)
            )
        )
        return aggregate_scores(scores)
    finally:
        await close_http_client()


def _wait_for_server(process: subprocess.Popen, timeout: float = 900.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"SGLang server exited with status {process.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{SERVER_PORT}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise TimeoutError("SGLang server did not become ready")


def _wait_for_retriever(process: subprocess.Popen, timeout: float = 900.0) -> None:
    deadline = time.monotonic() + timeout
    request = urllib.request.Request(
        RETRIEVER_URL,
        data=b'{"queries":["test"],"topk":3,"return_scores":false}',
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"BM25 server exited with status {process.returncode}")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                payload = json.loads(response.read())
                rows = payload.get("result")
                if response.status == 200 and isinstance(rows, list) and len(rows) == 1 and len(rows[0]) == 3:
                    return
        except Exception:
            time.sleep(2)
    raise TimeoutError("BM25 server did not become ready with top-k 3")


def _terminate(process: subprocess.Popen, timeout: float = 60.0) -> None:
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def evaluate(model_path: Path, records: list[dict[str, Any]], tokenizer, log_path: Path) -> float:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "0,1"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            build_sglang_command(model_path, SERVER_PORT),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_for_server(process)
            return asyncio.run(_evaluate_model(records, tokenizer))
        finally:
            _terminate(process)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/app/output/verifier/aggregates.json"))
    args = parser.parse_args()
    base = Path("/models/Qwen2.5-3B")
    selection = json.loads(Path("/app/output/selected-attempt.json").read_text(encoding="utf-8"))
    selected_attempt = validate_attempt_id(selection.get("selected_attempt"))
    candidate_source = Path("/var/lib/slime-task/runs") / selected_attempt / "train" / "hf_model"
    reference = Path("/models/reference/SearchR1-GRPO")
    expected_config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    validate_checkpoint_tree(base, expected_config)
    validate_checkpoint_tree(reference, expected_config)
    candidate = stage_candidate_checkpoint(
        candidate_source,
        args.output.parent / "candidate-snapshot",
        base,
        expected_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    records = pq.read_table(EVAL_DATA_PATH).to_pylist()
    if len(records) != EVAL_RECORDS:
        raise ValueError(f"hidden evaluation must contain exactly {EVAL_RECORDS} records")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    aggregates = {}
    with (args.output.parent / "bm25.log").open("w", encoding="utf-8") as retriever_log:
        retriever = subprocess.Popen(
            build_bm25_command(),
            stdout=retriever_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_for_retriever(retriever)
            for label, checkpoint in (("base", base), ("candidate", candidate), ("reference", reference)):
                aggregates[label] = evaluate(
                    checkpoint,
                    records,
                    tokenizer,
                    args.output.parent / f"sglang-{label}.log",
                )
        finally:
            _terminate(retriever, timeout=30)
    args.output.write_text(json.dumps(aggregates, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
