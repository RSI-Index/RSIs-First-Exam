#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TASKS = [
    ("gsm8k", 100),
    ("math500", 100),
    ("aime25", 30),
    ("humaneval", 100),
    ("livecodebench", 100),
]
GATE_DIR = Path("/logs/verifier/deepspec_output_gate")


def load_candidate_eval():
    sys.path.insert(0, "/app/project")
    spec = importlib.util.spec_from_file_location("candidate_deepspec_eval", "/app/project/eval.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load candidate eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_output_gate(module, local_rank: int) -> None:
    """Audit the actual target-verification probabilities used by DSpark.

    A separate full-sequence teacher-forced replay is not a bit-exact oracle for
    cached bf16 generation: attention reduction order can move near-tied logits.
    Instead, this wrapper checks every token committed by the protected official
    verifier against the target probabilities from that same cached forward.
    At temperature zero those probabilities are one-hot, so every committed
    token must have probability one.  A separate target ``generate`` call is
    retained on one sample per rank/task as a diagnostic, never as the gate.
    """
    evaluator_classes = set(module.EVALUATORS.values())
    base_classes = {
        base
        for evaluator in evaluator_classes
        for base in evaluator.__mro__
        if base.__name__ == "BaseEvaluator"
    }
    if len(base_classes) != 1:
        raise RuntimeError("could not identify the official BaseEvaluator")
    base = base_classes.pop()
    base_module = sys.modules[base.__module__]
    original_verify_draft_tokens = base_module.verify_draft_tokens
    verification_state = {
        "calls": 0,
        "tokens": 0,
        "active_selected_probabilities": [],
    }

    def trusted_verify_draft_tokens(*args, **kwargs):
        result = original_verify_draft_tokens(*args, **kwargs)
        committed = result.committed_tokens
        if committed is None or committed.ndim != 2 or committed.shape[0] != 1:
            raise RuntimeError("official verifier did not expose committed tokens")
        target_probs = result.target_probs[:, : committed.shape[1], :]
        if target_probs.shape[:2] != committed.shape:
            raise RuntimeError(
                "official verifier target-probability alignment is invalid: "
                f"probs={list(target_probs.shape)} committed={list(committed.shape)}"
            )
        selected = target_probs.gather(-1, committed.unsqueeze(-1)).squeeze(-1)
        verification_state["calls"] += 1
        verification_state["tokens"] += int(committed.numel())
        # Keep only the tiny gathered tensor, not the full vocabulary
        # distribution.  Do not call .item() here: a per-verification GPU sync
        # inside the timed decode would bias policies with different schedules.
        verification_state["active_selected_probabilities"].append(selected.detach())
        return result

    base_module.verify_draft_tokens = trusted_verify_draft_tokens
    original_run_dataset = base.run_dataset

    def trusted_run_dataset(self, *, dataset_name, max_samples):
        self._trusted_dataset_name = dataset_name
        self._trusted_gate_remaining = 1
        return original_run_dataset(self, dataset_name=dataset_name, max_samples=max_samples)

    base.run_dataset = trusted_run_dataset
    seen: set[type] = set()
    for evaluator in evaluator_classes:
        if evaluator in seen:
            continue
        seen.add(evaluator)
        original_generate = evaluator.generate_one_sample

        def trusted_generate(self, *, input_ids, stop_token_ids, _original=original_generate):
            before_calls = int(verification_state["calls"])
            before_tokens = int(verification_state["tokens"])
            verification_state["active_selected_probabilities"] = []
            torch.cuda.synchronize(device=self.device)
            decode_started = time.perf_counter()
            result = _original(self, input_ids=input_ids, stop_token_ids=stop_token_ids)
            torch.cuda.synchronize(device=self.device)
            decode_seconds = time.perf_counter() - decode_started
            diagnostic_checked = int(getattr(self, "_trusted_gate_remaining", 0)) > 0
            verified_calls = int(verification_state["calls"]) - before_calls
            verified_tokens = int(verification_state["tokens"]) - before_tokens
            selected_tensors = verification_state["active_selected_probabilities"]
            if selected_tensors:
                selected_probabilities = torch.cat(selected_tensors, dim=1)
                invalid_tokens = int(
                    (selected_probabilities < (1.0 - 1e-6)).sum().item()
                )
                max_probability_gap = float(
                    (1.0 - selected_probabilities).max().item()
                )
            else:
                invalid_tokens = 0
                max_probability_gap = 0.0
            target_verification_valid = (
                invalid_tokens == 0
                and verified_tokens >= max(0, int(result.num_output_tokens) - 1)
                and (verified_calls > 0 or int(result.num_output_tokens) <= 1)
            )
            exact = None
            tie_equivalent = False
            if diagnostic_checked:
                self._trusted_gate_remaining -= 1
                expected = self.target_model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=int(self.args.max_new_tokens),
                    do_sample=False,
                    eos_token_id=stop_token_ids,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    use_cache=True,
                )
                exact = bool(torch.equal(result.output_ids, expected))
                actual_ids = result.output_ids[0].tolist()
                expected_ids = expected[0].tolist()
                shared = min(len(actual_ids), len(expected_ids))
                first_mismatch = next(
                    (idx for idx in range(shared) if actual_ids[idx] != expected_ids[idx]),
                    shared if len(actual_ids) != len(expected_ids) else None,
                )
                mismatch_logits = None
                if first_mismatch is not None and first_mismatch < shared:
                    with torch.inference_mode():
                        reference_output = self.target_model(
                            input_ids=expected[:, :first_mismatch],
                            use_cache=False,
                            logits_to_keep=1,
                        )
                    next_logits = reference_output.logits[0, -1].float()
                    top_values, top_tokens = torch.topk(next_logits, k=5)
                    mismatch_logits = {
                        "actual_token": actual_ids[first_mismatch],
                        "expected_token": expected_ids[first_mismatch],
                        "actual_logit": float(next_logits[actual_ids[first_mismatch]].item()),
                        "expected_logit": float(next_logits[expected_ids[first_mismatch]].item()),
                        "top_tokens": top_tokens.tolist(),
                        "top_logits": top_values.tolist(),
                    }
                    tie_equivalent = (
                        mismatch_logits["actual_logit"]
                        == mismatch_logits["expected_logit"]
                    )
            record = {
                "rank": local_rank,
                "dataset": getattr(self, "_trusted_dataset_name", "unknown"),
                "output_tokens": int(result.num_output_tokens),
                "decode_seconds": decode_seconds,
                "gate_checked": True,
                "greedy_diagnostic_checked": diagnostic_checked,
                "greedy_exact": exact,
                "target_logit_tie": tie_equivalent,
                "target_verification_calls": verified_calls,
                "target_verification_tokens": verified_tokens,
                "target_verification_invalid_tokens": invalid_tokens,
                "target_verification_max_probability_gap": max_probability_gap,
                "target_verification_valid": target_verification_valid,
            }
            if diagnostic_checked:
                record.update(
                    {
                        "expected_output_tokens": len(expected_ids) - input_ids.shape[1],
                        "first_mismatch": first_mismatch,
                        "actual_tail": actual_ids[-16:],
                        "expected_tail": expected_ids[-16:],
                        "mismatch_logits": mismatch_logits,
                    }
                )
            with (GATE_DIR / f"rank_{local_rank}.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            if not target_verification_valid:
                raise RuntimeError(
                    f"target-verification gate failed on {record['dataset']} rank {local_rank}"
                )
            return result

        evaluator.generate_one_sample = trusted_generate


def trusted_worker(local_rank: int, args) -> None:
    os.chdir("/app/project")
    module = load_candidate_eval()
    install_output_gate(module, local_rank)
    module.main(local_rank, args)


def read_scalars(root: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    event_files = list(root.rglob("events.out.tfevents.*"))
    if not event_files:
        raise RuntimeError("official evaluator did not emit TensorBoard metrics")
    for parent in sorted({path.parent for path in event_files}):
        accumulator = EventAccumulator(str(parent))
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            events = accumulator.Scalars(tag)
            if events:
                values[tag] = float(events[-1].value)
    return values


def main() -> None:
    provenance = json.loads(Path("/app/output/provenance.json").read_text())
    if provenance.get("base_model") != "Qwen/Qwen3-4B" or provenance.get("draft_model") != "deepseek-ai/dspark_qwen3_4b_block7":
        raise RuntimeError("fixed target/draft model contract violated")
    tb_dir = Path("/tmp/deepspec_tb")
    tb_dir.mkdir(parents=True, exist_ok=True)
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    for stale in GATE_DIR.glob("*.jsonl"):
        stale.unlink()
    args = SimpleNamespace(
        target_name_or_path="Qwen/Qwen3-4B",
        draft_name_or_path="deepseek-ai/dspark_qwen3_4b_block7",
        max_new_tokens=int(os.getenv("DEEPSPEC_MAX_NEW_TOKENS", "1024")),
        temperature=0.0,
        confidence_threshold=0.0,
        tensorboard_dir=str(tb_dir),
        step=0,
        seed=980406,
        tasks=TASKS,
    )
    started = time.perf_counter()
    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("DeepSpec evaluation requires at least one CUDA device")
    torch.multiprocessing.spawn(trusted_worker, args=(args,), nprocs=gpu_count)
    elapsed = time.perf_counter() - started
    scalars = read_scalars(tb_dir)
    accept = {tag: value for tag, value in scalars.items() if tag.endswith("/accept_length")}
    verify = {tag: value for tag, value in scalars.items() if tag.endswith("/verify_rate")}
    if len(accept) != len(TASKS) or not verify:
        raise RuntimeError(f"incomplete official metrics: accept={accept.keys()} verify={verify.keys()}")
    records = [
        json.loads(line)
        for path in sorted(GATE_DIR.glob("rank_*.jsonl"))
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    checked = [record for record in records if record["greedy_diagnostic_checked"]]
    expected_checked_pairs = {
        (rank, task_name)
        for rank in range(gpu_count)
        for task_name, _ in TASKS
    }
    actual_checked_pairs = {
        (int(record["rank"]), str(record["dataset"])) for record in checked
    }
    if (
        not records
        or len(checked) != len(expected_checked_pairs)
        or actual_checked_pairs != expected_checked_pairs
        or not all(record["target_verification_valid"] for record in records)
    ):
        missing = sorted(expected_checked_pairs - actual_checked_pairs)
        unexpected = sorted(actual_checked_pairs - expected_checked_pairs)
        raise RuntimeError(
            "trusted target-verification records or independent diagnostics are "
            "missing, duplicated, or failed: "
            f"expected={len(expected_checked_pairs)} actual={len(checked)} "
            f"missing={missing} unexpected={unexpected}"
        )
    generated_tokens = sum(int(record["output_tokens"]) for record in records)
    decode_seconds_by_rank = {
        rank: sum(float(record["decode_seconds"]) for record in records if record["rank"] == rank)
        for rank in sorted({int(record["rank"]) for record in records})
    }
    decode_wall_seconds = max(decode_seconds_by_rank.values(), default=0.0)
    if decode_wall_seconds <= 0:
        raise RuntimeError("trusted decode timing is missing or invalid")
    metrics = {
        "end_to_end_wall_seconds": elapsed,
        "decode_wall_seconds": decode_wall_seconds,
        "decode_seconds_by_rank": decode_seconds_by_rank,
        "generated_tokens": generated_tokens,
        "generated_tokens_per_second": generated_tokens / decode_wall_seconds,
        "mean_acceptance_length": sum(accept.values()) / len(accept),
        "mean_verify_rate": sum(verify.values()) / len(verify),
        "target_verification_calls": sum(int(record["target_verification_calls"]) for record in records),
        "target_verification_token_checks": sum(int(record["target_verification_tokens"]) for record in records),
        "target_verification_invalid_tokens": sum(int(record["target_verification_invalid_tokens"]) for record in records),
        "target_verification_valid": True,
        "independent_greedy_diagnostic_checks": len(checked),
        "independent_greedy_exact_matches": sum(bool(record["greedy_exact"]) for record in checked),
        "target_logit_tie_checks": sum(bool(record["target_logit_tie"]) for record in checked),
        "tasks": [name for name, _ in TASKS],
        "temperature": 0.0,
        "seed": 980406,
    }
    Path("/logs/verifier/metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
