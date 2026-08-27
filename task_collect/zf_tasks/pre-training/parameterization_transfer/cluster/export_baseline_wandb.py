#!/usr/bin/env python3
"""Export compact, trusted AdamH cost/quality histories from the six W&B runs."""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import wandb


PROJECT = "RSI-Index/ale-rsi-pre-training"
WINDOW_TOKENS = 16_777_216
SCALES = {
    "E0": {"run_id": "5h5yihr2", "gbs": 16, "gpus": 8, "final_update": 44_317, "scoring_update": 44_317},
    "E1": {"run_id": "952nr4js", "gbs": 16, "gpus": 8, "final_update": 55_125, "scoring_update": 55_125},
    "E2": {"run_id": "3hz3kt5j", "gbs": 32, "gpus": 8, "final_update": 38_014, "scoring_update": 38_014},
    "E3": {"run_id": "lonea36p", "gbs": 64, "gpus": 32, "final_update": 40_283, "scoring_update": 40_283},
    "E4": {"run_id": "76xmgi2r", "gbs": 64, "gpus": 32, "final_update": 56_477, "scoring_update": 30_000},
    "E5": {"run_id": "aznfn0mh", "gbs": 128, "gpus": 128, "final_update": 35_510, "scoring_update": 35_000},
}
KEYS = [
    "samples vs steps",
    "lm loss",
    "_runtime",
    "_timestamp",
    "lm_eval/runtime_seconds",
    "lm_eval/longppl/runtime_seconds",
    "lm_eval/paloma/bits_per_byte",
    "lm_eval/paloma/macro_bits_per_byte",
]


def finite(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def export_scale(api: wandb.Api, scale: str, spec: dict[str, int | str]) -> dict[str, object]:
    run = api.run(f"{PROJECT}/{spec['run_id']}")
    rows = list(run.scan_history(keys=KEYS, page_size=10_000))
    gbs = int(spec["gbs"])
    gpus = int(spec["gpus"])
    window_steps = math.ceil(WINDOW_TOKENS / (gbs * 4096))
    losses: deque[float] = deque()
    loss_sum = 0.0
    first_runtime: float | None = None
    eval_seconds = 0.0
    compact_loss: list[dict[str, float | int]] = []
    paloma: list[dict[str, float | int]] = []
    last_iteration = 0
    last_raw_loss: float | None = None

    for row in rows:
        samples = finite(row.get("samples vs steps"))
        runtime = finite(row.get("_runtime"))
        loss = finite(row.get("lm loss"))
        if samples is None or runtime is None or loss is None:
            continue
        iteration = int(round(samples / gbs))
        if iteration <= 0 or iteration < last_iteration:
            continue
        last_iteration = iteration
        last_raw_loss = loss
        if first_runtime is None:
            first_runtime = runtime
        eval_seconds += finite(row.get("lm_eval/runtime_seconds")) or 0.0
        eval_seconds += finite(row.get("lm_eval/longppl/runtime_seconds")) or 0.0
        active_seconds = max(0.0, runtime - first_runtime - eval_seconds)
        cost = active_seconds * gpus

        losses.append(loss)
        loss_sum += loss
        while len(losses) > window_steps:
            loss_sum -= losses.popleft()
        window_loss = loss_sum / len(losses)

        micro = finite(row.get("lm_eval/paloma/bits_per_byte"))
        macro = finite(row.get("lm_eval/paloma/macro_bits_per_byte"))
        keep_loss = iteration % 100 == 0 or iteration in {
            int(spec["scoring_update"]), int(spec["final_update"])
        }
        if keep_loss:
            compact_loss.append(
                {
                    "iteration": iteration,
                    "cumulative_gpu_seconds": cost,
                    "fixed_window_loss": window_loss,
                }
            )
        if micro is not None and macro is not None:
            paloma.append(
                {
                    "iteration": iteration,
                    "cumulative_gpu_seconds": cost,
                    "bits_per_byte": micro,
                    "macro_bits_per_byte": macro,
                }
            )

    scoring_update = int(spec["scoring_update"])
    scoring_paloma = next((item for item in paloma if item["iteration"] == scoring_update), None)
    if scoring_paloma is None:
        raise RuntimeError(f"{scale} has no Paloma result at scoring update {scoring_update}")
    scoring_loss = min(compact_loss, key=lambda item: abs(int(item["iteration"]) - scoring_update))
    if int(scoring_loss["iteration"]) != scoring_update:
        raise RuntimeError(f"{scale} has no loss result at scoring update {scoring_update}")
    return {
        "model": run.name,
        "wandb_project": PROJECT,
        "wandb_run_id": str(spec["run_id"]),
        "wandb_state": run.state,
        "gpus": gpus,
        "global_batch_size": gbs,
        "final_update": int(spec["final_update"]),
        "scoring_update": scoring_update,
        "reference_status": "final" if scoring_update == int(spec["final_update"]) else "recorded_nonfinal_scoring_update",
        "paloma_bits_per_byte": scoring_paloma["bits_per_byte"],
        "paloma_macro_bits_per_byte": scoring_paloma["macro_bits_per_byte"],
        "scoring_gpu_seconds": scoring_paloma["cumulative_gpu_seconds"],
        "scoring_fixed_window_loss": scoring_loss["fixed_window_loss"],
        "last_raw_minibatch_loss": last_raw_loss,
        "fixed_window_tokens": WINDOW_TOKENS,
        "paloma_trajectory": paloma,
        "loss_cost_curve": compact_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    api = wandb.Api(timeout=120)
    payload = {
        "version": 2,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "cost_definition": "gpus * (wandb runtime minus startup and reported evaluation runtime)",
        "cost_limitation": "historical AdamH runs predate the candidate CUDA-event tracer; cost is active process time reconstructed from W&B",
        "scales": {scale: export_scale(api, scale, spec) for scale, spec in SCALES.items()},
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
