#!/usr/bin/env python3
"""Merge offline AdamH trajectories with exact-final E4/E5 Paloma results."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT = "RSI-Index/ale-rsi-pre-training"
BASE_RUNTIME_COMMIT = "accefd7b3a448ab45b015edc04c9d3b70f7cb3b7"
TASK_INFRASTRUCTURE_COMMIT = "3a6c6fb3811a10f532d26be7dd032457795a8e41"
MCORE_COMMIT = "58bf14e9e68915a5e0c7d70451620e6e713c07d5"
SCALES: dict[str, dict[str, int]] = {
    "E0": {"parameters": 550_337_664, "tokens": 2_904_358_912, "iterations": 44_317, "gbs": 16, "gpus": 8},
    "E1": {"parameters": 837_007_744, "tokens": 3_612_672_000, "iterations": 55_125, "gbs": 16, "gpus": 8},
    "E2": {"parameters": 998_036_992, "tokens": 4_982_571_008, "iterations": 38_014, "gbs": 32, "gpus": 8},
    "E3": {"parameters": 1_384_584_448, "tokens": 10_559_946_752, "iterations": 40_283, "gbs": 64, "gpus": 32},
    "E4": {"parameters": 1_934_716_160, "tokens": 14_805_106_688, "iterations": 56_477, "gbs": 64, "gpus": 32},
    "E5": {"parameters": 2_544_614_912, "tokens": 18_617_466_880, "iterations": 35_510, "gbs": 128, "gpus": 128},
}
RECORDED_TRAINING_GPUS = {"E0": 8, "E1": 8, "E2": 8, "E3": 8, "E4": 8, "E5": 16}
ENDPOINT_TRAINING = {
    "E4": {
        "cumulative_gpu_seconds": 1_946_380.994036227,
        "fixed_window_loss": 2.6527315489947796,
        "last_raw_minibatch_loss": 2.8462119102478027,
        "loss_window_updates": 64,
        "checkpoint": "/proj/datasets/interns/yuetai/agent_envs/more_task/runs/marin-adamh-baseline/20260729T182645Z-marin-adamh-1p8e20-p1935m-t14p805b-d2048-l21-8gpu-mbs2-ga4-llama3-b64-norecompute-nooffload-gradaccfusion-pretraining/checkpoints/iter_0056477",
    },
    "E5": {
        "cumulative_gpu_seconds": 10_703_010.365292992,
        "fixed_window_loss": 2.5926591902971268,
        "last_raw_minibatch_loss": 2.682365655899048,
        "loss_window_updates": 32,
        "checkpoint": "/proj/datasets/interns/yuetai/agent_envs/more_task/runs/marin-adamh-baseline/20260729T182645Z-marin-adamh-3e20-p2545m-t18p617b-d2304-l23-16gpu-mbs1-ga8-llama3-b128-norecompute-nooffload-gradaccfusion-pretraining/checkpoints/iter_0035510",
    },
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def paloma(path: Path, scale: str) -> dict[str, float | str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    aggregate = payload.get("paloma_aggregate", {})
    if int(aggregate.get("subsets", -1)) != 16:
        raise ValueError(f"{scale} endpoint did not evaluate all 16 Paloma subsets")
    result = {
        "bits_per_byte": float(aggregate["bits_per_byte"]),
        "macro_bits_per_byte": float(aggregate["macro_bits_per_byte"]),
        "result_sha256": hashlib.sha256(raw).hexdigest(),
    }
    if result["bits_per_byte"] <= 0 or result["macro_bits_per_byte"] <= 0:
        raise ValueError(f"{scale} endpoint contains an invalid Paloma metric")
    return result


def source_provenance(scale: str, record: dict[str, Any]) -> dict[str, str]:
    payload = {
        "scale": scale,
        "wandb_project": record["wandb_project"],
        "wandb_run_id": record["wandb_run_id"],
        "model": record["model"],
        "bridge_commit": BASE_RUNTIME_COMMIT,
        "task_infrastructure_commit": TASK_INFRASTRUCTURE_COMMIT,
        "mcore_commit": MCORE_COMMIT,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {**payload, "inventory_sha256": sha256_text(encoded)}


def build(
    source: dict[str, Any], endpoints: dict[str, dict[str, float | str]]
) -> dict[str, Any]:
    raw_scales = source.get("scales")
    if source.get("version") not in {2, 3} or not isinstance(raw_scales, dict):
        raise ValueError("source must be the locked AdamH offline workspace")
    if set(raw_scales) != set(SCALES):
        raise ValueError("source workspace must contain exactly E0-E5")

    scales: dict[str, dict[str, Any]] = {}
    for scale, contract in SCALES.items():
        record = deepcopy(raw_scales[scale])
        record.update(
            {
                "parameters": contract["parameters"],
                "tokens": contract["tokens"],
                "global_batch_size": contract["gbs"],
                "gpus": contract["gpus"],
                "final_update": contract["iterations"],
                "scoring_update": contract["iterations"],
                "reference_status": "exact_final",
                "wandb_project": PROJECT,
                "run_contract": {
                    "seed": 0,
                    "sequence_length": 4096,
                    "tensor_parallel": 1,
                    "pipeline_parallel": 1,
                    "context_parallel": 1,
                },
            }
        )
        provenance = source_provenance(scale, record)
        record["source_provenance"] = provenance
        record["source_inventory_sha256"] = provenance["inventory_sha256"]

        if scale in ENDPOINT_TRAINING:
            endpoint = ENDPOINT_TRAINING[scale]
            endpoint_result = endpoints[scale]
            metrics = {
                "bits_per_byte": endpoint_result["bits_per_byte"],
                "macro_bits_per_byte": endpoint_result["macro_bits_per_byte"],
            }
            final = contract["iterations"]
            cost = float(endpoint["cumulative_gpu_seconds"])
            record["paloma_trajectory"] = [
                point for point in record["paloma_trajectory"] if int(point["iteration"]) < final
            ]
            record["paloma_trajectory"].append(
                {
                    "iteration": final,
                    "cumulative_gpu_seconds": cost,
                    **metrics,
                }
            )
            record["loss_cost_curve"] = [
                point for point in record["loss_cost_curve"] if int(point["iteration"]) < final
            ]
            record["loss_cost_curve"].append(
                {
                    "iteration": final,
                    "cumulative_gpu_seconds": cost,
                    "fixed_window_loss": float(endpoint["fixed_window_loss"]),
                }
            )
            record.update(
                {
                    "paloma_bits_per_byte": metrics["bits_per_byte"],
                    "paloma_macro_bits_per_byte": metrics["macro_bits_per_byte"],
                    "scoring_gpu_seconds": cost,
                    "scoring_fixed_window_loss": float(endpoint["fixed_window_loss"]),
                    "last_raw_minibatch_loss": float(endpoint["last_raw_minibatch_loss"]),
                    "loss_window_updates": int(endpoint["loss_window_updates"]),
                    "endpoint_evaluation": {
                        "evaluator": "marin_paloma",
                        "subsets": 16,
                        "result_sha256": endpoint_result["result_sha256"],
                    },
                }
            )
            checkpoint_identifier = str(endpoint["checkpoint"])
        else:
            final_point = record["paloma_trajectory"][-1]
            final_loss = record["loss_cost_curve"][-1]
            if int(final_point["iteration"]) != contract["iterations"]:
                raise ValueError(f"{scale} source trajectory is not exact-final")
            if int(final_loss["iteration"]) != contract["iterations"]:
                raise ValueError(f"{scale} source loss curve is not exact-final")
            record.update(
                {
                    "paloma_bits_per_byte": float(final_point["bits_per_byte"]),
                    "paloma_macro_bits_per_byte": float(final_point["macro_bits_per_byte"]),
                    "scoring_gpu_seconds": float(final_point["cumulative_gpu_seconds"]),
                    "scoring_fixed_window_loss": float(final_loss["fixed_window_loss"]),
                }
            )
            checkpoint_identifier = (
                f"wandb://{PROJECT}/{record['wandb_run_id']}/checkpoint/"
                f"iter_{contract['iterations']:07d}"
            )
        record["checkpoint"] = {
            "iteration": contract["iterations"],
            "identifier": checkpoint_identifier,
            "path_sha256": sha256_text(checkpoint_identifier),
            "recorded_training_gpus": RECORDED_TRAINING_GPUS[scale],
        }
        scales[scale] = record

    return {
        "version": 3,
        "evidence_scope": "locked_wsd_exact_final",
        "cost_definition": "active_training_gpu_seconds_excluding_evaluation",
        "cost_note": "Historical controls use the locked topology-normalized cost coordinate; E4/E5 endpoints extend it with summed TensorBoard iteration time.",
        "offline_workspace": "Self-contained Paloma trajectories, per-update-derived loss summaries, costs, and provenance; no network access is required.",
        "source_workspace_version": source.get(
            "source_workspace_version", source.get("version")
        ),
        "scales": scales,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--e4-result", type=Path, required=True)
    parser.add_argument("--e5-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build(
        json.loads(args.source.read_text()),
        {"E4": paloma(args.e4_result, "E4"), "E5": paloma(args.e5_result, "E5")},
    )
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
