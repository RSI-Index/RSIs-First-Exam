#!/usr/bin/env python3
"""Score independently regenerated Paloma and LongPPL across all six rungs."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


SCALES = {
    "E0": {"iterations": 44_317, "gpus": 8, "parameters": 550_337_664},
    "E1": {"iterations": 55_125, "gpus": 8, "parameters": 837_007_744},
    "E2": {"iterations": 38_014, "gpus": 8, "parameters": 998_036_992},
    "E3": {"iterations": 40_283, "gpus": 32, "parameters": 1_384_584_448},
    "E4": {"iterations": 56_477, "gpus": 32, "parameters": 1_934_716_160},
    "E5": {"iterations": 35_510, "gpus": 128, "parameters": 2_544_614_912},
}
PARAMETER_PATTERN = re.compile(
    r"number of parameters on \(tensor, pipeline\) model parallel rank \(0, 0\):\s*([0-9]+)"
)


def finite_float(value: Any, label: str, minimum: float | None = None) -> float:
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"invalid {label}: {result}")
    return result


def score_scale(scale: str, run_dir: Path, baseline: dict[str, float]) -> dict[str, Any]:
    contract = SCALES[scale]
    step = f"step_{contract['iterations']:08d}"
    harness = json.loads((run_dir / "eval_harness" / step / "results.json").read_text())
    longppl_payload = json.loads((run_dir / "longppl" / step / "results.json").read_text())
    paloma = harness["paloma_aggregate"]
    long_results = longppl_payload["results"]
    protocol = longppl_payload["longppl_eval"]
    micro = finite_float(paloma["bits_per_byte"], f"{scale} Paloma micro", 0.0)
    macro = finite_float(paloma["macro_bits_per_byte"], f"{scale} Paloma macro", 0.0)
    longppl = finite_float(long_results["longppl"], f"{scale} LongPPL", 1.0)
    expected_protocol = {
        "iteration": contract["iterations"],
        "world_size": contract["gpus"],
        "longppl_commit": "b4f80af0015dd12202438e1e440cf12316f1cb23",
        "minimum_source_tokens": 16384,
        "maximum_source_length": 32768,
        "training_sequence_length": 4096,
        "truncation_length": 4096,
        "sliding_window": 1024,
        "target_tokenizer_revision": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
        "mode": "offline",
    }
    protocol_failures = {
        key: {"expected": expected, "actual": protocol.get(key)}
        for key, expected in expected_protocol.items()
        if protocol.get(key) != expected
    }
    if int(paloma["subsets"]) != 16:
        protocol_failures["paloma_subsets"] = {"expected": 16, "actual": paloma["subsets"]}
    if int(long_results["samples"]) != 50:
        protocol_failures["longppl_samples"] = {"expected": 50, "actual": long_results["samples"]}
    parameter_matches = PARAMETER_PATTERN.findall((run_dir / "run.log").read_text(errors="replace"))
    if not parameter_matches:
        raise ValueError(f"{scale} verifier log lacks the parameter count")
    parameters = max(int(value) for value in parameter_matches)
    guards = {
        "paloma_micro": micro <= float(baseline["paloma_bits_per_byte"]) * 1.01,
        "paloma_macro": macro <= float(baseline["paloma_macro_bits_per_byte"]) * 1.01,
        "parameter_budget": parameters <= int(contract["parameters"] * 1.02),
        "longppl_protocol": not protocol_failures,
        "longppl_improvement": longppl < float(baseline["longppl"]),
    }
    return {
        "paloma_bits_per_byte": micro,
        "paloma_macro_bits_per_byte": macro,
        "longppl": longppl,
        "longppl_loss": finite_float(long_results["longppl_loss"], f"{scale} loss", 0.0),
        "ordinary_ppl": finite_float(long_results["ppl"], f"{scale} ordinary PPL", 1.0),
        "parameters": parameters,
        "relative_longppl_gain": float(baseline["longppl"]) / longppl,
        "guards": guards,
        "protocol_failures": protocol_failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument(
        "--baselines",
        type=Path,
        default=Path("/task-data/positional_adamh_baselines.json"),
    )
    parser.add_argument("--policy-report", type=Path, required=True)
    parser.add_argument("--logs-dir", type=Path, default=Path("/logs/verifier"))
    args = parser.parse_args()
    baseline_payload = json.loads(args.baselines.read_text())
    if baseline_payload.get("version") != 1 or set(baseline_payload.get("scales", {})) != set(SCALES):
        raise ValueError("invalid six-scale baseline manifest")
    policy = json.loads(args.policy_report.read_text())
    metrics = {
        scale: score_scale(scale, args.runs_root / scale, baseline_payload["scales"][scale])
        for scale in SCALES
    }
    scientific_pass = all(all(item["guards"].values()) for item in metrics.values())
    policy_pass = policy.get("policy_gate") == 1
    passed = scientific_pass and policy_pass
    reward = (
        math.exp(sum(math.log(item["relative_longppl_gain"]) for item in metrics.values()) / len(metrics))
        if passed
        else 0.0
    )
    output = {
        "scales": metrics,
        "policy_gate": policy_pass,
        "all_scale_gates": scientific_pass,
        "reward": reward,
    }
    reward_result = {
        "reward": round(reward, 6),
        "policy_gate": int(policy_pass),
        "all_scale_gates": int(scientific_pass),
    }
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    (args.logs_dir / "metrics.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    (args.logs_dir / "reward.json").write_text(json.dumps(reward_result, indent=2, sort_keys=True) + "\n")
    (args.logs_dir / "reward.txt").write_text(f"{reward:.6f}\n")


if __name__ == "__main__":
    main()
