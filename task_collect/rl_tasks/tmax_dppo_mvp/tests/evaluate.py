#!/usr/bin/env python3
"""Matched final evaluation for official TMAX step_200 and one candidate."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any


FROZEN_DECODING = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_steps": 64,
    "max_tokens": 16384,
    "attempts_per_task": 1,
}


def core_config(config: Any) -> dict[str, Any]:
    keys = [
        "model_type",
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
    ]
    return {key: getattr(config, key, None) for key in keys}


def bootstrap_interval(values: list[float], draws: int = 5000) -> list[float]:
    rng = random.Random(3407)
    means = []
    for _ in range(draws):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return [means[int(0.025 * draws)], means[int(0.975 * draws)]]


def compare_results(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if baseline.get("protocol_sha256") != candidate.get("protocol_sha256"):
        raise ValueError("baseline and candidate protocol hashes differ")
    baseline_mean = float(baseline["mean_reward"])
    candidate_mean = float(candidate["mean_reward"])
    absolute_gain = candidate_mean - baseline_mean
    return {
        "baseline": baseline,
        "candidate": candidate,
        "absolute_gain": absolute_gain,
        "relative_gain": None if baseline_mean == 0.0 else absolute_gain / baseline_mean,
        "protocol_sha256": baseline["protocol_sha256"],
    }


def protocol_digest(manifest: Path, final_prompt: Path, final_reward: str) -> str:
    digest = hashlib.sha256()
    for label, payload in (
        ("manifest", manifest.read_bytes()),
        ("final_prompt", final_prompt.read_bytes()),
        ("final_reward", final_reward.encode()),
        ("decoding", json.dumps(FROZEN_DECODING, sort_keys=True).encode()),
        ("evaluator", Path(__file__).read_bytes()),
    ):
        digest.update(label.encode() + b"\0" + payload + b"\0")
    return digest.hexdigest()


def _read_metrics(results_root: Path, protocol_sha256: str) -> dict[str, Any]:
    metrics = json.loads((results_root / "metrics.json").read_text())
    per_task = metrics.get("per_task", {})
    if metrics.get("n_tasks") != 64 or len(per_task) != 64:
        raise RuntimeError("final evaluation did not complete all 64 tasks")
    attempts = metrics.get("attempts_per_task", [])
    if len(attempts) != 64 or any(int(value) != 1 for value in attempts):
        raise RuntimeError("final evaluation must have exactly one attempt per task")
    task_ids = sorted(per_task)
    scores = [float(per_task[task_id]["mean_reward"]) for task_id in task_ids]
    if any(not 0.0 <= score <= 1.0 for score in scores):
        raise RuntimeError("original programmatic-verifier reward outside [0, 1]")
    return {
        "hidden_tasks": 64,
        "attempts_per_task": 1,
        "mean_reward": sum(scores) / len(scores),
        "standard_error": float(metrics["sem_reward"]),
        "task_bootstrap_95ci": bootstrap_interval(scores),
        "task_ids": task_ids,
        "per_task_rewards": scores,
        "protocol_sha256": protocol_sha256,
        "decoding": FROZEN_DECODING,
    }


def evaluate_model(*, model_dir: Path, label: str, dataset: Path, protocol_sha256: str) -> dict[str, Any]:
    results_root = Path("/app/output/tmax-final-eval") / label
    results_root.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update(
        {
            "MODEL_PATH": str(model_dir),
            "MODEL_REVISION": "main",
            "SERVED_MODEL_NAME": f"tmax-{label}",
            "HARBOR_MODEL_NAME": f"hosted_vllm/tmax-{label}",
            "VLLM_VERSION": "0.19.1",
            "VLLM_TOOL_CALL_PARSER": "qwen3_xml",
            "TP_SIZE": "8",
            "DP_SIZE": "1",
            "MAX_MODEL_LEN": "67584",
            "DATASET": str(dataset),
            "HARBOR_ENV": "docker",
            "AGENT_IMPORT_PATH": "Vanillux2Agent:Vanillux2Agent",
            "N_CONCURRENT": "8",
            "N_ATTEMPTS": "1",
            "JOB_NAME": f"tmax-final-{label}",
            "RESULTS_DIR": str(results_root),
            "EXTRA_AGENT_KWARGS": "temperature=0.0\ntop_p=1.0\nmax_steps=64\nmax_tokens=16384\nmax_format_errors=64",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "UV_OFFLINE": "1",
            "OPENAI_API_KEY": "dummy",
        }
    )
    subprocess.run(
        ["bash", "/opt/project/scripts/beaker/run_eval_in_job.sh"],
        cwd="/opt/project",
        env=environment,
        check=True,
    )
    return _read_metrics(results_root, protocol_sha256)


def main() -> None:
    from transformers import AutoConfig

    candidate_dir = Path("/app/output/submission/model")
    baseline_dir = Path(os.environ.get("TMAX_OFFICIAL_BASELINE", "/models/TMAX-step-200"))
    base_dir = Path(os.environ.get("TMAX_BASE_MODEL", "/models/Qwen3.5-9B"))
    dataset = Path(os.environ.get("TMAX_HIDDEN_DATASET", "/datasets/tmax/eval/terminal-bench-2-hidden"))
    image_store = Path(os.environ.get("TMAX_PRELOADED_IMAGE_STORE", "/datasets/tmax/eval/oci-images"))
    final_prompt = Path(
        os.environ.get("TMAX_FINAL_PROMPT", "/opt/project/rl_data/generator/vanillux_prompts.yaml")
    )
    final_reward = os.environ.get("TMAX_FINAL_REWARD", "sealed_original_programmatic_verifiers")
    manifest = dataset / "manifest.json"
    if not manifest.is_file() or not image_store.is_dir() or not final_prompt.is_file():
        raise RuntimeError("sealed final-evaluation assets are missing")
    if json.loads(manifest.read_text()).get("task_count") != 64:
        raise RuntimeError("sealed evaluation manifest must contain exactly 64 tasks")

    reference_config = AutoConfig.from_pretrained(base_dir, local_files_only=True, trust_remote_code=False)
    for label, model_dir in (("official-step-200", baseline_dir), ("candidate", candidate_dir)):
        config = AutoConfig.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        if core_config(config) != core_config(reference_config):
            raise RuntimeError(f"{label} model architecture differs from frozen Qwen3.5-9B")

    for archive in sorted(image_store.glob("*.tar")):
        subprocess.run(["podman", "load", "-i", str(archive)], check=True)

    protocol_sha256 = protocol_digest(manifest, final_prompt, final_reward)
    baseline = evaluate_model(
        model_dir=baseline_dir,
        label="official-step-200",
        dataset=dataset,
        protocol_sha256=protocol_sha256,
    )
    candidate = evaluate_model(
        model_dir=candidate_dir,
        label="candidate",
        dataset=dataset,
        protocol_sha256=protocol_sha256,
    )
    comparison = compare_results(baseline, candidate)
    comparison.update(
        {
            "mean_reward": candidate["mean_reward"],
            "evaluation_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "final_prompt_sha256": hashlib.sha256(final_prompt.read_bytes()).hexdigest(),
            "final_reward": final_reward,
        }
    )
    output = Path("/app/output/final-comparison.json")
    output.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    Path("/app/output/verifier-metrics.json").write_text(output.read_text())


if __name__ == "__main__":
    main()
