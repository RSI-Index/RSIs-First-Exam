#!/usr/bin/env python3
"""Frozen launcher for one full-budget Slime Search-R1 attempt."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

try:
    from task_contract import (
        ALGORITHM_ROOT,
        BM25_INDEX_PATH,
        BM25_INDEX_REVISION,
        DATASET_REVISION,
        FINAL_ROLLOUT_ID,
        MODEL_PATH,
        MODEL_REVISION,
        NUM_ROLLOUT,
        SEARCH_R1_COMMIT,
        SEARCH_TOP_K,
        SLIME_COMMIT,
        WIKI_CORPUS_PATH,
        algorithm_source_sha256,
        build_training_arguments,
        canonical_command_sha256,
        load_algorithm_config,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from task_contract import (
        ALGORITHM_ROOT,
        BM25_INDEX_PATH,
        BM25_INDEX_REVISION,
        DATASET_REVISION,
        FINAL_ROLLOUT_ID,
        MODEL_PATH,
        MODEL_REVISION,
        NUM_ROLLOUT,
        SEARCH_R1_COMMIT,
        SEARCH_TOP_K,
        SLIME_COMMIT,
        WIKI_CORPUS_PATH,
        algorithm_source_sha256,
        build_training_arguments,
        canonical_command_sha256,
        load_algorithm_config,
    )


def parse_visible_devices(value: str) -> tuple[str, ...]:
    devices = tuple(part.strip() for part in value.split(",") if part.strip())
    if len(devices) != 8 or len(set(devices)) != 8:
        raise ValueError("exactly eight unique visible GPU devices are required")
    return devices


def discover_visible_devices() -> tuple[str, ...]:
    configured = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if configured:
        return parse_visible_devices(configured)
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        text=True,
    )
    return parse_visible_devices(",".join(line.strip() for line in output.splitlines() if line.strip()))


def build_slime_command(config: dict, output_dir: str | Path) -> list[str]:
    return ["python3", "/opt/slime/train.py", *build_training_arguments(config, output_dir)]


def require_stable_algorithm(before: str, after: str) -> str:
    if before != after:
        raise RuntimeError("algorithm source changed during training")
    return before


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _wait_for_retriever(process: subprocess.Popen, timeout: float = 600.0) -> None:
    deadline = time.monotonic() + timeout
    request = urllib.request.Request(
        "http://127.0.0.1:8000/retrieve",
        data=b'{"queries":["test"],"topk":1,"return_scores":false}',
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"BM25 service exited with status {process.returncode}")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise TimeoutError("BM25 service did not become ready")


def _resolve_dist_checkpoint(save_root: Path) -> Path:
    expected = save_root / f"iter_{FINAL_ROLLOUT_ID:07d}"
    if (expected / "common.pt").is_file() and (expected / ".metadata").is_file():
        return expected
    raise RuntimeError(f"required full-budget checkpoint is missing: {expected.name}")


def _export_hf(output: Path) -> None:
    distributed = _resolve_dist_checkpoint(output / "torch_dist")
    hf_output = output / "hf_model"
    subprocess.run(
        [
            "python3",
            "/opt/slime/tools/convert_torch_dist_to_hf.py",
            "--input-dir",
            str(distributed),
            "--output-dir",
            str(hf_output),
            "--origin-hf-dir",
            MODEL_PATH,
            "--vocab-size",
            "151936",
            "--force",
        ],
        check=True,
    )
    required = (hf_output / "config.json", hf_output / "model.safetensors.index.json", hf_output / "tokenizer.json")
    if not all(path.is_file() for path in required):
        raise RuntimeError("HF export is incomplete")


def run_attempt(config_path: Path, output: Path) -> None:
    discover_visible_devices()
    config = load_algorithm_config(config_path)
    algorithm_hash = algorithm_source_sha256(ALGORITHM_ROOT)
    command = build_slime_command(config, output)
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "training-command.json", {"argv": command, "sha256": canonical_command_sha256(command)})

    log_dir = output / "logs"
    log_dir.mkdir()
    retriever_log = (log_dir / "bm25.log").open("w", encoding="utf-8")
    retriever = subprocess.Popen(
        [
            "python3",
            "/opt/Search-R1/search_r1/search/retrieval_server.py",
            "--index_path",
            BM25_INDEX_PATH,
            "--corpus_path",
            WIKI_CORPUS_PATH,
            "--topk",
            str(SEARCH_TOP_K),
            "--retriever_name",
            "bm25",
        ],
        stdout=retriever_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        _wait_for_retriever(retriever)
        subprocess.run(["ray", "stop", "--force"], check=False)
        subprocess.run(
            ["ray", "start", "--head", "--node-ip-address", "127.0.0.1", "--num-gpus", "8", "--disable-usage-stats"],
            check=True,
        )
        runtime = json.dumps(
            {
                "env_vars": {
                    "PYTHONPATH": "/opt/Megatron-LM:/opt/slime/examples/search-r1",
                    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                }
            },
            separators=(",", ":"),
        )
        subprocess.run(
            [
                "ray",
                "job",
                "submit",
                "--address=http://127.0.0.1:8265",
                f"--runtime-env-json={runtime}",
                "--",
                *command,
            ],
            check=True,
        )
        _export_hf(output)
        require_stable_algorithm(algorithm_hash, algorithm_source_sha256(ALGORITHM_ROOT))
        _atomic_json(
            output / "run-contract.json",
            {
                "algorithm_sha256": algorithm_hash,
                "bm25_index_revision": BM25_INDEX_REVISION,
                "dataset_revision": DATASET_REVISION,
                "model_revision": MODEL_REVISION,
                "num_rollout": NUM_ROLLOUT,
                "search_r1_commit": SEARCH_R1_COMMIT,
                "slime_commit": SLIME_COMMIT,
                "training_command_sha256": canonical_command_sha256(command),
                "world_size": 8,
            },
        )
    finally:
        retriever.terminate()
        try:
            retriever.wait(timeout=30)
        except subprocess.TimeoutExpired:
            retriever.kill()
        retriever_log.close()
        subprocess.run(["ray", "stop", "--force"], check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm-config", type=Path, default=Path(ALGORITHM_ROOT) / "algorithm_config.toml")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_attempt(args.algorithm_config, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
