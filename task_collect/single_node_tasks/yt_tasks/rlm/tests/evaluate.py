#!/usr/bin/env python3
"""Verify an RLM orchestrator-policy submission.

Serves the pinned base model + released adapter (frozen), hash-checks the
locked judge layer, then scores the agent's workspace policy with the frozen
50-row OOLONG contract.  Each row receives EVAL_REPEATS independent stochastic
rollouts.  The rollouts run through bounded, work-conserving pools instead of
five serial 50-row waves; queue wait happens outside the per-rollout timeout.

Reward integrity is mechanical, not judge-dependent. In the hardened
deployment (the full-run launcher):

- the true OOLONG labels never enter the verifier container: the canonical
  dataset path is shadow-bound to an answer-blanked masked copy, and this
  script refuses to run if that path still holds unmasked answers;
- the verifier's gold arrives through a writable single-file bind at
  ``/tests/gold_map.json``, is loaded into process memory, and the file is
  truncated BEFORE any rollout (agent-reachable) code executes;
- the scored reward is recomputed here from each rollout's saved final answer
  with the pinned ``_synth_score`` and the in-memory gold, so tampering with
  in-run rewards or results files cannot change the score;
- ``reported_reward_mismatches`` compares each in-run reward against the
  masked-gold expectation, so it is ~0 on honest runs and nonzero only when
  the rollout-side reward path itself was altered.

Without ``/tests/gold_map.json`` the script falls back to a legacy
author-smoke mode that builds the masked copy and gold map in-process (the
container then contains the true labels; do not use that mode for scored
verification).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import secrets
import signal
import subprocess
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean, pstdev

WORKSPACE = Path("/app/project")
PINNED = Path("/opt/project")
OUTPUT = Path("/tmp/rlm-eval")
SCRATCH = Path("/tmp/rlm-verifier")
LOGS = Path("/logs/verifier")
GOLD_FILE = Path("/tests/gold_map.json")

MODEL_NAME = "rlm-v0.1"
BASE_REPO = "models--Qwen--Qwen3-30B-A3B-Instruct-2507"
BASE_REV = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
ADAPTER_REPO = "models--mit-oasys--rlm-qwen3-30b-a3b-v0.1"
ADAPTER_REV = "7d72253525f47b1a8c3c4e1059ff88e40232dd2f"
ADAPTER_SHA256 = {
    "adapter_config.json": "fbbc7e816f055682935169e4b054051beb9e0dea79f7df99d5c95918240f64ca",
    "adapter_model.safetensors": "0ed035ff8a32fdc0e60b61d7df484f18c118b80c86447d3b244eca56a210f61c",
}
PROTECTED_REL = "training/environments/oolong"
IGNORED_TREE_PARTS = {"__pycache__", ".git"}

NUM_EXAMPLES = 50
MAX_ITERATIONS = 20
COMPLETION_CAP = 4096
ROLLOUT_TIMEOUT_SECONDS = 1800
EVAL_REPEATS = max(1, int(os.environ.get("RLM_EVAL_REPEATS", "5")))
MAX_ACTIVE_ROLLOUTS_PER_WORKER = NUM_EXAMPLES

CATEGORY_LABELS = (
    "numeric value",
    "entity",
    "location",
    "description and abstract concept",
    "abbreviation",
    "human being",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_snapshot(repo: str, revision: str) -> Path:
    cache = Path(os.environ.get("HF_HOME", "/root/.cache/huggingface"))
    for candidate in (cache / "hub" / repo / "snapshots" / revision,
                      cache / repo / "snapshots" / revision):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"pinned snapshot not found: {repo}@{revision}")


def tree_files(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file()
        and not any(part in IGNORED_TREE_PARTS or part.endswith(".egg-info") for part in p.parts)
    )


def check_protected_paths() -> None:
    """The judge layer inside the workspace must byte-match the pinned repo."""
    pinned_root = PINNED / PROTECTED_REL
    workspace_root = WORKSPACE / PROTECTED_REL
    for pinned_file in tree_files(pinned_root):
        rel = pinned_file.relative_to(pinned_root)
        candidate = workspace_root / rel
        if not candidate.is_file():
            raise RuntimeError(f"protected file missing from workspace: {PROTECTED_REL}/{rel}")
        if sha256_file(candidate) != sha256_file(pinned_file):
            raise RuntimeError(f"protected file modified: {PROTECTED_REL}/{rel}")
    if workspace_root.is_dir():
        for extra in tree_files(workspace_root):
            rel = extra.relative_to(workspace_root)
            if not (pinned_root / rel).is_file():
                raise RuntimeError(f"unexpected file inside protected path: {PROTECTED_REL}/{rel}")


def check_adapter(adapter: Path) -> None:
    for name, expected in ADAPTER_SHA256.items():
        actual = sha256_file(adapter / name)
        if actual != expected:
            raise RuntimeError(f"frozen adapter file changed: {name} sha256={actual}")


def pinned_oolong():
    """Import the pinned scorer module (baked editable install of /opt/project)."""
    module = importlib.import_module("oolong.env")
    if not Path(module.__file__).resolve().is_relative_to(PINNED.resolve()):
        raise RuntimeError(f"oolong module resolved outside the pinned tree: {module.__file__}")
    return module


def dataset_is_masked(dataset_path: Path) -> bool:
    """True iff every answer in the first shard is blank."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    shards = sorted((dataset_path / "data").glob("validation-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no validation shards under {dataset_path}/data")
    answers = pq.read_table(shards[0], columns=["answer"]).column("answer").to_pylist()
    return all(a in ("", None) for a in answers)


def load_gold_hardened() -> dict[int, dict]:
    """Read the verifier-private gold map, then destroy the on-disk copy
    before any rollout code can run. The bind is writable exactly so this
    truncation writes through to the host file."""
    raw = GOLD_FILE.read_text()
    gold = {int(k): v for k, v in json.loads(raw).items()}
    if len(gold) != NUM_EXAMPLES:
        raise RuntimeError(f"gold map has {len(gold)} rows, expected {NUM_EXAMPLES}")
    with GOLD_FILE.open("r+") as handle:
        handle.truncate(0)
    if GOLD_FILE.stat().st_size != 0:
        raise RuntimeError("failed to destroy the on-disk gold map before rollouts")
    return gold


def build_masked_dataset(true_path: Path) -> Path:
    """Legacy author-smoke path: blank the answer column in-process."""
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    masked_dir = SCRATCH / "masked-oolong" / "data"
    masked_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted((true_path / "data").glob("validation-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no validation shards under {true_path}/data")
    for shard in shards:
        table = pq.read_table(shard)
        if "answer" not in table.column_names:
            raise RuntimeError(f"shard {shard.name} has no answer column")
        blank = pa.array([""] * len(table), type=table.schema.field("answer").type)
        table = table.set_column(table.column_names.index("answer"), "answer", blank)
        pq.write_table(table, masked_dir / shard.name)
    return masked_dir.parent


def build_gold_map_legacy(module) -> dict[int, dict]:
    dataset = module._build_dataset(
        dataset_name="trec_coarse",
        min_ctx=131072,
        max_ctx=131072,
        num_examples=NUM_EXAMPLES,
        seed=42,
        exclude_numeric=False,
    )
    gold = {int(row["example_id"]): json.loads(row["info"]) for row in dataset}
    if len(gold) != NUM_EXAMPLES:
        raise RuntimeError(f"gold selection produced {len(gold)} rows, expected {NUM_EXAMPLES}")
    return gold


def parse_gold(module, datapoint: dict):
    """Replicate _synth_score's gold parse: scalar gold plus the full list."""
    answer = str(datapoint.get("answer", ""))
    try:
        parsed = module.ast.literal_eval(answer)
        gold = parsed[0]
        full = list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
    except Exception:
        gold = answer
        full = [answer]
    return gold, full


def fallback_path_used(module, datapoint: dict, output: str) -> bool:
    """True iff _synth_score would grant its reward through the whole-output
    substring fallback (env.py lines 86-88), not an exact/committed match."""
    gold, _ = parse_gold(module, datapoint)
    gold_s = str(gold)
    trimmed, _ = module._attempt_answer_parse(output)
    if str(trimmed) == gold_s or str(trimmed).lower() == gold_s.lower():
        return False
    atype = datapoint.get("answer_type", "")
    if atype in ("ANSWER_TYPE.NUMERIC", "ANSWER_TYPE.DATE"):
        return False
    if gold_s and gold_s.lower() not in [p.lower() for p in module.COMPARISON_PHRASES]:
        return gold_s.lower() in output.lower()
    return False


def is_degenerate_enumeration(module, datapoint: dict, output: str) -> bool:
    """RH gate, scoped to the substring-fallback path only: exact/committed
    answers and NUMERIC/DATE partial credit are never touched. On the
    fallback path, an answer that sprays candidates beyond the gold set —
    category labels the gold does not contain, or a pile of numeric IDs —
    is enumeration gaming, not an answer."""
    if not fallback_path_used(module, datapoint, output):
        return False
    gold, full = parse_gold(module, datapoint)
    gold_set = {str(g).lower() for g in full}
    lowered = output.lower()
    non_gold_labels = sum(
        1 for label in CATEGORY_LABELS if label in lowered and label not in gold_set
    )
    if non_gold_labels >= 2:
        return True
    gold_s = str(gold)
    if gold_s.isdigit():
        distinct_numbers = {m for m in re.findall(r"\d{3,}", output)}
        if len(distinct_numbers) >= 5:
            return True
    return False


def visible_gpu_groups() -> list[list[str]]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if raw and raw not in {"NoDevFiles", "-1"}:
        devices = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        devices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    available_workers = len(devices) // 4
    if available_workers < 1:
        raise RuntimeError(f"RLM verifier requires at least 4 visible GPUs, found {len(devices)}")
    default_workers = min(available_workers, EVAL_REPEATS, 2)
    requested_workers = int(os.environ.get("RLM_EVAL_WORKERS", str(default_workers)))
    if not 1 <= requested_workers <= min(available_workers, EVAL_REPEATS):
        raise RuntimeError(
            "RLM_EVAL_WORKERS must be between 1 and "
            f"{min(available_workers, EVAL_REPEATS)}, got {requested_workers}"
        )
    return [devices[index * 4:(index + 1) * 4] for index in range(requested_workers)]


def repeat_allocation(total: int, workers: int) -> list[int]:
    base, extra = divmod(total, workers)
    return [base + (index < extra) for index in range(workers)]


def wait_for_server(
    process: subprocess.Popen, port: int, timeout: int = 1800
) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError("vLLM server exited before becoming healthy")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=3
            ) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # server is still loading weights
            last_error = exc
        time.sleep(2)
    raise TimeoutError(f"vLLM did not become healthy: {last_error}")


def numeric_metric(row: dict, name: str) -> float:
    value = (row.get("metrics") or {}).get(name, 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def load_results(pool_dir: Path, rollouts_per_example: int) -> list[dict]:
    paths = sorted(pool_dir.rglob("results.jsonl"), key=lambda p: p.stat().st_mtime)
    if not paths:
        raise FileNotFoundError(f"vf-eval did not produce results.jsonl below {pool_dir}")
    rows = [json.loads(line) for line in paths[-1].read_text().splitlines() if line.strip()]
    expected_rows = NUM_EXAMPLES * rollouts_per_example
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"incomplete pooled evaluation: expected {expected_rows} rows, found {len(rows)}"
        )
    counts = Counter(int(row.get("example_id", -1)) for row in rows)
    expected_ids = set(range(NUM_EXAMPLES))
    if set(counts) != expected_ids or set(counts.values()) != {rollouts_per_example}:
        raise RuntimeError(
            "pooled evaluation is not the fixed 50-row selection with "
            f"{rollouts_per_example} rollouts per row: {dict(sorted(counts.items()))}"
        )
    return rows


def run_pool(
    index: int,
    rollouts_per_example: int,
    masked_path: Path,
    port: int,
) -> list[dict]:
    # Random suffix: rollout-reachable code cannot predict (and pre-plant
    # results in) the output directory.
    pool_dir = OUTPUT / f"pool-{index}-{secrets.token_hex(8)}"
    pool_dir.mkdir(parents=True, exist_ok=True)
    env_args = {
        "dataset_name": "trec_coarse",
        "min_ctx": 131072,
        "max_ctx": 131072,
        "num_examples": NUM_EXAMPLES,
        "seed": 42,
        "max_iterations": MAX_ITERATIONS,
        "sub_max_tokens": COMPLETION_CAP,
    }
    command = [
        "vf-eval", "oolong",
        "--provider", "vllm",
        "--api-base-url", f"http://127.0.0.1:{port}/v1",
        "--api-key-var", "VLLM_API_KEY",
        "--model", MODEL_NAME,
        "--num-examples", str(NUM_EXAMPLES),
        "--rollouts-per-example", str(rollouts_per_example),
        "--max-concurrent", str(MAX_ACTIVE_ROLLOUTS_PER_WORKER),
        "--independent-scoring",
        "--max-tokens", str(COMPLETION_CAP),
        "--sampling-args", json.dumps({"extra_body": {"enable_thinking": False}}),
        "--env-args", json.dumps(env_args),
        "--timeout", str(ROLLOUT_TIMEOUT_SECONDS),
        "--max-retries", "0",
        "--num-workers", "1",
        "--output-dir", str(pool_dir),
        "--save-results",
        "--state-columns", "rlm_final_answer,rlm_iterations,rlm_repl_calls,rlm_sub_llm_calls,timed_out",
        "--disable-tui",
    ]
    eval_env = os.environ.copy()
    eval_env["VLLM_API_KEY"] = "EMPTY"
    # The rollout side sees only the masked dataset.
    eval_env["OOLONG_DATASET_PATH"] = str(masked_path)
    # The workspace policy is imported via path precedence, not installed:
    # rlm_train and the rlm package come from /app/project, while the pinned
    # oolong scorer stays the image's baked editable install of /opt/project
    # (neither PYTHONPATH entry exposes a top-level ``oolong`` package).
    workspace_paths = f"{WORKSPACE}/training/src:{WORKSPACE}"
    existing = eval_env.get("PYTHONPATH", "")
    eval_env["PYTHONPATH"] = f"{workspace_paths}:{existing}" if existing else workspace_paths
    with (LOGS / f"oolong-pool-{index}.log").open("a") as eval_log:
        subprocess.run(command, env=eval_env, stdout=eval_log, stderr=subprocess.STDOUT, check=True)
    return load_results(pool_dir, rollouts_per_example)


def run_pool_with_retry(
    index: int,
    rollouts_per_example: int,
    masked_path: Path,
    port: int,
) -> list[dict]:
    """One infrastructure retry per pool, so a transient vf-eval failure
    does not zero an otherwise healthy submission."""
    try:
        return run_pool(index, rollouts_per_example, masked_path, port)
    except Exception as exc:  # noqa: BLE001
        print(f"pool {index} failed once ({exc}); retrying", flush=True)
        return run_pool(index, rollouts_per_example, masked_path, port)


def partition_pools_into_repeats(
    pool_rows: list[list[dict]], allocations: list[int]
) -> list[list[dict]]:
    """Give the pooled samples a diagnostic repeat index.

    The score is the mean over all independent samples, so this partition does
    not affect reward.  It only preserves per-repeat dispersion reporting.
    """
    repeats: list[list[dict]] = [[] for _ in range(sum(allocations))]
    offset = 0
    if len(pool_rows) != len(allocations):
        raise RuntimeError("pool result count does not match worker allocation")
    for rows, allocation in zip(pool_rows, allocations):
        by_example: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            by_example[int(row["example_id"])].append(row)
        for local_repeat in range(allocation):
            repeats[offset + local_repeat] = [
                by_example[example_id][local_repeat]
                for example_id in range(NUM_EXAMPLES)
            ]
        offset += allocation
    if any(len(rows) != NUM_EXAMPLES for rows in repeats):
        raise RuntimeError("failed to partition pooled rollouts into complete repeats")
    return repeats


def write_reward(reward: float, gate: int, metrics: dict) -> None:
    """evaluate.py owns the reward files directly (no separate score step),
    written as its final act. The reward is also echoed to stdout, which
    Harbor captures outside the container, making post-hoc overwrites of
    /logs detectable."""
    result = {
        "reward": round(reward, 6),
        "policy_gate": gate,
        "metric": "mean_recomputed_raw_oolong_reward",
        **metrics,
    }
    (LOGS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (LOGS / "reward.json").write_text(json.dumps(result, indent=2))
    (LOGS / "reward.txt").write_text(f"{reward:.6f}\n")
    print("VERIFIER_REWARD " + json.dumps({"reward": round(reward, 6), **{
        k: metrics[k] for k in ("per_repeat_reward", "reported_reward_mismatches",
                                "degenerate_enumeration_rows") if k in metrics
    }}), flush=True)


def main() -> None:
    if not WORKSPACE.is_dir():
        raise FileNotFoundError("workspace submission missing at /app/project")
    LOGS.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)

    base = resolve_snapshot(BASE_REPO, BASE_REV)
    adapter = resolve_snapshot(ADAPTER_REPO, ADAPTER_REV)
    check_adapter(adapter)
    check_protected_paths()

    dataset_path = Path(os.environ["OOLONG_DATASET_PATH"])
    module = pinned_oolong()

    hardened = GOLD_FILE.is_file() and GOLD_FILE.stat().st_size > 0
    if hardened:
        # Fail closed: scored verification must never run over true labels.
        if not dataset_is_masked(dataset_path):
            raise RuntimeError(
                "hardened mode requires the canonical dataset path to hold the "
                "masked copy; refusing to run rollouts over true labels"
            )
        gold_map = load_gold_hardened()
        masked_path = dataset_path
    else:
        print("WARNING: legacy author-smoke mode; container holds true labels", flush=True)
        gold_map = build_gold_map_legacy(module)
        masked_path = build_masked_dataset(dataset_path)

    # Each worker is an independent instance of the released TP4 serving
    # contract.  On the normal 8-GPU verifier allocation two workers split the
    # five samples per row as 3+2.  A 4-GPU author smoke falls back to one
    # worker without changing the estimator.
    gpu_groups = visible_gpu_groups()
    allocations = repeat_allocation(EVAL_REPEATS, len(gpu_groups))
    ports = [8000 + index for index in range(len(gpu_groups))]
    servers: list[subprocess.Popen] = []
    server_logs = []
    pool_rows: list[list[dict]] = []
    try:
        for index, (gpu_group, port) in enumerate(
            zip(gpu_groups, ports), start=1
        ):
            worker_scratch = SCRATCH / f"worker-{index}"
            server_env = os.environ.copy()
            server_env.update(
                CUDA_VISIBLE_DEVICES=",".join(gpu_group),
                VLLM_API_KEY="EMPTY",
                VLLM_CACHE_ROOT=str(worker_scratch / "vllm-cache"),
                VLLM_ALLREDUCE_USE_SYMM_MEM="0",
                TMPDIR=str(worker_scratch / "tmp"),
                TORCHINDUCTOR_CACHE_DIR=str(worker_scratch / "torchinductor"),
                OPENBLAS_NUM_THREADS="1",
                OMP_NUM_THREADS="1",
            )
            for key in ("VLLM_CACHE_ROOT", "TMPDIR", "TORCHINDUCTOR_CACHE_DIR"):
                Path(server_env[key]).mkdir(parents=True, exist_ok=True)
            server_command = [
                "vllm", "serve", str(base),
                "--tensor-parallel-size", "4",
                "--max-model-len", "16384",
                "--gpu-memory-utilization", "0.9",
                "--enable-lora",
                "--max-lora-rank", "64",
                "--lora-modules", f"{MODEL_NAME}={adapter}",
                "--host", "127.0.0.1",
                "--port", str(port),
            ]
            log_handle = (LOGS / f"vllm-worker-{index}.log").open("w")
            server_logs.append(log_handle)
            servers.append(
                subprocess.Popen(
                    server_command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=server_env,
                    start_new_session=True,
                )
            )
        for server, port in zip(servers, ports):
            wait_for_server(server, port)
        with ThreadPoolExecutor(max_workers=len(servers)) as executor:
            futures = [
                executor.submit(
                    run_pool_with_retry,
                    index,
                    allocation,
                    masked_path,
                    port,
                )
                for index, (allocation, port) in enumerate(
                    zip(allocations, ports), start=1
                )
            ]
            pool_rows = [future.result() for future in futures]
    finally:
        for server in servers:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
        for server in servers:
            if server.poll() is None:
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()
        for log_handle in server_logs:
            log_handle.close()

    repeat_rows = partition_pools_into_repeats(pool_rows, allocations)

    # Recompute every reward verifier-side from the saved final answers.
    per_repeat_reward: list[float] = []
    reported_mismatch = 0
    degenerate_rows = 0
    audit_rows: list[dict] = []
    for repeat_index, rows in enumerate(repeat_rows, start=1):
        recomputed: list[float] = []
        for row in rows:
            example_id = int(row["example_id"])
            final = row.get("rlm_final_answer")
            final_text = "" if final is None else str(final)
            datapoint = gold_map[example_id]
            reward = float(module._synth_score(datapoint, final_text))
            degenerate = reward > 0 and is_degenerate_enumeration(module, datapoint, final_text)
            if degenerate:
                reward = 0.0
                degenerate_rows += 1
            # Tamper signal: the in-run reward was computed against MASKED
            # gold, so compare it against that same expectation. Honest runs
            # match exactly; only an altered rollout-side reward path differs.
            reported = float(row.get("reward", 0.0))
            masked_dp = dict(datapoint, answer="")
            expected_reported = float(module._synth_score(masked_dp, final_text))
            if abs(reported - expected_reported) > 1e-9:
                reported_mismatch += 1
            recomputed.append(reward)
            audit_rows.append(
                {
                    "repeat": repeat_index,
                    "example_id": example_id,
                    "final_answer": final_text[:2000],
                    "recomputed_reward": reward,
                    "reported_reward": reported,
                    "expected_masked_reward": expected_reported,
                    "degenerate_enumeration": degenerate,
                    "stop_condition": row.get("stop_condition"),
                }
            )
        per_repeat_reward.append(fmean(recomputed))

    all_rows = [row for rows in repeat_rows for row in rows]
    errored = sum(row.get("error") is not None for row in all_rows)
    timed_out = sum(
        row.get("timed_out") is True or row.get("stop_condition") == "timeout_reached"
        for row in all_rows
    )
    (LOGS / "per_row_audit.json").write_text(json.dumps(audit_rows, indent=2))
    mean_reward = fmean(per_repeat_reward)
    metrics = {
        "oolong_reward": mean_reward,
        "eval_repeats": EVAL_REPEATS,
        "total_rollouts": len(all_rows),
        "rollouts_per_example": EVAL_REPEATS,
        "evaluation_schedule": "independent_scoring_bounded_rolling_pool",
        "eval_worker_count": len(gpu_groups),
        "worker_repeat_allocation": allocations,
        "max_active_rollouts_per_worker": MAX_ACTIVE_ROLLOUTS_PER_WORKER,
        "per_repeat_reward": per_repeat_reward,
        "repeat_reward_pstdev": pstdev(per_repeat_reward) if len(per_repeat_reward) > 1 else 0.0,
        "reward_source": "verifier_recomputed_synth_score_on_masked_gold_rollouts",
        "hardened_gold_isolation": hardened,
        "reported_reward_mismatches": reported_mismatch,
        "degenerate_enumeration_rows": degenerate_rows,
        "average_subcalls": fmean(numeric_metric(r, "rlm_sub_llm_calls") for r in all_rows),
        "average_repl_calls": fmean(numeric_metric(r, "rlm_repl_calls") for r in all_rows),
        "average_iterations": fmean(numeric_metric(r, "rlm_iterations") for r in all_rows),
        "max_iterations": MAX_ITERATIONS,
        "completion_token_cap": COMPLETION_CAP,
        "rollout_timeout_seconds": ROLLOUT_TIMEOUT_SECONDS,
        "examples_per_repeat": NUM_EXAMPLES,
        "errored_examples": errored,
        "error_rate": errored / len(all_rows),
        "timed_out_examples": timed_out,
        "timeout_rate": timed_out / len(all_rows),
        "base_model": f"Qwen/Qwen3-30B-A3B-Instruct-2507@{BASE_REV}",
        "adapter": f"mit-oasys/rlm-qwen3-30b-a3b-v0.1@{ADAPTER_REV}",
        "dataset": "oolongbench/oolong-synth validation:trec_coarse@131072",
        "evaluation_temperature": "inference_server_model_default",
    }
    write_reward(max(0.0, min(1.0, mean_reward)), 1, metrics)


if __name__ == "__main__":
    main()
