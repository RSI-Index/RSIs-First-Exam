"""Frozen scientific contract for the Slime Search-R1 algorithm task."""

from __future__ import annotations

import hashlib
import json
import math
from ast import literal_eval
from pathlib import Path
from typing import Any


SLIME_COMMIT = "06ffdbe22be068b52f9ed0fc318c473f7030197e"
SEARCH_R1_COMMIT = "598e61bd1d36895726d28a8d06b3a15bed19f5d3"
MODEL_ID = "Qwen/Qwen2.5-3B"
MODEL_REVISION = "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"
DATASET_ID = "PeterJinGo/nq_hotpotqa_train"
DATASET_REVISION = "b7d80abfee334a7a91cb377544f09180d58b34f6"
BM25_INDEX_ID = "PeterJinGo/wiki-18-bm25-index"
BM25_INDEX_REVISION = "2c7554f25f425038c4bcb155735a0f831851fd78"
WIKI_CORPUS_ID = "PeterJinGo/wiki-18-corpus"
WIKI_CORPUS_REVISION = "69c1c00ffe7c5554c68d8548355cb22e46aabc51"
REFERENCE_MODEL_ID = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo"
REFERENCE_MODEL_REVISION = "d4d2f71e39ff3e0ed4def1e8f860b87d9f6bfac7"
ALL_ASSET_REVISIONS = (
    SLIME_COMMIT,
    SEARCH_R1_COMMIT,
    MODEL_REVISION,
    DATASET_REVISION,
    BM25_INDEX_REVISION,
    WIKI_CORPUS_REVISION,
    REFERENCE_MODEL_REVISION,
)

MODEL_PATH = "/models/Qwen2.5-3B"
REFERENCE_LOAD_PATH = MODEL_PATH
TRAIN_DATA_PATH = "/datasets/nq_hotpotqa_train/train.parquet"
EVAL_DATA_PATH = "/eval/nq_hotpotqa_test.parquet"
BM25_INDEX_PATH = "/retriever/bm25"
WIKI_CORPUS_PATH = "/retriever/wiki-18.jsonl"
ATTEMPT_OUTPUT_ROOT = "/app/output/attempts"
TRUSTED_RUN_ROOT = "/var/lib/slime-task/runs"
ALGORITHM_ROOT = "/app/project/rl_research"

NUM_ROLLOUT = 1000
FINAL_ROLLOUT_ID = NUM_ROLLOUT - 1
GLOBAL_BATCH_SIZE = 256
ROLLOUT_BATCH_SIZE = 32
N_SAMPLES_PER_PROMPT = 8
ROLLOUT_MAX_RESPONSE_LEN = 512
ROLLOUT_TEMPERATURE = 1.0
MAX_SEARCH_TURNS = 2
SEARCH_TOP_K = 3
SEED = 123
ACTOR_GPUS = 4
ROLLOUT_GPUS = 4
TENSOR_PARALLEL_SIZE = 2
EVAL_RECORDS = 512

GENERATE_HOOK = "generate_with_search.generate"
REWARD_HOOK = "generate_with_search.reward_func"
ALGORITHM_KEYS = {
    "adam_beta1",
    "adam_beta2",
    "advantage_estimator",
    "calculate_per_token_loss",
    "entropy_coef",
    "eps_clip",
    "eps_clip_high",
    "kl_loss_coef",
    "kl_loss_type",
    "learning_rate",
    "lr_decay_style",
    "max_grad_norm",
    "normalize_advantages",
    "optimizer",
    "tis_clip",
    "tis_clip_low",
    "use_kl_loss",
    "use_tis",
    "weight_decay",
}

BASELINE_ALGORITHM: dict[str, Any] = {
    "advantage_estimator": "grpo",
    "use_kl_loss": True,
    "kl_loss_coef": 0.001,
    "kl_loss_type": "low_var_kl",
    "entropy_coef": 0.0,
    "eps_clip": 0.2,
    "eps_clip_high": 0.28,
    "normalize_advantages": False,
    "use_tis": False,
    "tis_clip": 2.0,
    "tis_clip_low": 0.0,
    "calculate_per_token_loss": False,
    "optimizer": "adam",
    "learning_rate": 1e-6,
    "lr_decay_style": "constant",
    "weight_decay": 0.01,
    "adam_beta1": 0.9,
    "adam_beta2": 0.98,
    "max_grad_norm": 1.0,
}

_CHOICES = {
    "advantage_estimator": {"grpo", "gspo", "cispo", "ppo", "reinforce_plus_plus", "reinforce_plus_plus_baseline"},
    "kl_loss_type": {"k1", "k2", "k3", "low_var_kl"},
    "optimizer": {"adam"},
    "lr_decay_style": {"constant", "linear", "cosine"},
}
_FLOAT_KEYS = ALGORITHM_KEYS - set(_CHOICES) - {
    "calculate_per_token_loss",
    "normalize_advantages",
    "use_kl_loss",
    "use_tis",
}
_BOOL_KEYS = {
    "calculate_per_token_loss",
    "normalize_advantages",
    "use_kl_loss",
    "use_tis",
}
_CLI_NAMES = {
    "learning_rate": "lr",
    "max_grad_norm": "clip_grad",
}


def validate_algorithm_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a validated participant algorithm table."""

    if not isinstance(config, dict):
        raise ValueError("algorithm config must be a table")
    unknown = sorted(set(config) - ALGORITHM_KEYS)
    if unknown:
        raise ValueError(f"{unknown[0]} is not an editable algorithm key")
    merged = dict(BASELINE_ALGORITHM)
    merged.update(config)
    for key, choices in _CHOICES.items():
        if merged[key] not in choices:
            raise ValueError(f"invalid {key}")
    for key in _BOOL_KEYS:
        if not isinstance(merged[key], bool):
            raise ValueError(f"{key} must be boolean")
    for key in _FLOAT_KEYS:
        value = merged[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{key} must be finite numeric")
    if merged["advantage_estimator"] == "ppo":
        raise ValueError("ppo is incompatible with the frozen no-critic topology")
    if merged["advantage_estimator"] in {"reinforce_plus_plus", "reinforce_plus_plus_baseline"} and not merged[
        "normalize_advantages"
    ]:
        raise ValueError("REINFORCE++ requires normalize_advantages=true")
    if merged["learning_rate"] <= 0 or merged["eps_clip"] < 0 or merged["eps_clip_high"] < merged["eps_clip"]:
        raise ValueError("invalid optimizer or clipping range")
    if merged["tis_clip_low"] < 0 or merged["tis_clip"] < merged["tis_clip_low"]:
        raise ValueError("invalid TIS clipping range")
    return dict(config)


def _fallback_flat_toml(text: str) -> dict[str, Any]:
    in_algorithm = False
    result: dict[str, Any] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            in_algorithm = line == "[algorithm]"
            continue
        if not in_algorithm or "=" not in line:
            continue
        key, raw_value = (part.strip() for part in line.split("=", 1))
        normalized = raw_value.replace("true", "True").replace("false", "False")
        try:
            result[key] = literal_eval(normalized)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"invalid algorithm TOML value on line {line_number}") from exc
    return result


def load_algorithm_config(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import tomllib
    except ImportError:
        algorithm = _fallback_flat_toml(text)
    else:
        document = tomllib.loads(text)
        extra_tables = set(document) - {"algorithm"}
        if extra_tables:
            raise ValueError(f"unexpected TOML table: {sorted(extra_tables)[0]}")
        algorithm = document.get("algorithm", {})
    return validate_algorithm_config(algorithm)


def _validated_attempt_output(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    relative = None
    for root in (Path(ATTEMPT_OUTPUT_ROOT), Path(TRUSTED_RUN_ROOT)):
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        break
    if relative is None or len(relative.parts) != 2 or relative.parts[1] != "train" or relative.parts[0] in {"", ".", ".."}:
        raise ValueError("attempt output must be an approved <root>/<id>/train path")
    return path


def _append_value(args: list[str], flag: str, value: Any) -> None:
    if isinstance(value, bool):
        if value:
            args.append(flag)
        return
    args.extend((flag, str(value)))


def build_training_arguments(config: dict[str, Any], output_dir: str | Path) -> list[str]:
    """Build Slime's train.py arguments around immutable task inputs."""

    participant = validate_algorithm_config(config)
    algorithm = dict(BASELINE_ALGORITHM)
    algorithm.update(participant)
    output = _validated_attempt_output(output_dir)
    args = [
        "--actor-num-nodes", "1",
        "--actor-num-gpus-per-node", str(ACTOR_GPUS),
        "--rollout-num-gpus", str(ROLLOUT_GPUS),
        "--colocate",
        "--swiglu",
        "--num-layers", "36",
        "--hidden-size", "2048",
        "--ffn-hidden-size", "11008",
        "--num-attention-heads", "16",
        "--use-rotary-position-embeddings",
        "--disable-bias-linear",
        "--add-qkv-bias",
        "--normalization", "RMSNorm",
        "--norm-epsilon", "1e-6",
        "--rotary-base", "1000000",
        "--group-query-attention",
        "--num-query-groups", "2",
        "--vocab-size", "151936",
        "--hf-checkpoint", MODEL_PATH,
        "--load", REFERENCE_LOAD_PATH,
        "--ref-load", REFERENCE_LOAD_PATH,
        "--save", str(output / "torch_dist"),
        "--save-interval", str(NUM_ROLLOUT),
        "--prompt-data", TRAIN_DATA_PATH,
        "--input-key", "prompt",
        "--label-key", "reward_model",
        "--apply-chat-template",
        "--rollout-shuffle",
        "--num-rollout", str(NUM_ROLLOUT),
        "--rollout-batch-size", str(ROLLOUT_BATCH_SIZE),
        "--n-samples-per-prompt", str(N_SAMPLES_PER_PROMPT),
        "--rollout-max-response-len", str(ROLLOUT_MAX_RESPONSE_LEN),
        "--rollout-temperature", str(ROLLOUT_TEMPERATURE),
        "--rollout-top-p", "1.0",
        "--rollout-top-k", "-1",
        "--global-batch-size", str(GLOBAL_BATCH_SIZE),
        "--balance-data",
        "--tensor-model-parallel-size", str(TENSOR_PARALLEL_SIZE),
        "--sequence-parallel",
        "--pipeline-model-parallel-size", "1",
        "--context-parallel-size", "1",
        "--expert-model-parallel-size", "1",
        "--expert-tensor-parallel-size", "1",
        "--recompute-granularity", "full",
        "--recompute-method", "uniform",
        "--recompute-num-layers", "1",
        "--use-dynamic-batch-size",
        "--max-tokens-per-gpu", "9216",
        "--rollout-num-gpus-per-engine", "2",
        "--sglang-mem-fraction-static", "0.7",
        "--attention-dropout", "0.0",
        "--hidden-dropout", "0.0",
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32",
        "--attention-backend", "flash",
        "--custom-generate-function-path", GENERATE_HOOK,
        "--custom-rm-path", REWARD_HOOK,
        "--seed", str(SEED),
        "--rollout-seed", str(SEED),
        "--sglang-enable-deterministic-inference",
    ]
    for key in sorted(algorithm):
        flag = "--" + _CLI_NAMES.get(key, key).replace("_", "-")
        _append_value(args, flag, algorithm[key])
    return args


def algorithm_source_sha256(path: str | Path) -> str:
    """Hash the sole declarative participant config, rejecting every extra entry."""

    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("algorithm root must be a regular directory")
    entries = list(root.iterdir())
    if len(entries) != 1 or entries[0].name != "algorithm_config.toml":
        raise ValueError("algorithm root may contain only algorithm_config.toml")
    file_path = entries[0]
    if file_path.is_symlink() or not file_path.is_file():
        raise ValueError("algorithm config must be a regular file")
    digest = hashlib.sha256()
    digest.update(b"algorithm_config.toml\0")
    digest.update(file_path.read_bytes())
    digest.update(b"\0")
    return digest.hexdigest()


def canonical_command_sha256(command: list[str]) -> str:
    return hashlib.sha256(json.dumps(command, separators=(",", ":")).encode("utf-8")).hexdigest()
