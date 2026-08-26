"""Pure scoring and checkpoint validation for Search-R1 evaluation."""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import math
from pathlib import Path
from typing import Any, Iterable

try:
    from task_contract import BM25_INDEX_PATH, EVAL_RECORDS, MODEL_PATH, SEARCH_TOP_K, WIKI_CORPUS_PATH
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from task_contract import BM25_INDEX_PATH, EVAL_RECORDS, MODEL_PATH, SEARCH_TOP_K, WIKI_CORPUS_PATH


def build_bm25_command() -> list[str]:
    return [
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
    ]


def build_sglang_command(model_path: str | Path, port: int) -> list[str]:
    return [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model_path),
        "--tokenizer-path",
        MODEL_PATH,
        "--tp-size",
        "2",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--mem-fraction-static",
        "0.8",
        "--enable-deterministic-inference",
    ]


def reward_label(record: dict[str, Any]) -> dict[str, Any]:
    """Validate and preserve the label shape consumed by pinned reward_func."""

    reward_model = record.get("reward_model")
    if not isinstance(reward_model, dict) or not isinstance(reward_model.get("ground_truth"), dict):
        raise ValueError("evaluation row lacks reward_model.ground_truth")
    target = reward_model["ground_truth"].get("target")
    if isinstance(target, str):
        target = [target]
    if not isinstance(target, list) or not target or not all(isinstance(value, str) and value for value in target):
        raise ValueError("evaluation ground truth target is invalid")
    return {"ground_truth": {"target": target}}


def stage_candidate_checkpoint(
    source: str | Path,
    target: str | Path,
    frozen_base: str | Path,
    expected_config: dict[str, Any],
) -> Path:
    """Snapshot only candidate weights beside frozen inference configuration."""

    source = Path(source)
    target = Path(target)
    frozen_base = Path(frozen_base)
    validate_checkpoint_tree(source, expected_config, tokenizer_reference=frozen_base)
    if target.exists() or target.is_symlink():
        raise ValueError("candidate staging target must not already exist")
    target.mkdir(mode=0o700)

    index_source = source / "model.safetensors.index.json"
    index = json.loads(index_source.read_text(encoding="utf-8"))
    weight_names = sorted(set(index["weight_map"].values()))
    shutil.copyfile(index_source, target / index_source.name)
    for name in weight_names:
        _regular_file(source / name)
        shutil.copyfile(source / name, target / name)
    for name in ("config.json", "generation_config.json"):
        base_asset = frozen_base / name
        if base_asset.exists():
            _regular_file(base_asset)
            shutil.copyfile(base_asset, target / name)
    for entry in target.iterdir():
        os.chmod(entry, 0o400)
    os.chmod(target, 0o500)
    return target


def aggregate_scores(values: Iterable[float]) -> float:
    scores = list(values)
    if len(scores) != EVAL_RECORDS:
        raise ValueError(f"evaluation requires exactly {EVAL_RECORDS} row scores")
    normalized = []
    for value in scores:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("row scores must be finite numbers")
        value = float(value)
        if value < 0.0 or value > 1.0:
            raise ValueError("row scores must be within [0, 1]")
        normalized.append(value)
    result = sum(normalized) / len(normalized)
    if not math.isfinite(result):
        raise ValueError("aggregate score is non-finite")
    return result


def normalized_reward(candidate: float, base: float, reference: float) -> float:
    values = (candidate, base, reference)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in values):
        raise ValueError("dynamic anchor metrics must be finite numeric values")
    candidate, base, reference = (float(value) for value in values)
    if reference <= base:
        raise ValueError("released GRPO anchor must improve over the base model")
    return min(1.0, max(0.0, (candidate - base) / (reference - base)))


def _regular_file(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"checkpoint entry must be a regular file: {path.name}")


ARCHITECTURE_CONFIG_KEYS = (
    "architectures",
    "model_type",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
    "max_position_embeddings",
    "rope_theta",
    "hidden_act",
    "rms_norm_eps",
    "tie_word_embeddings",
    "use_sliding_window",
    "sliding_window",
)
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)
EXECUTABLE_SUFFIXES = {".py", ".pyc", ".pyo", ".pth", ".so", ".dll", ".dylib"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_checkpoint_tree(
    path: str | Path,
    expected_config: dict[str, Any],
    *,
    tokenizer_reference: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("checkpoint root must be a regular directory")
    for entry in root.rglob("*"):
        if entry.is_symlink():
            raise ValueError("checkpoint tree must contain regular entries only")
        if entry.is_file() and entry.suffix.lower() in EXECUTABLE_SUFFIXES:
            raise ValueError("checkpoint contains executable or remote-code content")
    config_path = root / "config.json"
    tokenizer_path = root / "tokenizer.json"
    index_path = root / "model.safetensors.index.json"
    for required in (config_path, tokenizer_path, index_path):
        _regular_file(required)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("model config must be an object")
    if "auto_map" in config:
        raise ValueError("checkpoint remote-code auto_map is prohibited")
    for key in ARCHITECTURE_CONFIG_KEYS:
        if key not in expected_config:
            continue
        if config.get(key) != expected_config.get(key):
            raise ValueError(f"checkpoint {key} differs from the frozen base")
    if tokenizer_reference is not None:
        reference = Path(tokenizer_reference)
        for name in TOKENIZER_FILES:
            expected_path = reference / name
            candidate_path = root / name
            if expected_path.exists():
                _regular_file(expected_path)
                _regular_file(candidate_path)
                if _sha256(candidate_path) != _sha256(expected_path):
                    raise ValueError(f"checkpoint tokenizer artifact differs from the frozen base: {name}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("checkpoint weight map is empty")
    indexed_files = set()
    for name in weight_map.values():
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("weight map contains an unsafe path")
        indexed_files.add(name)
    for name in indexed_files:
        _regular_file(root / name)
    actual_weights = {entry.name for entry in root.glob("*.safetensors") if entry.is_file()}
    if actual_weights != indexed_files:
        raise ValueError("checkpoint contains missing or unindexed safetensors weights")
    return config
