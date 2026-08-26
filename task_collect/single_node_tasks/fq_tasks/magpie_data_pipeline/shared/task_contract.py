"""Frozen scientific inputs for the Magpie data-pipeline benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
from types import MappingProxyType


ASSET_REVISIONS = MappingProxyType(
    {
        "magpie-align/magpie": "b734a36818e4ba7ef7f0f582fc55f7e860407d26",
        "meta-llama/Meta-Llama-3-70B-Instruct": "50fd307e57011801c7833c87efa1984ddf2db42f",
        "meta-llama/Meta-Llama-3-8B": "8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920",
        "Magpie-Align/Magpie-Pro-MT-300K-v0.1": "3c3045600d578b8f37413924b1b3bf117929755e",
        "Magpie-Align/Llama-3-8B-Magpie-Align-SFT-v0.1": "1ed587f54f70334f495efb9c027acb03e96fe24f",
        "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "Axolotl": "7c2bf3091f5e73c787afe839dfdcc8220b770a1a",
        "vLLM inference runtime": "643c125fab66d5ed5ec3143b7e764a77e7ae8ac7",
        "Transformers for Qwen 3.5": "943628458a1691f8af09c47ea9fc6e314734722f",
        "AlpacaEval code and prompts": "tatsu-lab/alpaca_eval@cd543a149df89434d8a54582c0151c0b945c3d20",
        "AlpacaEval 2 data": "tatsu-lab/alpaca_eval@2edc6fad8be6b14ea7230aabfd08188da6b8b814",
        "Arena-Hard code": "lmarena/arena-hard-auto@196f6b826783b3da7310e361a805fa36f0be83f3",
        "Arena-Hard data": "lmarena-ai/arena-hard-auto@15f3746e21432264ce9b453999bde4f3c946d2e6",
        "WildBench code": "allenai/WildBench@d6b8dcaf377d173d031980f97c16e1a82618c03d",
        "WildBench data": "allenai/WildBench@26c49eb39d7d5ce2099b0bbafed5a88dcce954ec",
    }
)

LLAMA3_CHAT_TEMPLATE_SHA256 = "4f46ecd43418f216df8c26936c60e5d04bcc81db47db5d86fbe889f64212dd4c"

RUNTIME_PATHS = MappingProxyType(
    {
        "generator_model": "/models/Meta-Llama-3-70B-Instruct",
        "student_model": "/models/Meta-Llama-3-8B",
        "baseline_dataset": "/task-assets/Magpie-Pro-MT-300K-v0.1.jsonl",
        "baseline_checkpoint": "/models/Llama-3-8B-Magpie-Align-SFT-v0.1",
        "judge_model": "/models/Qwen3.5-9B",
        "train_python": "/opt/train-venv/bin/python",
        "infer_python": "/opt/infer-venv/bin/python",
        "baseline_budget": "/task-assets/baseline_budget.json",
        "llama3_chat_template": "/task-tools/llama3_chat_template.jinja",
    }
)


def load_llama3_chat_template(path: str | Path | None = None) -> str:
    """Load the frozen official Llama-3 template instead of tokenizer metadata."""

    source = Path(path) if path is not None else Path(RUNTIME_PATHS["llama3_chat_template"])
    if not source.is_file() or source.is_symlink() or any(parent.is_symlink() for parent in source.parents):
        raise ValueError("frozen Llama-3 chat template must be an ordinary file")
    payload = source.read_bytes()
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise ValueError("frozen Llama-3 chat template source must have one final newline")
    try:
        template = payload[:-1].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("frozen Llama-3 chat template must be UTF-8") from exc
    if hashlib.sha256(template.encode("utf-8")).hexdigest() != LLAMA3_CHAT_TEMPLATE_SHA256:
        raise ValueError("frozen Llama-3 chat template hash mismatch")
    return template


@dataclass(frozen=True)
class ScientificContract:
    max_records: int = 300_000
    sequence_length: int = 8_192
    global_batch_size: int = 32
    micro_batch_size: int = 1
    epochs: int = 2
    validation_fraction: float = 0.001
    seed: int = 42
    sample_packing: bool = True
    evaluation_sample_packing: bool = False
    optimizer: str = "paged_adamw_8bit"
    learning_rate: float = 2e-5
    lr_scheduler: str = "cosine"
    warmup_steps: int = 100
    weight_decay: float = 0.0
    bf16: bool = True
    flash_attention: bool = True
    gradient_checkpointing: bool = True
    gradient_checkpointing_use_reentrant: bool = False
    assistant_response_only_loss: bool = True
    eval_per_source: int = 256
    judge_master_seed: int = 20_260_807
    generation_output_tokens: int = 2_048
    judge_temperature: float = 1.0
    judge_top_p: float = 0.95
    judge_top_k: int = 20
    judge_min_p: float = 0.0
    judge_presence_penalty: float = 1.5
    judge_repetition_penalty: float = 1.0
    judge_max_output_tokens: int = 32_768


CONTRACT = ScientificContract()
MAX_RECORDS = CONTRACT.max_records
SEQUENCE_LENGTH = CONTRACT.sequence_length


def assert_contract() -> None:
    """Check immutable values and their derived relationships on demand."""

    if CONTRACT.max_records != 300_000 or CONTRACT.sequence_length != 8_192:
        raise ValueError("record and sequence limits must match the frozen scientific contract")
    if CONTRACT.global_batch_size != 32 or CONTRACT.micro_batch_size != 1:
        raise ValueError("global and micro batch sizes must match the frozen scientific contract")
    if CONTRACT.epochs != 2 or CONTRACT.seed != 42:
        raise ValueError("epochs and seed must match the frozen scientific contract")
    if CONTRACT.validation_fraction != 0.001 or not CONTRACT.sample_packing or CONTRACT.evaluation_sample_packing:
        raise ValueError("validation and packing must match the frozen scientific contract")
    if (
        CONTRACT.optimizer != "paged_adamw_8bit"
        or CONTRACT.learning_rate != 2e-5
        or CONTRACT.lr_scheduler != "cosine"
        or CONTRACT.warmup_steps != 100
        or CONTRACT.weight_decay != 0.0
    ):
        raise ValueError("optimizer settings must match the frozen scientific contract")
    if (
        not CONTRACT.bf16
        or not CONTRACT.flash_attention
        or not CONTRACT.gradient_checkpointing
        or CONTRACT.gradient_checkpointing_use_reentrant
        or not CONTRACT.assistant_response_only_loss
    ):
        raise ValueError("execution settings must match the frozen scientific contract")
    if CONTRACT.eval_per_source != 256 or CONTRACT.judge_master_seed != 20_260_807:
        raise ValueError("evaluation contract must match the frozen scientific contract")
    if CONTRACT.generation_output_tokens != 2_048:
        raise ValueError("generation output cap must match the frozen scientific contract")
    if (
        CONTRACT.judge_temperature != 1.0
        or CONTRACT.judge_top_p != 0.95
        or CONTRACT.judge_top_k != 20
        or CONTRACT.judge_min_p != 0.0
        or CONTRACT.judge_presence_penalty != 1.5
        or CONTRACT.judge_repetition_penalty != 1.0
        or CONTRACT.judge_max_output_tokens != 32_768
    ):
        raise ValueError("judge settings must match the frozen scientific contract")
    if CONTRACT.global_batch_size % CONTRACT.micro_batch_size:
        raise ValueError("global batch must be divisible by micro batch size")
    if any(len(revision.rsplit("@", 1)[-1]) != 40 for revision in ASSET_REVISIONS.values()):
        raise ValueError("all pinned asset revisions must be full commits")
    if any(not PurePosixPath(path).is_absolute() for path in RUNTIME_PATHS.values()):
        raise ValueError("runtime paths must be absolute")
    if len(LLAMA3_CHAT_TEMPLATE_SHA256) != 64:
        raise ValueError("Llama-3 chat template must have a frozen SHA-256")


def accumulation_for_world_size(world_size: int) -> int:
    """Preserve the frozen global batch for a positive data-parallel world size."""

    assert_contract()
    if not isinstance(world_size, int) or isinstance(world_size, bool):
        raise ValueError("world size must divide frozen global batch 32")
    denominator = world_size * CONTRACT.micro_batch_size
    if world_size <= 0 or CONTRACT.global_batch_size % denominator:
        raise ValueError("world size must divide frozen global batch 32")
    return CONTRACT.global_batch_size // denominator
