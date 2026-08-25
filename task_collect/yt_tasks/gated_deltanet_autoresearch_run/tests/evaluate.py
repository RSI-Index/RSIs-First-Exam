#!/usr/bin/env python3
"""Frozen final evaluator for fixed-time Gated DeltaNet autoresearch.

Unlike the recipe-task evaluator, the candidate is EXPECTED to change the
model architecture, so the model class and config come from the submitted
candidate workspace. Everything that determines the score stays frozen here:

- the validation files, shuffle seed, partitioning, batch size, and batch
  count come from the pristine release protocol;
- the data loader is imported from the PRISTINE /opt/project copy, never
  from the candidate workspace;
- the loss is the release protocol's exact computation
  (torch.nn.functional.cross_entropy with ignore_index=-1 over the full
  sequence, equal to the repository chunked_cross_entropy with chunk_size=0);
- the sequence/vocabulary invariants are enforced before any batch is scored.

There is intentionally no parameter cap: under a fixed training clock, model
size, memory and throughput are part of the optimization rather than matched
controls. The pristine parameter count is reported only as context.

The candidate must remain loadable as lit_gpt.config.Config.from_name(NAME)
plus lit_gpt.model.GPT from its own workspace, where NAME is the
`model_config` string recorded in /app/output/provenance.json.
"""
from __future__ import annotations

import glob
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data import DataLoader


SEED = 3407
RELEASED_WORLD_SIZE = 32
VALIDATION_BLOCK_SIZE = 16_384 + 1
VALIDATION_BATCH_SIZE = 4
PRISTINE_PROJECT = Path("/opt/project")
ARCHITECTURE_INVARIANTS = {
    "block_size": 4096,
    "vocab_size": 32000,
    "padded_vocab_size": 32000,
}


def load_pristine_parameter_count() -> int:
    code = """
from lit_gpt.config import Config
from lit_gpt.model import GPT
model = GPT(Config.from_name('GatedDeltaNet_H1_0.4B'))
print('OFFICIAL_PARAMETER_COUNT=' + str(sum(p.numel() for p in model.parameters())))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PRISTINE_PROJECT)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PRISTINE_PROJECT),
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )
    marker = "OFFICIAL_PARAMETER_COUNT="
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(marker):
            return int(line[len(marker):])
    raise RuntimeError(f"pristine source did not report its parameter count: {result.stdout}")


def load_frozen_packed_dataset():
    """Import the PRISTINE PackedDataset module under a private name."""
    path = PRISTINE_PROJECT / "lit_gpt" / "packed_dataset.py"
    spec = importlib.util.spec_from_file_location("frozen_packed_dataset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def submitted_model_name() -> str:
    provenance_path = Path("/app/output/provenance.json")
    if not provenance_path.is_file():
        raise RuntimeError("missing /app/output/provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    name = provenance.get("model_config")
    if not isinstance(name, str) or not name.strip():
        raise RuntimeError(
            "provenance.json must record model_config: the exact "
            "Config.from_name name of the submitted architecture"
        )
    return name.strip()


def build_candidate_model(model_name: str):
    candidate = Path("/app/project")
    sys.path.insert(0, str(candidate))
    from lit_gpt.config import Config
    from lit_gpt.model import GPT

    config = Config.from_name(model_name)
    mismatches = {
        key: {"required": value, "actual": getattr(config, key, None)}
        for key, value in ARCHITECTURE_INVARIANTS.items()
        if getattr(config, key, None) != value
    }
    if mismatches:
        raise RuntimeError(f"architecture invariants violated: {mismatches}")

    model = GPT(config)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    pristine_parameters = load_pristine_parameter_count()
    state_path = Path("/app/output/model_state.pt")
    if not state_path.is_file():
        raise RuntimeError("missing safe tensor-only /app/output/model_state.pt")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise RuntimeError("model_state.pt must contain a non-empty state_dict")
    for prefix in ("_forward_module.", "module."):
        if state and all(str(key).startswith(prefix) for key in state):
            state = {str(key)[len(prefix):]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model, parameters, pristine_parameters


def build_validation_loader(fabric: L.Fabric):
    frozen = load_frozen_packed_dataset()

    root = Path(
        os.environ.get(
            "GDN_VALIDATION_DATA_DIR",
            "/datasets/gated_deltanet-official/slimpajama/packed/slim",
        )
    )
    filenames = sorted(glob.glob(str(root / "validation*")))
    if not filenames:
        raise RuntimeError(f"no official packed validation files found in {root}")
    # pretrain.py shuffles the sorted filename list even when block shuffle is
    # disabled, then partitions it by global rank.
    random.seed(SEED)
    random.shuffle(filenames)
    packed = frozen.PackedDataset(
        filenames,
        n_chunks=2,  # ceil(8 / released nodes)
        block_size=VALIDATION_BLOCK_SIZE,
        shuffle=False,
        seed=SEED + fabric.global_rank,
        num_processes=fabric.world_size,
        process_rank=fabric.global_rank,
    )
    dataset = frozen.CombinedDataset(datasets=[packed], seed=SEED, weights=[1.0])
    loader = DataLoader(
        dataset,
        batch_size=VALIDATION_BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
    )
    return root, fabric.setup_dataloaders(loader)


def release_protocol_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Frozen release loss: chunked_cross_entropy(..., chunk_size=0)."""
    if isinstance(logits, list):
        logits = torch.cat(logits, dim=1)
    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)
    return torch.nn.functional.cross_entropy(logits, targets, ignore_index=-1)


@torch.no_grad()
def evaluate(fabric: L.Fabric, model, loader, eval_iters: int) -> list[float]:
    model.eval()
    losses = torch.zeros(eval_iters, 2, device=fabric.device)
    completed = 0
    for batch_index, val_data in enumerate(loader):
        if batch_index >= eval_iters:
            break
        for index, length in enumerate((2048, 4096)):
            input_ids = val_data[:, :length].contiguous()
            targets = val_data[:, 1:length + 1].contiguous()
            logits = model(input_ids)
            loss = release_protocol_loss(logits, targets)
            losses[batch_index, index] = loss.item()
        completed += 1
    if completed != eval_iters:
        raise RuntimeError(
            f"release validation requested {eval_iters} batches but completed {completed}"
        )
    return [float(value) for value in losses.mean(0).tolist()]


def main() -> None:
    official_checkpoints = sorted(Path("/app/output").rglob("final-model-ckpt.pth"))
    if len(official_checkpoints) != 1:
        raise RuntimeError(
            f"expected exactly one official final-model-ckpt.pth, found {len(official_checkpoints)}"
        )
    model_name = submitted_model_name()

    single_gpu_smoke = os.environ.get("GDN_SINGLE_GPU_VERIFIER_SMOKE") == "1"
    devices = 1 if single_gpu_smoke else torch.cuda.device_count()
    if single_gpu_smoke:
        fabric = L.Fabric(devices=1, precision="bf16-mixed")
    else:
        sys.path.insert(0, "/app/project")
        import lit_gpt.model as candidate_model_module

        wrap_classes = {candidate_model_module.Block}
        if hasattr(candidate_model_module, "MBlock"):
            wrap_classes.add(candidate_model_module.MBlock)
        strategy = FSDPStrategy(
            auto_wrap_policy=wrap_classes,
            state_dict_type="full",
            sharding_strategy="HYBRID_SHARD",
        )
        fabric = L.Fabric(devices=devices, strategy=strategy, precision="bf16-mixed")
    fabric.launch()
    if not single_gpu_smoke and fabric.world_size != RELEASED_WORLD_SIZE:
        raise RuntimeError(
            f"scored release validation requires world size 32, found {fabric.world_size}"
        )
    fabric.seed_everything(SEED)

    with fabric.init_module(empty_init=False):
        model, parameters, pristine_parameters = (
            build_candidate_model(model_name)
        )
    model = fabric.setup(model)
    validation_root, loader = build_validation_loader(fabric)
    eval_iters = int(os.environ.get("GDN_EVAL_ITERS", "15"))
    if eval_iters <= 0:
        raise RuntimeError("GDN_EVAL_ITERS must be positive")
    losses = evaluate(fabric, model, loader, eval_iters)

    if fabric.global_rank == 0:
        geomean_loss = (losses[0] + losses[1]) / 2.0
        metrics = {
            "metric": "NVlabs release validation perplexity, geometric mean of 1x and 2x",
            "model_name": model_name,
            "checkpoint": official_checkpoints[0].relative_to("/app/output").as_posix(),
            "parameters": parameters,
            "pristine_parameters": pristine_parameters,
            "parameter_ratio_to_pristine": parameters / pristine_parameters,
            "validation_data": validation_root.as_posix(),
            "seed": SEED,
            "world_size": fabric.world_size,
            "eval_batches": eval_iters,
            "validation_batch_size": VALIDATION_BATCH_SIZE,
            "val_loss@1x": losses[0],
            "val_ppl@1x": math.exp(losses[0]),
            "val_loss@2x": losses[1],
            "val_ppl@2x": math.exp(losses[1]),
            "val_ppl_geomean": math.exp(geomean_loss),
            "length@1x": 2048,
            "length@2x": 4096,
            "run_kind": "single_gpu_smoke" if single_gpu_smoke else "fixed_time_4x8",
        }
        Path("/app/output/verifier-metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
    fabric.barrier()


if __name__ == "__main__":
    main()
