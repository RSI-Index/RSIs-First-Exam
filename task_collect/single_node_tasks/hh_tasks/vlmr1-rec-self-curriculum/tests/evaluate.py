#!/usr/bin/env python3
"""Frozen evaluator for rsi/vlmr1-rec-self-curriculum.

Protocol is VLM-R1's own: the four helpers below are transcribed VERBATIM from
src/eval/test_rec_r1.py at commit 90052f478646e38bc67363862f699b7afca6b337 so that the
scored number is the repository's number and not a reimplementation of it. The additions
are (a) both the headline OOD set and the in-domain guard sets in one pass, (b) the
per-round budget assertion, and (c) the CK-1 base-weight identity check.

Nothing here reads provenance.json for anything it enforces. The reference package's
evaluator hardcoded a model name while its instruction.md promised a caller-declared one;
here the contract is explicit: the ARCHITECTURE and BASE are fixed by contract, the
curriculum is the free variable, and the evaluator does not need the agent's word for
anything it can hash.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# ---------------------------------------------------------------------------------------
# VERBATIM from src/eval/test_rec_r1.py -- do not "improve" these; they define the metric.
# ---------------------------------------------------------------------------------------
QUESTION_TEMPLATE = (
    "{Question} First output the thinking process in <think> </think> tags and then output "
    "the final answer in <answer> </answer> tags. Output the final answer in JSON format."
)


def extract_bbox_answer(content):
    answer_tag_pattern = r"<answer>(.*?)</answer>"
    bbox_pattern = r"\{.*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]\s*.*\}"
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        bbox_match = re.search(bbox_pattern, content_answer, re.DOTALL)
        if bbox_match:
            return [int(bbox_match.group(i)) for i in (1, 2, 3, 4)]
    return [0, 0, 0, 0]


def resize_bbox(bbox, input_height, input_width, image_height, image_width):
    bbox[0] = bbox[0] / input_width * image_width
    bbox[1] = bbox[1] / input_height * image_height
    bbox[2] = bbox[2] / input_width * image_width
    bbox[3] = bbox[3] / input_height * image_height
    return bbox


def iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2] - 1, box2[2] - 1)
    inter_y2 = min(box1[3] - 1, box2[3] - 1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
    else:
        inter = 0
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    return float(inter) / union


# ---------------------------------------------------------------------------------------
# Frozen protocol constants, mirrored from contract.yaml (contract_check.py asserts equality)
# ---------------------------------------------------------------------------------------
NUM_SAMPLES = 2000
SAMPLING_SEED = 42
IOU_THRESHOLD = 0.5
MAX_NEW_TOKENS = 256
BSZ = 2
HEADLINE_DATASET = "lisa_test"
GUARD_DATASETS = ("refcoco_val", "refcocop_val", "refcocog_val")
BUDGET_OPTIMIZER_STEPS = 150
BUDGET_COMPLETIONS_PER_STEP = 128
BUDGET_NUM_GENERATIONS = 8
BUDGET_MAX_COMPLETION_LENGTH = 2048


# The CK-1 hash lives in tests/dir_hash.py, not here, because environment/fetch_assets.sh also
# needs it (to record the staged model into the lockfile) and runs on a staging host with no
# torch. Two implementations of the same digest would make CK-1 fail on every round and look
# like tampering.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dir_hash import dir_manifest_sha256  # noqa: E402


# ---------------------------------------------------------------------------------------
# Asset staging preflight. This package ships NO data and NO weights -- only URLs -- so the
# state "mounts are empty" is reachable and must not be scoreable.
# ---------------------------------------------------------------------------------------
EVAL_REQUIRED = {
    "jsons": (4, "lisa_test + three val json"),
    "lisa": (100, "LISA-Grounding images"),
    "coco": (1000, "COCO train2014 images for the in-domain guard sets"),
}


def assert_assets_staged(eval_data: Path, base: Path,
                         manifest: Path = Path("/assets/manifest.yaml")) -> dict:
    """Refuse to produce a number from unstaged inputs.

    The failure this blocks is not a crash. An empty /eval-data/jsons raises on the first
    json.load, which is fine -- but an UNDER-populated one does not: it scores whatever subset
    arrived, at full confidence, and the number is silently computed over the wrong
    denominator. So the check is on counts, not on existence.
    """
    facts: dict = {"manifest_state": None, "counts": {}}
    if manifest.is_file():
        for line in manifest.read_text().splitlines():
            if line.startswith("state:"):
                # Inline comment stripped on purpose; see assets.yaml's note above `state:`.
                facts["manifest_state"] = line.split(":", 1)[1].split("#", 1)[0].strip()
                break
        if facts["manifest_state"] == "NOT_STAGED":
            raise RuntimeError(
                f"{manifest} says state: NOT_STAGED. The asset URLs were never fetched, so "
                "there is nothing to score. Run environment/fetch_assets.sh, fill the "
                "integrity fields, then flip the state -- in that order."
            )
    for name, (minimum, why) in EVAL_REQUIRED.items():
        d = eval_data / name
        if not d.is_dir():
            raise RuntimeError(f"missing eval mount {d} ({why}); assets are declared by URL "
                               f"only and must be bind-mounted read-only by the harness")
        n = sum(1 for p in d.rglob("*") if p.is_file())
        facts["counts"][name] = n
        if n < minimum:
            raise RuntimeError(
                f"{d} holds {n} file(s), expected >= {minimum} ({why}). A partially staged "
                f"eval mount yields a number computed over the wrong denominator; refusing."
            )
    if not base.is_dir() or not any(base.rglob("*.safetensors")):
        raise RuntimeError(f"base checkpoint not staged at {base}; CK-1 cannot be verified")
    return facts


def assert_budget(trainer_state: Path, expected_base_hash: str, base_dir: Path) -> dict:
    """B-2 and CK-1, enforced from the trainer's own counters, never from a claim.

    The agent cannot satisfy this by editing provenance.json: `trainer_state.json` is
    written by the HF trainer, and the base-weight hash is computed here over the staged
    read-only checkpoint.
    """
    if not trainer_state.is_file():
        raise RuntimeError(f"missing trainer counters at {trainer_state}; cannot verify the budget")
    state = json.loads(trainer_state.read_text())
    steps = int(state.get("global_step", -1))
    if steps != BUDGET_OPTIMIZER_STEPS:
        raise RuntimeError(
            f"per-round budget violated: global_step={steps}, contract requires exactly "
            f"{BUDGET_OPTIMIZER_STEPS}"
        )
    observed_base = dir_manifest_sha256(base_dir)
    if expected_base_hash not in ("MEASURE", observed_base):
        raise RuntimeError(
            "CK-1 violated: the staged base checkpoint does not match the contract hash "
            f"(expected {expected_base_hash}, observed {observed_base})"
        )
    return {"global_step": steps, "base_weight_sha256": observed_base}


def evaluate_dataset(model, processor, data_root: Path, image_root: Path, ds: str,
                     rank: int, world_size: int, device: str) -> float | None:
    data = json.loads((data_root / f"{ds}.json").read_text())
    random.seed(SAMPLING_SEED)
    random.shuffle(data)
    data = data[:NUM_SAMPLES]

    per_rank = len(data) // world_size
    start = rank * per_rank
    end = start + per_rank if rank < world_size - 1 else len(data)
    rank_data = data[start:end]

    messages = [
        [{"role": "user", "content": [
            {"type": "image", "image": f"file://{image_root / x['image']}"},
            {"type": "text", "text": QUESTION_TEMPLATE.format(Question=x["problem"])},
        ]}]
        for x in rank_data
    ]

    rank_outputs = []
    for i in range(0, len(messages), BSZ):
        batch = messages[i:i + BSZ]
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch]
        image_inputs, video_inputs = process_vision_info(batch)
        inputs = processor(text=texts, images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            generated = model.generate(**inputs, use_cache=True,
                                       max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        decoded = processor.batch_decode(trimmed, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
        input_h = inputs["image_grid_thw"][:, 1] * 14
        input_w = inputs["image_grid_thw"][:, 2] * 14
        for j, text in enumerate(decoded):
            rank_outputs.append((start + i + j, text, int(input_h[j]), int(input_w[j])))

    gathered: list = [None] * world_size
    dist.all_gather_object(gathered, rank_outputs)
    if rank != 0:
        return None

    slots: list = [None] * len(data)
    for chunk in gathered:
        for idx, text, ih, iw in chunk:
            slots[idx] = (text, ih, iw)
    if any(slot is None for slot in slots):
        raise RuntimeError(f"{ds}: incomplete gather; refusing to score a partial evaluation")

    correct = 0
    for example, (text, ih, iw) in zip(data, slots):
        from PIL import Image
        with Image.open(image_root / example["image"]) as img:
            image_w, image_h = img.size
        pred = resize_bbox(extract_bbox_answer(text), ih, iw, image_h, image_w)
        if iou(pred, example["solution"]) > IOU_THRESHOLD:
            correct += 1
    return correct / len(data) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, default=Path("/app/output/submission"))
    parser.add_argument("--base", type=Path, default=Path("/models/Qwen2.5-VL-3B-Instruct"))
    parser.add_argument("--eval-data", type=Path,
                        default=Path(os.environ.get("VLMR1_EVAL_DATA", "/eval-data")))
    parser.add_argument("--trainer-state", type=Path,
                        default=Path("/app/output/submission/trainer_state.json"))
    parser.add_argument("--expected-base-hash", default=os.environ.get("VLMR1_BASE_SHA256", "MEASURE"))
    parser.add_argument("--out", type=Path, default=Path("/app/output/verifier-metrics.json"))
    parser.add_argument("--skip-budget-check", action="store_true",
                        help="ONLY for measuring the pristine-base anchor, which has no training run")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    world_size, rank = dist.get_world_size(), dist.get_rank()
    device = f"cuda:{local_rank}"

    # Before anything expensive, and before rank 0 has a chance to write a metrics file.
    assets = assert_assets_staged(args.eval_data, args.base)

    budget = {"skipped": True}
    if not args.skip_budget_check:
        budget = assert_budget(args.trainer_state, args.expected_base_hash, args.base)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.submission, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map={"": local_rank})
    processor = AutoProcessor.from_pretrained(args.submission)

    scores: dict[str, float] = {}
    for ds in (HEADLINE_DATASET, *GUARD_DATASETS):
        image_root = args.eval_data / ("lisa" if ds == "lisa_test" else "coco")
        value = evaluate_dataset(model, processor, args.eval_data / "jsons", image_root,
                                 ds, rank, world_size, device)
        if rank == 0:
            scores[ds] = value

    if rank == 0:
        guard = sum(scores[d] for d in GUARD_DATASETS) / len(GUARD_DATASETS)
        metrics = {
            "metric": "VLM-R1 REC accuracy @ IoU>0.5, frozen protocol",
            "lisa_test_acc_iou50": scores[HEADLINE_DATASET],
            "indomain_avg_acc_iou50": guard,
            "per_dataset": scores,
            "samples_per_dataset": NUM_SAMPLES,
            "sampling_seed": SAMPLING_SEED,
            "iou_threshold": IOU_THRESHOLD,
            "decode": "greedy",
            "world_size": world_size,
            "budget": budget,
            "assets": assets,
        }
        args.out.write_text(json.dumps(metrics, indent=2) + "\n")
    dist.barrier()


if __name__ == "__main__":
    main()
