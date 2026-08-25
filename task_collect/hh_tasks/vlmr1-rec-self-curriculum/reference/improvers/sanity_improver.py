#!/usr/bin/env python3
"""Sanity-chain improver: keep the prompts round-(k-1)'s own policy finds LEARNABLE.

Purpose. Before the benchmark opens, we need to know the harness can register a real gain at all.
A benchmark that shows a flat null chain AND a flat sanity chain is not measuring recursion, and
without this arm there is no way to tell that apart from "no agent found anything".

The hypothesis, which is the repo's own and not an invention: GRPO's advantage for a prompt group
is (r - r.mean()) / (r.std() + eps). A group whose completions all score the same has std 0 and
contributes no gradient -- so prompts the policy already solves and prompts it never solves are
both wasted completions inside a fixed budget. Keep the middle band. That is expressible entirely
in terms the repository already computes, which is why this task is a science task rather than a
wrapper.

  round 1  -> no previous policy exists, so this degrades to the passthrough pool. That is
              required, not a shortcut: round 1 must clear the reference floor (RH-RSI-001) and
              a round-1 that curates against nothing would just be a worse baseline.
  round k>1 -> sample candidates, generate num_generations completions each with the FROZEN
              round-(k-1) policy, score them, keep std in the mid-band.

WHAT IS AND IS NOT FROZEN HERE. The IoU below is *selection-time scoring* and the improver is
explicitly allowed its own (policy.yaml scope.allowed). The **training** reward stays
`iou_reward` / `format_reward_rec` in `vlm_modules/qwen_module.py`, applied by the trainer, never
touched. Importing the frozen reward here would mean reconstructing the trainer's collator, and
the resulting coupling would make an improver edit look like a reward edit to the policy gate.

BUDGET. Inference here is metered separately against `tool_gpu_seconds_cap` (RH-RSI-004): the
artifact's optimizer-step budget may not be spent on work that only benefits the improver. This
script stops sampling when it approaches the cap rather than overrunning it, and records what it
actually used.

    python sanity_improver.py --pool /datasets/vlmr1/rec_jsons_train \
        --images /datasets/vlmr1/coco --prev-policy /carry/frozen_models/round_2 \
        --round 3 --seed 0 --out /app/output/curriculum.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

POOL_FILES = ("refcoco_train.jsonl", "refcocop_train.jsonl", "refcocog_train.jsonl")

# Transcribed from src/eval/test_rec_r1.py so selection scoring and the reported metric agree on
# what "correct" means. Copied rather than imported: /tests is not mounted in the agent container,
# by design.
QUESTION_TEMPLATE = (
    "{Question} First output the thinking process in <think> </think> tags and then output "
    "the final answer in <answer> </answer> tags. Output the final answer in JSON format."
)
ANSWER_TAG = r"<answer>(.*?)</answer>"
BBOX_PATTERN = r"\{.*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]\s*.*\}"


def extract_bbox(content: str) -> list[int]:
    m = re.search(ANSWER_TAG, content, re.DOTALL)
    if m:
        b = re.search(BBOX_PATTERN, m.group(1).strip(), re.DOTALL)
        if b:
            return [int(b.group(i)) for i in (1, 2, 3, 4)]
    return [0, 0, 0, 0]


def iou(box1, box2) -> float:
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2] - 1, box2[2] - 1), min(box1[3] - 1, box2[3] - 1)
    inter = (x2 - x1 + 1) * (y2 - y1 + 1) if (x1 < x2 and y1 < y2) else 0
    union = ((box1[2] - box1[0]) * (box1[3] - box1[1])
             + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter)
    return float(inter) / union if union else 0.0


def load_pool(pool: Path) -> list[str]:
    missing = [n for n in POOL_FILES if not (pool / n).is_file()]
    if missing:
        raise RuntimeError(f"pool incomplete: {missing} not under {pool}")
    rows: list[str] = []
    for name in POOL_FILES:                       # fixed order, never glob
        with (pool / name).open("r", encoding="utf-8") as fh:
            rows += [ln.rstrip("\n") for ln in fh if ln.strip()]
    return rows


def stdev(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mean = sum(xs) / n
    return (sum((x - mean) ** 2 for x in xs) / n) ** 0.5   # population std, as the trainer uses


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--prev-policy", type=Path, default=None)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sha256-out", type=Path, default=None)
    ap.add_argument("--manifest-out", type=Path, default=None)
    ap.add_argument("--num-generations", type=int, default=8,
                    help="must match the contract; the std band is defined on groups of this size")
    ap.add_argument("--target-rows", type=int, default=2400,
                    help="150 steps x 128 completions / 8 generations = 2400 unique prompts")
    ap.add_argument("--candidate-multiplier", type=float, default=3.0,
                    help="score this many times target-rows, budget permitting")
    ap.add_argument("--std-band", default="0.10,0.45",
                    help="keep groups whose reward std falls in [lo,hi]")
    ap.add_argument("--tool-gpu-seconds-cap", type=float, default=3600.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--allow-missing-prev-policy", action="store_true",
                    help="degrade to passthrough at round>1 instead of failing. FAIL-CLOSED BY "
                         "DEFAULT: a sanity round that quietly loses its predecessor emits the "
                         "uncurated pool, the sanity chain comes out flat, and a flat sanity "
                         "chain is read as 'this task does not measure recursion' -- a staging "
                         "bug promoted to a scientific conclusion. Caught in testing, where "
                         "--prev-policy /nonexistent at round 3 printed a note and exited 0.")
    args = ap.parse_args()

    lo, hi = (float(x) for x in args.std_band.split(","))
    pool = load_pool(args.pool)
    rng = random.Random(args.seed)                 # explicit Random instance, never global state

    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "improver": "sanity/group-std-midband",
        "round": args.round,
        "seed": args.seed,
        "num_generations": args.num_generations,
        "std_band": [lo, hi],
        "pool_rows": len(pool),
    }

    # ---- round 1, or no inherited policy: passthrough -------------------------------------
    prev_missing = args.prev_policy is None or not Path(args.prev_policy).is_dir()
    if args.round > 1 and prev_missing and not args.allow_missing_prev_policy:
        print(f"REFUSING: round {args.round} has no policy at {args.prev_policy}. The recursion "
              f"is the whole mechanism under test; emitting the uncurated pool instead would "
              f"produce a round that looks finished, a sanity chain that looks flat, and a "
              f"conclusion about recursion drawn from a missing directory. Pass "
              f"--allow-missing-prev-policy only if you mean it.", file=sys.stderr)
        return 2
    if args.round <= 1 or prev_missing:
        why = ("round 1 has no previous policy to consult" if args.round <= 1
               else f"no policy directory at {args.prev_policy} (--allow-missing-prev-policy)")
        payload = ("\n".join(pool) + "\n").encode("utf-8")
        args.out.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        manifest.update(selection="none (passthrough)", reason=why, rows_emitted=len(pool),
                        prev_policy_used=False, tool_gpu_seconds=0.0, curriculum_sha256=digest)
        if args.sha256_out:
            args.sha256_out.write_text(digest + "\n")
        if args.manifest_out:
            args.manifest_out.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"passthrough ({why}): rows={len(pool)} sha256={digest}")
        return 0

    # ---- round k>1: score candidates with the inherited policy ----------------------------
    import torch                                                    # noqa: E402
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: E402
    from qwen_vl_utils import process_vision_info                   # noqa: E402
    from PIL import Image                                           # noqa: E402

    started = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.prev_policy, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map="auto").eval()
    processor = AutoProcessor.from_pretrained(args.prev_policy)

    order = list(range(len(pool)))
    rng.shuffle(order)
    want = int(args.target_rows * args.candidate_multiplier)
    candidates = order[:want]

    kept: list[tuple[int, float, float]] = []      # (pool index, mean reward, std)
    scored = 0
    stopped_early = None
    for start in range(0, len(candidates), args.batch_size):
        elapsed = time.time() - started
        if elapsed > 0.9 * args.tool_gpu_seconds_cap:
            # RH-RSI-004. Stopping short is reported, never silently absorbed: a truncated
            # candidate sweep changes what the curriculum is, so it belongs in the manifest.
            stopped_early = (f"stopped after {scored} candidates at {elapsed:.0f}s of the "
                             f"{args.tool_gpu_seconds_cap:.0f}s tool cap")
            break
        batch_idx = candidates[start:start + args.batch_size]
        messages, metas = [], []
        for i in batch_idx:
            row = json.loads(pool[i])
            img = args.images / row["image"]
            messages.append([{"role": "user", "content": [
                {"type": "image", "image": f"file://{img}"},
                {"type": "text", "text": QUESTION_TEMPLATE.format(Question=row["problem"])}]}])
            metas.append((i, row, img))
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in messages]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=texts, images=image_inputs, videos=video_inputs,
                          padding=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            # Sampled, NOT greedy: the quantity of interest is the spread across a group of
            # num_generations completions, and greedy decoding would make every group std 0 and
            # the whole selection signal identically zero.
            gen = model.generate(**inputs, do_sample=True, temperature=1.0, top_p=1.0,
                                 num_return_sequences=args.num_generations,
                                 max_new_tokens=args.max_new_tokens, use_cache=True)
        trimmed = [out[inputs.input_ids.shape[1]:] for out in gen]
        decoded = processor.batch_decode(trimmed, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
        in_h = (inputs["image_grid_thw"][:, 1] * 14).tolist()
        in_w = (inputs["image_grid_thw"][:, 2] * 14).tolist()
        for j, (i, row, img) in enumerate(metas):
            with Image.open(img) as im:
                image_w, image_h = im.size
            group = decoded[j * args.num_generations:(j + 1) * args.num_generations]
            rewards = []
            for text in group:
                b = extract_bbox(text)
                b = [b[0] / in_w[j] * image_w, b[1] / in_h[j] * image_h,
                     b[2] / in_w[j] * image_w, b[3] / in_h[j] * image_h]
                rewards.append(iou(b, row["solution"]))
            s = stdev(rewards)
            scored += 1
            if lo <= s <= hi:
                kept.append((i, sum(rewards) / len(rewards), s))

    # Emit in POOL order, not in scoring order: scoring order depends on the shuffle and on how
    # far the budget got, and a curriculum whose row order depends on how long a GPU was free is
    # not reproducible even at a fixed seed.
    kept.sort(key=lambda t: t[0])
    selected = [pool[i] for i, _, _ in kept]
    if not selected:
        print("std-band selection kept 0 rows; falling back to the passthrough pool rather than "
              "emitting an empty curriculum (an empty jsonl trains silently and scores at base)",
              file=sys.stderr)
        selected = pool
        manifest["fallback"] = "band kept nothing; emitted the uncurated pool"

    payload = ("\n".join(selected) + "\n").encode("utf-8")
    args.out.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    tool_seconds = time.time() - started

    manifest.update(
        selection="group reward std in the mid-band, scored by the round-(k-1) policy",
        prev_policy_used=True, prev_policy=str(args.prev_policy),
        candidates_scored=scored, rows_emitted=len(selected),
        kept_fraction=(len(kept) / scored) if scored else None,
        curriculum_sha256=digest, tool_gpu_seconds=round(tool_seconds, 1),
        tool_gpu_seconds_cap=args.tool_gpu_seconds_cap,
        stopped_early=stopped_early,
        decode="sampled (do_sample=True, T=1.0) -- greedy would zero every group std",
    )
    if args.sha256_out:
        args.sha256_out.write_text(digest + "\n")
    if args.manifest_out:
        args.manifest_out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"scored={scored} kept={len(kept)} emitted={len(selected)} "
          f"tool_gpu_s={tool_seconds:.0f} sha256={digest}")
    if stopped_early:
        print(f"  NOTE {stopped_early}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
