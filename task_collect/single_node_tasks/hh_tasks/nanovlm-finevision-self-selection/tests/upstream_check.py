#!/usr/bin/env python3
"""Verify contract.yaml's claims about upstream, at IMAGE BUILD time, in both images.

This package holds no clone of nanoVLM -- it is configuration only -- so the claims in
contract.yaml cannot be checked from inside the package. They are checked here instead, against
the freshly cloned and patched /opt/project, and a failure means the image does not build. That is
the right severity: a contract that cites a mechanism the code no longer has is not a small
documentation problem, it is a task description that no longer describes the task.

Two classes of check, deliberately different in severity:

  SYMBOLS   must exist. A rename means the mechanism moved and the contract is stale.
  PATCH     must be exactly the declared shape: the pinned commit, the declared file set, the
            declared hunk counts, and the ordering the frozen-val-split rule depends on.

Line numbers cited in contract.yaml are NOT checked. They are hints, they rot on any insertion,
and a checker that fails on a shifted line teaches people to pass --warn-only. Every line number
in the first draft of contract.yaml for models/config.py was wrong by 2-6 lines; the symbols were
all correct. That asymmetry is the argument.

Usage:  python tests/upstream_check.py --project /opt/project --contract /tests/contract.yaml
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml

# (relative path, regex, what it is). Every pattern ends in \s*\( or \b so a rename cannot satisfy
# it: `def\s+iou_reward` matched `def iou_reward_v2` in the sibling package, meaning the checker
# passed on exactly the drift it existed to catch.
CLAIMS = [
    ("data/datasets.py", r"class\s+BaseDataset\b",
     "the dataset wrapper the rating thresholds live in"),
    ("data/datasets.py", r"def\s+_get_messages\s*\(",
     "per-TURN rating filter; the selection baseline this task is built on"),
    ("data/datasets.py", r"relevance_ratings", "rating field 1 of 4"),
    ("data/datasets.py", r"image_correspondence_ratings", "rating field 2 of 4"),
    ("data/datasets.py", r"visual_dependency_ratings", "rating field 3 of 4"),
    ("data/datasets.py", r"formatting_ratings", "rating field 4 of 4"),
    ("train.py", r"def\s+get_dataloaders\s*\(", "pool load, val split and VQADataset construction"),
    ("train.py", r"total_shards\s*=\s*56",
     "the hardcoded shard count the staged pool must match exactly"),
    ("train.py", r"load_from_disk\s*\(", "the local-shard branch the staged pool is consumed by"),
    ("train.py", r"while\s+global_step\s*<\s*train_cfg\.max_training_steps",
     "the step budget, enforced upstream (half 1 of 2)"),
    ("train.py", r"if\s+global_step\s*>=\s*train_cfg\.max_training_steps",
     "the step budget, enforced upstream (half 2 of 2)"),
    ("train.py", r"total_samples_processed\s*=", "upstream's own samples-seen definition"),
    ("data/advanced_datasets.py", r"class\s+ConstantLengthDataset\b",
     "knapsack packing; why samples_seen is approximate and the STEP count is the budget"),
    ("models/config.py", r"class\s+TrainConfig\b", "the frozen recipe"),
    ("models/config.py", r"class\s+VLMConfig\b", "the frozen architecture"),
]

# What the adapter must look like after it is applied. The ordering check is the one that matters:
# contract.yaml chain.val_split_is_frozen claims the keep list is applied AFTER the val split, and
# if that were ever reversed every round would score against its own val slice while the contract
# still said otherwise.
PATCH_SHAPE = [
    ("train.py", r'add_column\(\s*"rsi_idx"', "stable pool row ids"),
    ("train.py", r"train_cfg\.rsi_keep_list", "the keep list is consumed"),
    ("train.py", r"--rsi_keep_list", "the flag exists"),
    ("train.py", r"--max_training_steps", "the step-budget flag exists"),
]


def sh(args: list[str], cwd: Path) -> str:
    return subprocess.run(args, cwd=str(cwd), check=True,
                          capture_output=True, text=True).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--warn-only", action="store_true",
                    help="report without failing the build. For local inspection only.")
    args = ap.parse_args()
    contract = yaml.safe_load(args.contract.read_text())
    problems: list[str] = []

    # ---- the pinned commit ----------------------------------------------------------------
    want = contract["pristine_base"]["commit"]
    head = sh(["git", "rev-parse", "HEAD"], args.project)
    if head != want:
        problems.append(f"/opt/project HEAD is {head}, contract pins {want}")

    # ---- the patch is exactly the declared shape ------------------------------------------
    declared = {t["path"]: int(t["hunks"]) for t in contract["pristine_base"]["patch"]["touches"]}
    changed = [p for p in sh(["git", "diff", "--name-only"], args.project).splitlines() if p]
    if set(changed) != set(declared):
        problems.append(
            f"the working tree differs from the pinned commit in {sorted(changed)}, but "
            f"contract pristine_base.patch declares {sorted(declared)}. An undeclared "
            f"modification to the pristine base is indistinguishable from tampering."
        )
    for path in sorted(set(changed) & set(declared)):
        # --unified=0 counts CHANGE SITES, which is a property of the edit. The default 3 lines of
        # context coalesced the seven one-line edits in models/config.py into a single rendered
        # hunk, so a count taken from the .patch file measures how the diff was printed rather
        # than what it does. contract.yaml declares the zero-context count and says so.
        diff = sh(["git", "diff", "--unified=0", "--", path], args.project)
        sites = len(re.findall(r"^@@", diff, re.MULTILINE))
        if sites != declared[path]:
            problems.append(f"{path} has {sites} change sites at --unified=0, contract declares "
                            f"{declared[path]}")

    for rel, pattern, what in PATCH_SHAPE:
        text = (args.project / rel).read_text()
        if not re.search(pattern, text):
            problems.append(f"adapter missing from {rel}: {pattern} ({what})")

    # ---- THE ORDERING the frozen val split depends on --------------------------------------
    train_src = (args.project / "train.py").read_text()
    val_split = re.search(r"train_ds\s*=\s*train_ds\.select\(range\(val_size,", train_src)
    keep_apply = re.search(r"if\s+train_cfg\.rsi_keep_list", train_src)
    if not val_split or not keep_apply:
        problems.append("cannot locate both the val split and the keep-list application in "
                        "train.py; the ordering claim in contract.yaml is unverifiable")
    elif keep_apply.start() < val_split.start():
        problems.append(
            "the keep list is applied BEFORE the val split. contract.yaml "
            "chain.val_split_is_frozen claims the opposite, and with selection first every round "
            "holds out a different val slice, so no cross-round val comparison means anything."
        )

    # ---- symbols the contract's mechanism claims rest on ----------------------------------
    for rel, pattern, what in CLAIMS:
        path = args.project / rel
        if not path.is_file():
            problems.append(f"{rel} does not exist ({what})")
            continue
        if not re.search(pattern, path.read_text()):
            problems.append(f"{rel}: /{pattern}/ not found -- {what}")

    if problems:
        print("UPSTREAM CLAIM CHECK FAILED", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 0 if args.warn_only else 1
    print(f"upstream claims verified at {head[:12]}: {len(CLAIMS)} symbols, "
          f"{sum(declared.values())} patch hunks in {len(declared)} files, val-split ordering OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
