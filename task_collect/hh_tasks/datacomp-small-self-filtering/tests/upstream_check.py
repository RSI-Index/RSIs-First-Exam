#!/usr/bin/env python3
"""Verify contract.yaml's claims about upstream, at IMAGE BUILD time, in both images.

This package holds no clone of datacomp -- it is configuration only -- so its claims cannot be
checked from inside it. They are checked here against the freshly cloned /opt/project, and a failure
means the image does not build.

The most important check is not a symbol: it is that SCALE_CONFIGS["small"] still holds the exact
recipe contract.yaml frozen_recipe declares. That dict IS the per-round budget. A change to it --
upstream's or a local one -- would redefine the budget while task.toml, policy.yaml and
instruction.md all still described the old one, and every round of every chain would be off-budget
in a way no other check would notice.

Line numbers in contract.yaml are hints and are NOT checked here. They rot on any insertion, and a
checker that fails on a shifted line teaches people to pass --warn-only. Symbols and values are
checked; positions are not.

Usage:  python tests/upstream_check.py --project /opt/project --contract /tests/contract.yaml
"""
from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

import yaml

# (relative path, regex, what it is). Patterns end in \b, \s*\( or = so a rename cannot satisfy them:
# `def\s+iou_reward` matched `def iou_reward_v2` in a sibling package, so the checker passed on
# exactly the drift it existed to catch.
CLAIMS = [
    ("baselines.py", r"^BASELINES\s*=", "the seven shipped filters, including no_filter"),
    ("baselines.py", r"\bno_filter\b", "the pristine training arm's filter"),
    ("baselines.py", r"\bclip_score\b", "the zero-GPU-cost baseline that ships with the metadata"),
    ("baselines.py", r"def\s+check_args\s*\(", "which flag combinations upstream itself rejects"),
    ("resharder.py", r"argparse", "the step from a uid .npy to a webdataset"),
    ("scale_configs.py", r"^SCALE_CONFIGS\s*=", "the frozen recipe"),
    ("scale_configs.py", r"def\s+get_scale_config\s*\(", "how train.py reads it"),
    ("train.py", r"from\s+training\.main\s+import\s+main",
     "open_clip's INTERNAL entry point, imported not shelled out -- why open_clip must be pinned"),
    ("train.py", r"--dataset-resampled",
     "resampling with replacement: why a small subset still costs the full budget"),
    ("train.py", r"train_num_samples\s*=\s*config\[",
     "the budget comes from the scale config, not from the subset size"),
    ("aggregate_scores.py", r"def\s+get_aggregate_scores\s*\(", "the headline mean, imported not copied"),
    ("aggregate_scores.py", r"assert\s+len\(df\)\s*==\s*38",
     "THE 38 ASSERT. Reimplementing the mean would discard this check along with the code."),
    ("aggregate_scores.py", r"dropna\(subset=\[\"main_metric\"\]\)",
     "how FairFace and UTKFace fall out of the 40 implicitly"),
    ("evaluate.py", r"eval_results\.jsonl", "the results file tests/evaluate.py reads"),
    ("eval_utils/main.py", r"fairness/", "the fairness branch, 2 of whose 4 tasks are in the 38"),
    ("download_upstream.py", r"argparse", "the pool fetcher"),
    ("download_evalsets.py", r"argparse", "the evaluation-set fetcher"),
    ("tasklist.yml", r"^imagenet1k:", "the guard dataset's task key"),
]


def sh(args: list[str], cwd: Path) -> str:
    return subprocess.run(args, cwd=str(cwd), check=True, capture_output=True, text=True).stdout.strip()


def read_scale_configs(path: Path) -> dict:
    """Parse SCALE_CONFIGS without importing it, so this runs with no dependencies."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "SCALE_CONFIGS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("SCALE_CONFIGS not found in scale_configs.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--warn-only", action="store_true", help="report without failing. Local use only.")
    args = ap.parse_args()
    contract = yaml.safe_load(args.contract.read_text())
    problems: list[str] = []

    want_commit = contract["pristine_base"]["commit"]
    head = sh(["git", "rev-parse", "HEAD"], args.project)
    if head != want_commit:
        problems.append(f"/opt/project HEAD is {head}, contract pins {want_commit}")

    # This package declares patch: none, so ANY local modification is undeclared.
    if contract["pristine_base"].get("patch") not in (None, "none"):
        problems.append("contract declares a patch but this package's Dockerfiles apply none")
    dirty = [p for p in sh(["git", "status", "--porcelain"], args.project).splitlines() if p]
    if dirty:
        problems.append(f"/opt/project has local modifications while contract declares patch: none "
                        f"-> {dirty[:5]}")

    # ---- THE RECIPE. This is the per-round budget, so it is checked by value. -----------------
    recipe = contract["frozen_recipe"]
    scale = recipe["scale"]
    configs = read_scale_configs(args.project / "scale_configs.py")
    if scale not in configs:
        problems.append(f"SCALE_CONFIGS has no {scale!r} entry")
    else:
        got = configs[scale]
        for key, contract_key in (("batch_size", "batch_size"), ("learning_rate", "learning_rate"),
                                  ("train_num_samples", "train_num_samples"), ("warmup", "warmup"),
                                  ("model", "model"), ("beta2", "beta2")):
            if got.get(key) != recipe.get(contract_key):
                problems.append(
                    f"SCALE_CONFIGS[{scale!r}][{key!r}] is {got.get(key)!r}, contract "
                    f"frozen_recipe.{contract_key} says {recipe.get(contract_key)!r}. This dict IS "
                    f"the per-round budget; a mismatch means every round runs off-budget while "
                    f"task.toml and instruction.md still describe the declared one."
                )
        if got.get("train_num_samples") != contract["per_round_budget"]["samples_seen"]:
            problems.append("per_round_budget.samples_seen != SCALE_CONFIGS train_num_samples")

    # ---- symbols ------------------------------------------------------------------------------
    for rel, pattern, what in CLAIMS:
        path = args.project / rel
        if not path.is_file():
            problems.append(f"{rel} does not exist ({what})")
            continue
        if not re.search(pattern, path.read_text(), re.MULTILINE):
            problems.append(f"{rel}: /{pattern}/ not found -- {what}")

    # ---- the 38/40 fact, checked rather than trusted -------------------------------------------
    tasks = yaml.safe_load((args.project / "tasklist.yml").read_text())
    without = sorted(k for k, v in tasks.items() if "main_metric" not in (v or {}))
    fairness_without = [k for k in without if k.startswith("fairness/")]
    if len(tasks) - len(fairness_without) != contract["metric"]["protocol"]["n_datasets_asserted"]:
        problems.append(
            f"tasklist.yml has {len(tasks)} tasks and {len(fairness_without)} fairness tasks with no "
            f"main_metric ({fairness_without}); that gives "
            f"{len(tasks) - len(fairness_without)} scored, not "
            f"{contract['metric']['protocol']['n_datasets_asserted']}. The headline's dataset count "
            f"changed upstream and contract.yaml still claims the old one."
        )
    if sorted(fairness_without) != ["fairness/fairface", "fairness/utkface"]:
        problems.append(
            f"the fairness tasks lacking main_metric are {sorted(fairness_without)}, but "
            f"contract.yaml metric.headline.the_38_vs_40_trap names fairface and utkface "
            f"specifically. Which two drop out is a documented fact here; if it changed, the "
            f"documentation is now wrong about what the headline averages."
        )

    if problems:
        print("UPSTREAM CLAIM CHECK FAILED", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 0 if args.warn_only else 1
    print(f"upstream claims verified at {head[:12]}: {len(CLAIMS)} symbols, "
          f"SCALE_CONFIGS[{scale!r}] matches the frozen recipe, "
          f"{len(tasks)} tasks -> {contract['metric']['protocol']['n_datasets_asserted']} scored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
