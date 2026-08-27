#!/usr/bin/env python3
"""Verify the upstream code claims in contract.yaml, INSIDE the built image.

Why this file exists. The package holds no copy of VLM-R1 -- it is configuration only, and the
repository arrives as a pinned clone at image-build time. But contract.yaml makes specific,
load-bearing claims ABOUT that code: three named mechanism seams (the jsonl loader, the
per-group reward mean/std, the repeat sampler) and two frozen reward functions. Those claims are
why the task is expressible in the repository's own terms rather than being a wrapper we
invented. With no local tree, nothing outside the image can check them, and a line-number
citation is exactly the kind of thing that rots.

So the check runs where the code is: both Dockerfiles invoke it at build time and the image does
not build if a citation has gone stale. Line numbers are treated as HINTS -- the anchor is the
symbol or expression, searched for across the file, and the check reports the true line. A
citation that has merely shifted is a warning; a citation whose content is GONE is a failure.

    python tests/upstream_check.py --project /opt/project --contract /tests/contract.yaml
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml

SRC = "src/open-r1-multimodal/src/open_r1"

# (path relative to repo root, regex that must match, what the contract claims it is)
#
# PATTERNS ARE ANCHORED ON PURPOSE. The first version used bare `def\s+iou_reward`, and a
# fixture test renaming the function to `iou_reward_v2` still MATCHED -- a prefix pattern
# cannot detect a rename, so the check passed on precisely the drift it exists to catch. Every
# symbol pattern below terminates in `\s*\(` or `\b`.
CLAIMS = [
    (f"{SRC}/grpo_jsonl.py", r"Dataset\.from_list\s*\(",
     "the jsonl -> Dataset seam the curriculum is injected through"),
    (f"{SRC}/trainer/grpo_trainer.py", r"\.std\(",
     "per-group reward std; groups with std=0 contribute zero advantage"),
    (f"{SRC}/trainer/grpo_trainer.py", r"class\s+RepeatRandomSampler\b",
     "the sampler that turns one prompt into num_generations completions"),
    (f"{SRC}/vlm_modules/qwen_module.py", r"def\s+iou_reward\s*\(",
     "frozen reward function iou_reward"),
    (f"{SRC}/vlm_modules/qwen_module.py", r"def\s+format_reward_rec\s*\(",
     "frozen reward function format_reward_rec"),
    # The reward-integrity hazard the environment image is built around: an OpenAI client
    # constructed at import, and an llm reward path that degrades to string match on exception.
    (f"{SRC}/grpo_jsonl.py", r"OpenAI\(",
     "OpenAI client instantiated at import -- why the agent image omits `openai`"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, default=Path("/opt/project"))
    ap.add_argument("--contract", type=Path, default=Path("/tests/contract.yaml"))
    ap.add_argument("--warn-only", action="store_true",
                    help="report without failing; for inspecting a newer commit deliberately")
    args = ap.parse_args()

    problems: list[str] = []
    warnings: list[str] = []

    contract = yaml.safe_load(args.contract.read_text())
    pinned = contract["upstream"]["commit"]

    # 1. The tree is the pinned commit. A silently newer checkout invalidates every citation.
    try:
        head = subprocess.run(["git", "-C", str(args.project), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        if head != pinned:
            problems.append(f"/opt/project is at {head}, contract pins {pinned}")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        problems.append(f"cannot read the project's git HEAD ({exc}); the clone must be a real "
                        f"repository -- the reference package shipped a tree with no .git and "
                        f"its own commit assertion could never pass")

    # 2. Every cited symbol still exists, and report where it actually is now.
    for rel, pattern, what in CLAIMS:
        path = args.project / rel
        if not path.is_file():
            problems.append(f"MISSING FILE {rel} -- contract cites it for: {what}")
            continue
        text = path.read_text(errors="replace")
        match = re.search(pattern, text)
        if not match:
            problems.append(f"{rel}: /{pattern}/ not found -- contract claims it holds: {what}")
        else:
            line = text[:match.start()].count("\n") + 1
            print(f"  ok  {rel}:{line}  {what}")

    # 3. The line hints in contract.yaml improver.mechanism_seams_in_upstream. Drift here is a
    #    warning: the citation is still true, it just points at the wrong line, and a reader
    #    following it would land somewhere confusing.
    for seam in contract.get("improver", {}).get("mechanism_seams_in_upstream", []) or []:
        m = re.match(r"^(\S+?):(\d+)(?:-(\d+))?", str(seam))
        if not m:
            continue
        rel, lo = m.group(1), int(m.group(2))
        path = args.project / rel
        if not path.is_file():
            problems.append(f"seam citation points at a missing file: {seam}")
            continue
        n = len(path.read_text(errors="replace").splitlines())
        if lo > n:
            problems.append(f"seam citation {seam} is past end of file ({n} lines)")
        elif lo > n * 0.995:
            warnings.append(f"seam citation {seam} sits at the very end of a {n}-line file")

    for w in warnings:
        print(f"  warn  {w}")
    if problems:
        print("UPSTREAM CLAIM CHECK FAILED")
        for p in problems:
            print(f"  - {p}")
        # Under --warn-only the exit code is 0, but do NOT print "verified" -- a report that
        # ends in a success line after listing failures is how a stale citation gets waved
        # through by whoever is skimming the build log.
        return 0 if args.warn_only else 1
    print(f"upstream claims verified against {pinned}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
