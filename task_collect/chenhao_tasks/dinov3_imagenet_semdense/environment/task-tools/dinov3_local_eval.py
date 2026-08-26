#!/usr/bin/env python3
"""Agent-side screening probes (not the frozen verifier).

Runs the repository's k-NN probe (and optionally the fast dense proxy) on a
checkpoint for triage. GPU time counts against the attempt's budget —
record it in the ledger. Full four-probe scoring always comes from the
separate frozen verifier; on the determinism lane, compare these numbers
against the sealed screening anchors in the instruction.

Usage:
  python /task-tools/dinov3_local_eval.py \
      --checkpoint /app/output/attempts/<id>/outputs/.../teacher_checkpoint.pth \
      --model-config <config name> [--knn-only]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(os.environ.get("DINO_REPO_ROOT", "/app/project/dinov3"))
IMAGENET_DIR = os.environ.get("DINO_IMAGENET_DIR", "/datasets/imagenet-1k")
ADE20K_DIR = os.environ.get("DINO_ADE20K_DIR", "/datasets/ade20k")


def run(cmd: list[str], cwd: Path) -> int:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(cwd)).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config", required=True,
                        help="train config name whose model section defines the backbone")
    parser.add_argument("--knn-only", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        print(f"error: checkpoint not found: {checkpoint}", file=sys.stderr)
        return 1
    workdir = Path(tempfile.mkdtemp(prefix="dinov3-local-eval-"))

    # BUILD-TIME VERIFY: exact eval CLI flags re-checked against the pinned
    # commit when the image is first built (dinov3/eval/knn.py and the
    # 20-iteration ADE20k linear proxy config used for screening anchors).
    rc = run([
        sys.executable, "-m", "dinov3.eval.knn",
        "--config-file", args.model_config,
        "--pretrained-weights", str(checkpoint),
        "--output-dir", str(workdir / "knn"),
        f"--train-dataset=ImageNet:split=TRAIN:root={IMAGENET_DIR}:extra={IMAGENET_DIR}",
        f"--val-dataset=ImageNet:split=VAL:root={IMAGENET_DIR}:extra={IMAGENET_DIR}",
    ], REPO_ROOT)
    if rc != 0:
        print("error: knn probe failed", file=sys.stderr)
        return rc

    if not args.knn_only:
        rc = run([
            sys.executable, "-m", "dinov3.eval.segmentation.run",
            "--config-file",
            str(REPO_ROOT / "dinov3/eval/segmentation/config-ade20k-linear-training.yaml"),
            "--pretrained-weights", str(checkpoint),
            "--output-dir", str(workdir / "seg-proxy"),
            f"--dataset-root={ADE20K_DIR}",
            "--max-iter", "20",
        ], REPO_ROOT)
        if rc != 0:
            print("warning: dense proxy failed; knn result still valid",
                  file=sys.stderr)

    summary = {"workdir": str(workdir),
               "results": "see per-probe output dirs (results json/log)"}
    output = Path(args.output) if args.output else checkpoint.with_suffix(".localeval.json")
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
