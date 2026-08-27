#!/usr/bin/env python3
"""Agent-side screening evaluation (not the frozen verifier).

Runs the pinned datacomp evaluation over a subset of the 38-task suite for
cheap triage. Full-suite numbers for selection claims come from running
without --fast; the scored result always comes from the separate frozen
verifier. GPU time spent here counts against the attempt's budget — record
it in the ledger.

Usage:
  python /task-tools/clip_local_eval.py \
      --model-config ViT-B-32 \
      --checkpoint /app/output/attempts/<id>/outputs/.../epoch_latest.pt \
      [--fast]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

DATACOMP_DIR = Path("/app/project/datacomp")
OPEN_CLIP_SRC = Path("/app/project/open_clip/src")
EVAL_DIR = os.environ.get("DATACOMP_EVAL_DIR", "/datasets/datacomp-eval")

# Screening subset: ImageNet plus a spread of distribution shift, retrieval,
# and fine-grained tasks. Chosen once at task build; keys follow tasklist.yml.
FAST_TASKS = [
    "imagenet1k",
    "imagenetv2",
    "vtab/cifar100",
    "vtab/flowers",
    "retrieval/mscoco_2014_5k_test_image_text_retrieval",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True,
                        help="open_clip model config name (resolvable from the workspace)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fast", action="store_true",
                        help="evaluate only the screening subset")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default=None,
                        help="where to write the metrics json (default: alongside checkpoint)")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        print(f"error: checkpoint not found: {checkpoint}", file=sys.stderr)
        return 1

    workdir = Path(tempfile.mkdtemp(prefix="clip-local-eval-"))
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{OPEN_CLIP_SRC}:{env.get('PYTHONPATH', '')}"

    if args.fast:
        # Restrict the pinned evaluator to the screening subset by giving it a
        # trimmed tasklist in an overlay copy of the datacomp tree.
        import yaml
        overlay = workdir / "datacomp"
        overlay.symlink_to(DATACOMP_DIR)
        tasks = yaml.safe_load((DATACOMP_DIR / "tasklist.yml").read_text())
        subset = {k: v for k, v in tasks.items() if k in FAST_TASKS}
        missing = [k for k in FAST_TASKS if k not in tasks]
        if missing:
            print(f"warning: screening keys missing from tasklist.yml: {missing}",
                  file=sys.stderr)
        overlay_dir = workdir / "datacomp_fast"
        overlay_dir.mkdir()
        for entry in DATACOMP_DIR.iterdir():
            (overlay_dir / entry.name).symlink_to(entry)
        (overlay_dir / "tasklist.yml").unlink()
        (overlay_dir / "tasklist.yml").write_text(yaml.safe_dump(subset))
        eval_script = overlay_dir / "evaluate.py"
    else:
        eval_script = DATACOMP_DIR / "evaluate.py"

    out_dir = workdir / "results"
    cmd = [
        sys.executable, str(eval_script),
        "--train_output_dir", str(out_dir),
        "--data_dir", EVAL_DIR,
        "--use_model", f"{args.model_config} {checkpoint}",
        "--batch_size", str(args.batch_size),
        "--skip_hf", "--skip_db", "--skip_notification",
    ]
    print("+", " ".join(cmd))
    result = subprocess.run(cmd, env=env, cwd=str(eval_script.parent))
    if result.returncode != 0:
        print("error: evaluation failed", file=sys.stderr)
        return result.returncode

    results_file = out_dir / "eval_results.jsonl"
    metrics = {}
    for line in results_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        metrics[row["key"]] = row["metrics"].get("main_metric")
    scored = [v for v in metrics.values() if v is not None]
    summary = {
        "subset": "fast" if args.fast else "full_38",
        "num_tasks": len(scored),
        "average_main_metric": sum(scored) / len(scored) if scored else None,
        "imagenet1k": metrics.get("imagenet1k"),
        "per_task": metrics,
    }
    output = Path(args.output) if args.output else checkpoint.with_suffix(".localeval.json")
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_task"}, indent=2))
    print(f"full metrics: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
