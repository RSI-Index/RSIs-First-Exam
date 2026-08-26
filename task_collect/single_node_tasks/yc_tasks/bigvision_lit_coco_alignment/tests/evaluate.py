#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path


RETRIEVAL_KEYS = {
    "i2t_r1": "z/retr/coco_img2txt_recall@1",
    "i2t_r5": "z/retr/coco_img2txt_recall@5",
    "i2t_r10": "z/retr/coco_img2txt_recall@10",
    "t2i_r1": "z/retr/coco_txt2img_recall@1",
    "t2i_r5": "z/retr/coco_txt2img_recall@5",
    "t2i_r10": "z/retr/coco_txt2img_recall@10",
}
IMAGENET_KEY = "z/0shot/imagenet2012_accuracy"


def parse_metrics(path: Path) -> dict:
    merged = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid Big Vision metrics line {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"metrics line {line_number} is not an object")
        merged.update(row)
    missing = [key for key in [*RETRIEVAL_KEYS.values(), IMAGENET_KEY] if key not in merged]
    if missing:
        raise ValueError(f"clean evaluator is missing metrics: {missing}")
    result = {name: float(merged[key]) * 100.0 for name, key in RETRIEVAL_KEYS.items()}
    result["imagenet_zero_shot_accuracy"] = float(merged[IMAGENET_KEY]) * 100.0
    if not all(math.isfinite(value) and 0.0 <= value <= 100.0 for value in result.values()):
        raise ValueError("evaluation metrics must be finite percentages in [0,100]")
    return {"status": "pass", **result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/app/output/artifacts/model.npz"))
    parser.add_argument("--workdir", type=Path, default=Path("/logs/verifier/eval"))
    parser.add_argument("--output", type=Path, default=Path("/logs/verifier/metrics.json"))
    args = parser.parse_args()
    if not args.model.is_file() or args.model.is_symlink():
        raise ValueError("missing regular model artifact")
    if args.workdir.exists():
        raise ValueError(f"fresh verifier workdir required: {args.workdir}")
    args.workdir.mkdir(parents=True)
    env = {
        **os.environ,
        "BIGVISION_FINAL_EVAL": "1",
        "BIGVISION_EVAL_CHECKPOINT": str(args.model),
        "BIGVISION_CANDIDATE_CONFIG": "/app/project/candidate/config.toml",
    }
    subprocess.run([
        "python3", "/opt/big_vision/big_vision/tools/eval_only.py",
        "--config=/task-tools/lit_coco_autoresearch.py",
        f"--workdir={args.workdir}",
    ], check=True, env=env)
    metrics = parse_metrics(args.workdir / "big_vision_metrics.txt")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()

