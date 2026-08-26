#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import yaml


def partition(tasklist: dict, count: int) -> list[dict]:
    keys = list(tasklist)
    source_order = {key: index for index, key in enumerate(keys)}
    assignments = [dict() for _ in range(count)]
    loads = [0] * count
    for key in sorted(keys, key=lambda item: (-int(tasklist[item].get("time", 0)), source_order[item])):
        worker = min(range(count), key=lambda index: (loads[index], index))
        assignments[worker][key] = tasklist[key]
        loads[worker] += int(tasklist[key].get("time", 0))
    return assignments


def evaluate(checkpoint: Path, data_root: Path, log_root: Path, workers: int = 4) -> dict:
    source = Path("/opt/datacomp")
    tasklist = yaml.safe_load((source / "tasklist.yml").read_text())
    if len(tasklist) != 40:
        raise RuntimeError(f"expected 40 attempted task configurations, found {len(tasklist)}")
    run_root = log_root / "workers"
    if run_root.exists():
        raise RuntimeError("verifier worker root must be fresh")
    run_root.mkdir(parents=True)
    processes = []
    outputs = []
    for index, subset in enumerate(partition(tasklist, workers)):
        worker = run_root / f"worker_{index:02d}"
        private_source = worker / "source"
        output = worker / "output"
        shutil.copytree(source, private_source, ignore=shutil.ignore_patterns(".git", "__pycache__"))
        (private_source / "tasklist.yml").write_text(yaml.safe_dump(subset, sort_keys=False))
        output.mkdir()
        stdout = (worker / "stdout.log").open("w")
        stderr = (worker / "stderr.log").open("w")
        command = [
            "python", "evaluate.py", "--train_output_dir", str(output), "--output_dir", str(output),
            "--data_dir", str(data_root), "--batch_size", "64",
            "--use_model", f"ViT-B-32 {checkpoint}",
        ]
        process = subprocess.Popen(command, cwd=private_source, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(index)}, stdout=stdout, stderr=stderr)
        processes.append((process, stdout, stderr))
        outputs.append(output / "eval_results.jsonl")
    statuses = []
    for process, stdout, stderr in processes:
        statuses.append(process.wait())
        stdout.close()
        stderr.close()
    if any(status != 0 for status in statuses):
        raise RuntimeError(f"one or more evaluator workers failed: {statuses}")
    rows = []
    for output in outputs:
        rows.extend(json.loads(line) for line in output.read_text().splitlines() if line.strip())
    if len(rows) != 40:
        raise RuntimeError(f"expected 40 evaluator rows, found {len(rows)}")
    datasets = [str(row.get("dataset")) for row in rows]
    if len(set(datasets)) != 40:
        raise RuntimeError("evaluator rows contain duplicate datasets")
    components = {}
    for row in rows:
        value = row.get("metrics", {}).get("main_metric")
        if value is None:
            continue
        value = float(value)
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite main metric for {row.get('dataset')}")
        components[str(row["dataset"])] = value
    if len(components) != 38:
        raise RuntimeError(f"expected 38 non-null main metrics, found {len(components)}")
    metrics = {"status": "pass", "component_metrics": dict(sorted(components.items())), "attempted_task_count": 40, "component_count": 38, "worker_returncodes": statuses}
    (log_root / "metrics.json").write_text(json.dumps(metrics, sort_keys=True, indent=2) + "\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("/logs/verifier/sanitized/epoch_5.pt"))
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("DATACOMP_EVAL_ROOT", "/datasets/evaluation")))
    parser.add_argument("--log-root", type=Path, default=Path("/logs/verifier"))
    args = parser.parse_args()
    evaluate(args.checkpoint, args.data_root, args.log_root)


if __name__ == "__main__":
    main()
