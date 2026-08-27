#!/usr/bin/env python3
"""Frozen evaluator for the openclip-datacomp task family.

Runs after the policy gate. Mirrors the gated_deltanet contract: the model
ARCHITECTURE is loaded from the candidate workspace (so new named model
configs the agent added under open_clip are honored), while the evaluation
PROTOCOL (datacomp evaluate.py, tasklist, eval data) is the pinned pristine
copy under /opt/project — never the agent-editable one.

Steps:
  1. read provenance.json -> model_config; check staged submission artifacts
  2. rebuild the model from the candidate workspace code + tensor-only
     model_state.pt; enforce the whole-model parameter budget
  3. run the pinned datacomp evaluate.py over the staged offline eval suite
  4. write /app/output/verifier-metrics.json {avg_38, imagenet1k, parameters}

Env knobs (task infrastructure, not agent-visible):
  DATACOMP_EVAL_DIR   staged eval datasets (default /datasets/datacomp-eval)
  CLIP_PARAM_BUDGET   whole-model parameter cap (default 189000000)
  CLIP_EVAL_BATCH     eval batch size (default 64)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

OUTPUT_ROOT = Path("/app/output")
CANDIDATE_OPEN_CLIP_SRC = Path("/app/project/open_clip/src")
PINNED_DATACOMP = Path("/opt/project/datacomp")
EVAL_DIR = os.environ.get("DATACOMP_EVAL_DIR", "/datasets/datacomp-eval")
PARAM_BUDGET = int(os.environ.get("CLIP_PARAM_BUDGET", "189000000"))
BATCH_SIZE = int(os.environ.get("CLIP_EVAL_BATCH", "64"))
SMOKE = os.environ.get("CLIP_VERIFIER_SMOKE", "0") == "1"


def fail(reason: str) -> int:
    print(f"verifier failure: {reason}", file=sys.stderr)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier") / "evaluate-failure.json").write_text(
        json.dumps({"reason": reason}, indent=2) + "\n", encoding="utf-8")
    return 1


def main() -> int:
    provenance_path = OUTPUT_ROOT / "provenance.json"
    if not provenance_path.exists():
        return fail("missing provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    model_config = provenance.get("model_config")
    if not model_config or not isinstance(model_config, str):
        return fail("provenance.model_config missing")

    checkpoint = OUTPUT_ROOT / "submission" / "checkpoint.pt"
    model_state = OUTPUT_ROOT / "model_state.pt"
    for path in (checkpoint, model_state):
        if not path.exists():
            return fail(f"missing staged artifact: {path}")

    # Candidate workspace provides the architecture definition only.
    sys.path.insert(0, str(CANDIDATE_OPEN_CLIP_SRC))
    import torch
    import open_clip

    try:
        model = open_clip.create_model(model_config, pretrained=None)
    except Exception as exc:  # noqa: BLE001 - report, do not crash the gate
        return fail(f"could not construct model_config={model_config!r} "
                    f"from candidate workspace: {exc}")

    state = torch.load(model_state, map_location="cpu", weights_only=True)
    parameters = sum(int(v.numel()) for v in state.values())
    if parameters > PARAM_BUDGET:
        return fail(f"parameter budget exceeded: {parameters} > {PARAM_BUDGET}")
    try:
        missing, unexpected = model.load_state_dict(state, strict=True)
    except Exception as exc:  # noqa: BLE001
        return fail(f"state dict does not match architecture: {exc}")
    del model  # construction + budget check done; eval reloads via datacomp

    # Frozen protocol from the pristine tree; candidate code only supplies
    # the open_clip package on PYTHONPATH so custom configs resolve.
    workdir = Path(tempfile.mkdtemp(prefix="clip-verifier-"))
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{CANDIDATE_OPEN_CLIP_SRC}:{env.get('PYTHONPATH', '')}"
    env["HF_HUB_OFFLINE"] = "1"
    cmd = [
        sys.executable, str(PINNED_DATACOMP / "evaluate.py"),
        "--train_output_dir", str(workdir / "results"),
        "--data_dir", EVAL_DIR,
        "--use_model", f"{model_config} {checkpoint}",
        "--batch_size", str(BATCH_SIZE),
        "--skip_hf", "--skip_db", "--skip_notification",
    ]
    print("+", " ".join(cmd))
    proc = subprocess.run(cmd, env=env, cwd=str(PINNED_DATACOMP))
    if proc.returncode != 0:
        return fail(f"pinned datacomp evaluation exited {proc.returncode}")

    results_file = workdir / "results" / "eval_results.jsonl"
    if not results_file.exists():
        return fail("evaluation produced no eval_results.jsonl")

    import yaml
    tasks = yaml.safe_load((PINNED_DATACOMP / "tasklist.yml").read_text())
    per_task: dict[str, float | None] = {}
    for line in results_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        per_task[row["key"]] = row["metrics"].get("main_metric")

    missing_tasks = [key for key in tasks if key not in per_task]
    if missing_tasks and not SMOKE:
        return fail(f"evaluation incomplete; missing tasks: {missing_tasks}")
    scored = [per_task[key] for key in tasks if per_task[key] is not None]
    if not scored:
        return fail("no scored tasks")

    metrics = {
        "smoke": SMOKE,
        "avg_38": sum(scored) / len(scored),
        "imagenet1k": per_task.get("imagenet1k"),
        "num_tasks_scored": len(scored),
        "parameters": parameters,
        "parameter_budget": PARAM_BUDGET,
        "model_config": model_config,
        "per_task": per_task,
    }
    (OUTPUT_ROOT / "verifier-metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_task"},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
