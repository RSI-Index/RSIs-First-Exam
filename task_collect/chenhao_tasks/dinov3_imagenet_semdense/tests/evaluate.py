#!/usr/bin/env python3
"""Frozen four-probe evaluator for dinov3-imagenet-semdense.

Runs after the policy gate. Contract (mirrors gated_deltanet):

- The backbone ARCHITECTURE is loaded from the candidate workspace
  (`/app/project/dinov3` first on PYTHONPATH), so new named configs the
  agent added are honored, and it must satisfy the interface contract:
  constructible from the recorded `model_config` name and exposing the
  repository's `get_intermediate_layers()` semantics — the dense probes
  consume exactly that API.
- The probe PROTOCOL is frozen: the pinned driver files under
  `/opt/project/dinov3/dinov3/eval/` are executed with their default
  (pinned) probe recipes, fixed seeds, and the staged offline data. The
  same recipe scored the sealed baseline.

Probes and metric extraction:
  1. IN1k k-NN top-1            -> knn_top1
  2. IN1k linear probe top-1    -> linear_top1
  3. ADE20k linear seg mIoU     -> ade20k_miou
  4. NYUv2 linear depth RMSE    -> nyu_rmse (lower is better)

Writes /app/output/verifier-metrics.json with raw metrics + parameters.
score.py maps them to the composite reward.

BUILD-TIME VERIFY: the exact CLI of the pinned eval drivers
(dinov3/eval/{knn,linear}.py, eval/segmentation/run.py, eval/depth/run.py)
and their results-file names are re-checked against the pinned commit when
the tests image is first built; the RESULT_PATTERNS below are updated then.
Env knobs (task infrastructure): DINO_VERIFIER_SMOKE=1 restricts to
construction + parameter check + k-NN on a subset.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

OUTPUT_ROOT = Path("/app/output")
CANDIDATE_ROOT = Path("/app/project/dinov3")
PINNED_ROOT = Path("/opt/project/dinov3")
IMAGENET_DIR = os.environ.get("DINO_IMAGENET_DIR", "/datasets/imagenet-1k")
ADE20K_DIR = os.environ.get("DINO_ADE20K_DIR", "/datasets/ade20k")
NYU_DIR = os.environ.get("DINO_NYU_DIR", "/datasets/nyu-depth-v2")
PARAM_BUDGET = int(os.environ.get("DINO_PARAM_BUDGET", "307000000"))
SMOKE = os.environ.get("DINO_VERIFIER_SMOKE", "0") == "1"


def fail(reason: str) -> int:
    print(f"verifier failure: {reason}", file=sys.stderr)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier") / "evaluate-failure.json").write_text(
        json.dumps({"reason": reason}, indent=2) + "\n", encoding="utf-8")
    return 1


def run_probe(name: str, module_args: list[str], workdir: Path,
              env: dict) -> Path:
    out_dir = workdir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable] + module_args + ["--output-dir", str(out_dir)]
    print("+", " ".join(cmd))
    proc = subprocess.run(cmd, env=env, cwd=str(PINNED_ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"probe {name} exited {proc.returncode}")
    return out_dir


# metric name -> regex applied to every results/log file in the probe's
# output dir, last match wins. Updated at build-time verification.
RESULT_PATTERNS = {
    "knn_top1": r'"?top-?1"?[=:\s]+([0-9.]+)',
    "linear_top1": r'"?top-?1"?[=:\s]+([0-9.]+)',
    "ade20k_miou": r'"?m?IoU"?[=:\s]+([0-9.]+)',
    "nyu_rmse": r'"?rmse"?[=:\s]+([0-9.]+)',
}


def extract_metric(out_dir: Path, metric: str) -> float:
    pattern = re.compile(RESULT_PATTERNS[metric], re.IGNORECASE)
    value = None
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".txt", ".log", ""}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in pattern.finditer(text):
            value = float(match.group(1))
    if value is None:
        raise RuntimeError(f"could not extract {metric} from {out_dir}")
    return value / 100.0 if metric != "nyu_rmse" and value > 1.5 else value


def main() -> int:
    provenance_path = OUTPUT_ROOT / "provenance.json"
    if not provenance_path.exists():
        return fail("missing provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    model_config = provenance.get("model_config")
    if not model_config or not isinstance(model_config, str):
        return fail("provenance.model_config missing")

    backbone_ckpt = OUTPUT_ROOT / "submission" / "backbone.pt"
    model_state = OUTPUT_ROOT / "model_state.pt"
    for path in (backbone_ckpt, model_state):
        if not path.exists():
            return fail(f"missing staged artifact: {path}")

    # --- interface contract + parameter budget ---
    sys.path.insert(0, str(CANDIDATE_ROOT))
    import torch

    state = torch.load(model_state, map_location="cpu", weights_only=True)
    parameters = sum(int(v.numel()) for v in state.values())
    if parameters > PARAM_BUDGET:
        return fail(f"parameter budget exceeded: {parameters} > {PARAM_BUDGET}")
    try:
        # BUILD-TIME VERIFY: exact builder API at the pinned commit
        # (dinov3.models.build_model_from_cfg or equivalent config loader).
        from dinov3.configs import load_and_merge_config  # type: ignore
        from dinov3.models import build_model_from_cfg  # type: ignore
        cfg = load_and_merge_config(model_config)
        backbone = build_model_from_cfg(cfg, only_teacher=True)
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        if unexpected:
            return fail(f"state dict has unexpected keys: {unexpected[:5]}...")
        if not hasattr(backbone, "get_intermediate_layers"):
            return fail("interface contract: backbone lacks get_intermediate_layers()")
        backbone.eval()
        with torch.no_grad():
            feats = backbone.get_intermediate_layers(
                torch.zeros(1, 3, 224, 224), n=1)
        del backbone
    except Exception as exc:  # noqa: BLE001 - report, do not crash the gate
        return fail(f"could not construct/validate model_config={model_config!r} "
                    f"from candidate workspace: {exc}")

    # --- frozen probes: pinned drivers, candidate package on PYTHONPATH ---
    workdir = Path(tempfile.mkdtemp(prefix="dinov3-verifier-"))
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{CANDIDATE_ROOT}:{env.get('PYTHONPATH', '')}"
    ckpt = str(backbone_ckpt)
    in1k_train = f"ImageNet:split=TRAIN:root={IMAGENET_DIR}:extra={IMAGENET_DIR}"
    in1k_val = f"ImageNet:split=VAL:root={IMAGENET_DIR}:extra={IMAGENET_DIR}"

    metrics: dict = {"parameters": parameters, "parameter_budget": PARAM_BUDGET,
                     "model_config": model_config}
    try:
        out = run_probe("knn", [
            str(PINNED_ROOT / "dinov3/eval/knn.py"),
            "--config-file", model_config,
            "--pretrained-weights", ckpt,
            f"--train-dataset={in1k_train}", f"--val-dataset={in1k_val}",
        ], workdir, env)
        metrics["knn_top1"] = extract_metric(out, "knn_top1")

        if not SMOKE:
            out = run_probe("linear", [
                str(PINNED_ROOT / "dinov3/eval/linear.py"),
                "--config-file", model_config,
                "--pretrained-weights", ckpt,
                f"--train-dataset={in1k_train}", f"--val-dataset={in1k_val}",
            ], workdir, env)
            metrics["linear_top1"] = extract_metric(out, "linear_top1")

            out = run_probe("segmentation", [
                str(PINNED_ROOT / "dinov3/eval/segmentation/run.py"),
                "--config-file",
                str(PINNED_ROOT / "dinov3/eval/segmentation/config-ade20k-linear-training.yaml"),
                "--pretrained-weights", ckpt,
                f"--dataset-root={ADE20K_DIR}",
            ], workdir, env)
            metrics["ade20k_miou"] = extract_metric(out, "ade20k_miou")

            out = run_probe("depth", [
                str(PINNED_ROOT / "dinov3/eval/depth/run.py"),
                "--config-file",
                str(PINNED_ROOT / "dinov3/eval/depth/config-nyu.yaml"),
                "--pretrained-weights", ckpt,
                f"--dataset-root={NYU_DIR}",
            ], workdir, env)
            metrics["nyu_rmse"] = extract_metric(out, "nyu_rmse")
    except RuntimeError as exc:
        return fail(str(exc))

    metrics["smoke"] = SMOKE
    (OUTPUT_ROOT / "verifier-metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
