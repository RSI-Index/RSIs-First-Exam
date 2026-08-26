#!/usr/bin/env python3
"""Composite semantic+dense reward for dinov3-imagenet-semdense.

r_knn = knn/B_knn   r_lin = linear/B_linear
r_seg = miou/B_miou r_dep = B_rmse/rmse          (lower RMSE is better)
G = geomean(r_knn, r_lin, r_seg, r_dep)

reward = 0                          if min(r) < FLOOR   (no-regression floor)
reward = clip((G - 1)/(G_REF - 1), 0, 1)   otherwise

PROVISIONAL ANCHORS: B_knn/B_linear are the pinned config's own header
numbers (82.2/83.3); B_miou/B_rmse are rough ESTIMATES (no published value
exists for this exact config). The task authors run the pristine config
once before release; the sealed run replaces every constant, and FLOOR /
G_REF are re-placed outside the measured 3-run probe-noise band (env
overrides DINO_B_* / DINO_FLOOR / DINO_G_REF are honored for that purpose).
"""
import json
import math
import os
from pathlib import Path

B_KNN = float(os.environ.get("DINO_B_KNN", "0.822"))
B_LINEAR = float(os.environ.get("DINO_B_LINEAR", "0.833"))
B_MIOU = float(os.environ.get("DINO_B_MIOU", "0.40"))
B_RMSE = float(os.environ.get("DINO_B_RMSE", "0.45"))
FLOOR = float(os.environ.get("DINO_FLOOR", "0.98"))
G_REF = float(os.environ.get("DINO_G_REF", "1.03"))

metrics = json.loads(Path("/logs/verifier/metrics.json").read_text())
if metrics.get("smoke"):
    raise SystemExit("smoke-mode metrics must not be scored")

ratios = {
    "r_knn": float(metrics["knn_top1"]) / B_KNN,
    "r_linear": float(metrics["linear_top1"]) / B_LINEAR,
    "r_seg": float(metrics["ade20k_miou"]) / B_MIOU,
    "r_depth": B_RMSE / float(metrics["nyu_rmse"]),
}
finite = all(math.isfinite(r) and r > 0 for r in ratios.values())
if not finite:
    reward, G = 0.0, float("nan")
else:
    G = math.prod(ratios.values()) ** 0.25
    if min(ratios.values()) < FLOOR or G <= 1.0:
        reward = 0.0
    else:
        reward = max(0.0, min(1.0, (G - 1.0) / (G_REF - 1.0)))
result = {
    "reward": round(reward, 6),
    "policy_gate": 1,
    "metric": "composite_G",
    "G": None if not finite else round(G, 6),
    **{k: round(v, 6) for k, v in ratios.items() if math.isfinite(v)},
    "floor": FLOOR,
    "G_ref": G_REF,
    "baselines": {"knn": B_KNN, "linear": B_LINEAR, "miou": B_MIOU, "rmse": B_RMSE},
    "anchors_status": "provisional_values_pending_author_reproduction",
    **metrics,
}
Path("/logs/verifier/reward.json").write_text(json.dumps(result, indent=2) + "\n")
Path("/logs/verifier/reward.txt").write_text(f"{reward:.6f}\n")
print(json.dumps(result, indent=2))
