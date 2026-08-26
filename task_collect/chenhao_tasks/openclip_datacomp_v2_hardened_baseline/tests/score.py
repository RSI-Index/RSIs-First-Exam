#!/usr/bin/env python3
"""Reward mapping for openclip-datacomp-medium-v2-hardened.

reward = 0                                      if avg_38 <= S_BASE
reward = clip((avg_38 - S_BASE) / (S_REF - S_BASE), 0, 1)  otherwise

PROVISIONAL ANCHORS: S_BASE below is a literature-derived ESTIMATE of the
hardened-baseline sweep maximum (SigLIP-loss + patch-dropout composition
over the DataComp paper medium baseline 0.328). The task authors execute
the full disclosed sweep on the sealed snapshot before release; the
measured maximum replaces this constant (env overrides CLIP_S_BASE /
CLIP_S_REF are honored for that purpose). S_REF is the same frontier
anchor as v1.
"""
import json
import math
import os
from pathlib import Path

S_BASE = float(os.environ.get("CLIP_S_BASE", "0.345"))
S_REF = float(os.environ.get("CLIP_S_REF", "0.390"))

metrics = json.loads(Path("/logs/verifier/metrics.json").read_text())
score = float(metrics["avg_38"])
if not math.isfinite(score) or score <= S_BASE:
    reward = 0.0
else:
    reward = max(0.0, min(1.0, (score - S_BASE) / (S_REF - S_BASE)))
result = {
    "reward": round(reward, 6),
    "policy_gate": 1,
    "metric": "avg_38",
    "S_base": S_BASE,
    "S_ref": S_REF,
    "anchors_status": "provisional_estimates_pending_author_sweep",
    **{k: v for k, v in metrics.items() if k != "per_task"},
}
Path("/logs/verifier/reward.json").write_text(json.dumps(result, indent=2) + "\n")
Path("/logs/verifier/reward.txt").write_text(f"{reward:.6f}\n")
print(json.dumps(result, indent=2))
