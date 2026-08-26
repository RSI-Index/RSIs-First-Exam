#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path


metrics = json.loads(Path("/logs/verifier/metrics.json").read_text(encoding="utf-8"))
candidate = float(metrics["bytes_per_token"])
reference = float(metrics["reference_bytes_per_token"])
reward = 0.0
if math.isfinite(candidate) and math.isfinite(reference) and candidate > 0 and reference > 0:
    reward = candidate / (candidate + reference)
result = {
    "reward": round(reward, 6),
    "policy_gate": 1,
    "metric": "bytes_per_token",
    **metrics,
}
Path("/logs/verifier/reward.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
Path("/logs/verifier/reward.txt").write_text(f"{reward:.6f}\n", encoding="utf-8")
