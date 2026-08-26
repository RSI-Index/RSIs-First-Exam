#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path


root = Path("/logs/verifier")
metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
reward = float(metrics["aggregates"]["Average"])
if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
    raise RuntimeError(f"invalid official DataComp Average: {reward}")
result = {
    "reward": round(reward, 6),
    "policy_gate": 1,
    **metrics,
}
(root / "reward.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
(root / "reward.txt").write_text(f"{reward:.6f}\n", encoding="utf-8")
