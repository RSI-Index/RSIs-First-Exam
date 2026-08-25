#!/usr/bin/env python3
import json
import math
from pathlib import Path

metrics = json.loads(Path("/logs/verifier/metrics.json").read_text())
accuracy = float(metrics["oolong_reward"])
if not math.isfinite(accuracy):
    reward = 0.0
else:
    reward = max(0.0, min(1.0, accuracy))
result = {
    "reward": round(reward, 6),
    "policy_gate": 1,
    "metric": "mean_raw_oolong_reward",
    **metrics,
}
Path("/logs/verifier/reward.json").write_text(json.dumps(result, indent=2))
Path("/logs/verifier/reward.txt").write_text(f"{reward:.6f}\n")
