#!/usr/bin/env python3
import json
import math
from pathlib import Path

metrics = json.loads(Path("/logs/verifier/metrics.json").read_text())
score = float(metrics["mean_reward"])
reward = score if math.isfinite(score) and 0.0 <= score <= 1.0 else 0.0
result = {"reward": round(reward, 6), "policy_gate": 1, **metrics}
Path("/logs/verifier/reward.json").write_text(json.dumps(result, indent=2) + "\n")
Path("/logs/verifier/reward.txt").write_text(f"{reward:.6f}\n")

