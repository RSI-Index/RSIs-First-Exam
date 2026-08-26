#!/usr/bin/env python3
import json
import math
from pathlib import Path

metrics = json.loads(Path("/logs/verifier/metrics.json").read_text())
score = float(metrics["video_point_score"])
reward = 0.0 if not math.isfinite(score) else min(max(score, 0.0), 1.0)
result = {
    "reward": round(reward, 6),
    "policy_gate": 1,
    "metric": "video_point_score",
    **metrics,
}
Path("/logs/verifier/reward.json").write_text(json.dumps(result, indent=2) + "\n")
Path("/logs/verifier/reward.txt").write_text(f"{reward:.6f}\n")
