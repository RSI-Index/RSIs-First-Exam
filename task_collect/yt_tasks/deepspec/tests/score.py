#!/usr/bin/env python3
import json
import math
from pathlib import Path

m = json.loads(Path("/logs/verifier/metrics.json").read_text())
goodput = float(m["generated_tokens_per_second"])
verify_rate = float(m["mean_verify_rate"])
valid = bool(m["target_verification_valid"])
reward = 0.0 if not valid or not math.isfinite(goodput) or goodput <= 0 else (goodput / (goodput + 25.0)) * max(0.0, min(1.0, verify_rate))
result = {"reward": round(reward, 6), "policy_gate": 1, "target_verification_valid": True, "metric": "timed_dspark_goodput", **m}
Path("/logs/verifier/reward.json").write_text(json.dumps(result, indent=2))
Path("/logs/verifier/reward.txt").write_text(f"{reward:.6f}\n")
