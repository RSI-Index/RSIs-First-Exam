#!/usr/bin/env python3
"""Frozen reward mapping for the architecture task.

reward = 1 / (1 + ln(P)) with P the geometric mean of the release-protocol
val_ppl@1x and val_ppl@2x. Monotone in both perplexities, so it preserves
architecture ranking while encoding the release's own two-length protocol.
"""
import json
import math
from pathlib import Path

metrics = json.loads(Path("/logs/verifier/metrics.json").read_text())
ppl_1x = float(metrics["val_ppl@1x"])
ppl_2x = float(metrics["val_ppl@2x"])
geomean = math.sqrt(ppl_1x * ppl_2x)
valid = math.isfinite(geomean) and geomean >= 1.0
reward = 0.0 if not valid else 1.0 / (1.0 + math.log(geomean))
result = {
    "reward": round(reward, 6),
    "policy_gate": 1,
    "metric": "val_ppl_geomean",
    "val_ppl_geomean_scored": geomean,
    **metrics,
}
Path("/logs/verifier/reward.json").write_text(json.dumps(result, indent=2) + "\n")
Path("/logs/verifier/reward.txt").write_text(f"{reward:.6f}\n")
