#!/usr/bin/env python3
import json
import math
from pathlib import Path

m = json.loads(Path("/logs/verifier/metrics.json").read_text())
quality = ((1 - m["text_edit_distance"]) + (1 - m["formula_edit_distance"]) + m["table_teds"] + (1 - m["reading_order_edit_distance"])) / 4
tokens = float(m["average_visual_tokens"])
reward = 0.0 if tokens <= 0 else max(0.0, min(1.0, quality / math.sqrt(max(1.0, tokens / 250.0))))
result = {"reward": round(reward, 6), "policy_gate": 1, "metric": "ocr_quality_visual_token_pareto", "quality": quality, **m}
Path("/logs/verifier/reward.json").write_text(json.dumps(result, indent=2))
Path("/logs/verifier/reward.txt").write_text(f"{reward:.6f}\n")
