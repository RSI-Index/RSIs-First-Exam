#!/usr/bin/env bash
set -uo pipefail
mkdir -p /logs/verifier

zero() {
  python3 - "$1" <<'PY'
import json, sys
from pathlib import Path
result = {"reward": 0.0, "policy_gate": 0, "reason": sys.argv[1]}
Path("/logs/verifier/reward.json").write_text(json.dumps(result, indent=2) + "\n")
Path("/logs/verifier/reward.txt").write_text("0.0\n")
PY
}

if ! /opt/policy-venv/bin/python /tests/policy_check.py \
    --policy /tests/policy.yaml \
    --clean /opt/project \
    --candidate /app/project \
    --output-dir /app/output; then
  zero reward_integrity_gate
  exit 0
fi
if ! python3 /tests/evaluate.py; then
  zero encoding_evaluation_failed
  exit 0
fi
cp /app/output/verifier-metrics.json /logs/verifier/metrics.json
python3 /tests/score.py
