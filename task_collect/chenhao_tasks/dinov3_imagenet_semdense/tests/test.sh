#!/usr/bin/env bash
set -uo pipefail
mkdir -p /logs/verifier

zero() {
  python3 - "$1" <<'PY'
import json, sys
result = {"reward": 0.0, "policy_gate": 0.0}
failure = {"reason": sys.argv[1]}
open("/logs/verifier/reward.json", "w").write(json.dumps(result, indent=2) + "\n")
open("/logs/verifier/failure.json", "w").write(json.dumps(failure, indent=2) + "\n")
open("/logs/verifier/reward.txt", "w").write("0.0\n")
PY
}

if ! /opt/policy-venv/bin/python /tests/policy_check.py \
    --policy /tests/policy.yaml --clean /opt/project \
    --candidate /app/project --output-dir /app/output; then
  zero reward_integrity_gate
  exit 0
fi
# DINO_VERIFIER_SMOKE=1 restricts evaluate.py to construction + parameter
# checks + the k-NN probe on a subset — infrastructure debugging only.
if ! python3 /tests/evaluate.py; then
  zero frozen_evaluation_failed
  exit 0
fi
cp /app/output/verifier-metrics.json /logs/verifier/metrics.json
python3 /tests/score.py
