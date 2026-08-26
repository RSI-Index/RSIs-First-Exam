#!/usr/bin/env bash
set -uo pipefail
mkdir -p /logs/verifier

zero() {
  python3 - "$1" <<'PY'
import json, sys
result = {"reward": 0.0, "policy_gate": 0, "reason": sys.argv[1]}
open("/logs/verifier/reward.json", "w").write(json.dumps(result, indent=2))
open("/logs/verifier/reward.txt", "w").write("0.0\n")
PY
}

if ! /opt/policy-venv/bin/python /tests/policy_check.py --policy /tests/policy.yaml --clean /opt/project --candidate /app/project; then
  zero reward_integrity_gate
  exit 0
fi
# evaluate.py computes the recomputed reward and writes reward.json itself
# (no separate score step: a subprocess-writable handoff file would be a
# tamper window between two verifier stages).
if ! python3 /tests/evaluate.py; then
  zero official_evaluation_failed
  exit 0
fi
if [ ! -s /logs/verifier/reward.json ]; then
  zero verifier_reward_missing
  exit 0
fi
exit 0
