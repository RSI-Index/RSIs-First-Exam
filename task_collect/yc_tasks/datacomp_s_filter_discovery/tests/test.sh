#!/usr/bin/env bash
set -uo pipefail
log_root="${LOG_DIR:-/logs/verifier}"
mkdir -p "$log_root"
baseline="${DATACOMP_BASELINE_CONTRACT:-/opt/contracts/baseline_contract.json}"
if [[ ! -f "$baseline" ]]; then
  baseline=/tests/baseline_contract.json
fi

zero() {
  python - "$log_root" "$1" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
reason = sys.argv[2]
(root / "reward.json").write_text(json.dumps({"reward": 0.0, "policy_gate": 0, "reason": reason}, indent=2) + "\n")
(root / "reward.txt").write_text("0.0\n")
(root / "failure.json").write_text(json.dumps({"reason": reason}, indent=2) + "\n")
PY
}

if ! python /tests/policy_check.py --policy /tests/policy.yaml --clean /opt/candidate-clean --candidate /app/project --output-dir /app/output; then
  zero reward_integrity_gate
  exit 0
fi
if ! python - "$baseline" <<'PY'
import json
import sys
from pathlib import Path
value = json.loads(Path(sys.argv[1]).read_text())
if value.get('status') != 'complete':
    raise SystemExit(1)
for field in ('reference_run_id', 'reference_metric', 'candidate_threshold',
              'evaluator_tolerance', 'reference_checkpoint_sha256',
              'initialization_parameter_digest', 'asset_manifest_sha256',
              'runtime_image_id'):
    if value.get(field) in (None, ''):
        raise SystemExit(1)
PY
then
  zero baseline_contract_invalid
  exit 0
fi
if ! python /tests/artifact_check.py --baseline "$baseline"; then
  zero artifact_contract_failed
  exit 0
fi
if ! python /tests/evaluate.py; then
  zero clean_evaluation_failed
  exit 0
fi
if ! python /tests/score.py --baseline "$baseline"; then
  zero score_mapping_failed
  exit 0
fi
