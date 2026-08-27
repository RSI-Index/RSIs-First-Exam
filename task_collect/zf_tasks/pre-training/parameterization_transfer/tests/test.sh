#!/usr/bin/env bash
set -uo pipefail

logs=/logs/verifier
mkdir -p -- "${logs}"

if [[ ${PARAMETERIZATION_SMOKE_VERIFIER:-0} == 1 ]]; then
    /opt/venv/bin/python - <<'PY'
import json
from pathlib import Path

Path("/app/project").resolve(strict=True)
Path("/app/output").resolve(strict=True)
logs = Path("/logs/verifier")
(logs / "reward.json").write_text(json.dumps({"reward": 1.0}, indent=2) + "\n")
(logs / "smoke.json").write_text(json.dumps({
    "mode": "infrastructure-smoke",
    "note": "Visible-freeze contract verification is intentionally disabled.",
}, indent=2) + "\n")
(logs / "reward.txt").write_text("1.0\n")
PY
    exit 0
fi

zero() {
    python3 - "${1}" <<'PY'
import json
import sys
from pathlib import Path

logs = Path("/logs/verifier")
(logs / "reward.json").write_text(json.dumps({"reward": 0.0, "policy_gate": 0}, indent=2) + "\n")
(logs / "reward.txt").write_text("0.0\n")
(logs / "failure.json").write_text(json.dumps({"reason": sys.argv[1]}, indent=2) + "\n")
PY
}

deadline_file=/run-contract/RESEARCH_DEADLINE_UTC
if [[ ! -r ${deadline_file} ]]; then
    zero trusted_research_deadline_missing
    exit 0
fi
if ! deadline_epoch=$(date -u -d "$(tr -d '[:space:]' <"${deadline_file}")" +%s 2>/dev/null); then
    zero trusted_research_deadline_invalid
    exit 0
fi
if (( $(date -u +%s) < deadline_epoch )); then
    zero agent_exited_before_research_deadline
    exit 0
fi

cd /
if ! /opt/venv/bin/python /task-tools/parameterization_scaling_task.py restore-source \
    --output-root /app/output \
    --clean-project /opt/project \
    --project /app/project; then
    zero incumbent_source_restore_failed
    exit 0
fi

if ! /opt/venv/bin/python /tests/policy_check.py \
    --policy /tests/policy.yaml \
    --clean /opt/project \
    --candidate /app/project \
    --report "${logs}/policy.json"; then
    zero source_policy_gate
    exit 0
fi

paths_manifest=${logs}/submission_paths.json
if ! /opt/venv/bin/python /tests/validate_submission.py \
    --output-root /app/output \
    --manifest "${paths_manifest}" >"${logs}/validated_submission.json"; then
    zero invalid_submission
    exit 0
fi

if ! /opt/venv/bin/python /tests/score.py \
    --output-root /app/output \
    --policy-report "${logs}/policy.json" \
    --logs-dir "${logs}"; then
    zero invalid_evaluation_results
    exit 0
fi
