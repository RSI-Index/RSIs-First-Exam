#!/usr/bin/env bash
set -uo pipefail

logs=/logs/verifier
mkdir -p -- "${logs}"

if [[ ${LR_SCHEDULE_SMOKE_VERIFIER:-0} == 1 ]]; then
    /opt/venv/bin/python - <<'PY'
import json
from pathlib import Path

Path("/app/project").resolve(strict=True)
Path("/app/output").resolve(strict=True)
logs = Path("/logs/verifier")
(logs / "reward.json").write_text(json.dumps({"reward": 1.0}, indent=2) + "\n")
(logs / "smoke.json").write_text(json.dumps({
    "mode": "infrastructure-smoke",
    "note": "Full six-rung Paloma verification is intentionally disabled.",
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
if ! /opt/venv/bin/python /task-tools/lr_schedule_task.py restore-source \
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
    --baselines /task-data/lr_schedule_baselines.json \
    --manifest "${paths_manifest}" >"${logs}/validated_submission.json"; then
    zero invalid_submission
    exit 0
fi

verifier_id=${LSB_JOBID:-$$}
verifier_root="/app/output/verifier-eval-${verifier_id}"
if [[ -e ${verifier_root} ]]; then
    zero verifier_output_collision
    exit 0
fi
mkdir -p -- "${verifier_root}"

scales=(E0 E1 E2 E3 E4 E5)
pids=()
for scale in "${scales[@]}"; do
    checkpoint=$(/opt/venv/bin/python - "${paths_manifest}" "${scale}" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1]))["scales"][sys.argv[2]]["checkpoint"])
PY
)
    /task-tools/run_lr_schedule_ladder.sh \
        --eval-only \
        --scale "${scale}" \
        --checkpoint "${checkpoint}" \
        --run-dir "${verifier_root}/${scale}" \
        >"${logs}/eval-${scale}.log" 2>&1 &
    pids+=("$!")
done

evaluation_failed=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[index]}"; then
        printf 'scale %s independent evaluation failed\n' "${scales[index]}" >&2
        evaluation_failed=1
    fi
done
if (( evaluation_failed == 1 )); then
    zero independent_evaluation_failed
    exit 0
fi

if ! /opt/venv/bin/python /tests/score.py \
    --runs-root "${verifier_root}" \
    --baselines /task-data/lr_schedule_baselines.json \
    --policy-report "${logs}/policy.json" \
    --logs-dir "${logs}"; then
    zero invalid_evaluation_results
    exit 0
fi
