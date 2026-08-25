#!/usr/bin/env bash
set -euo pipefail

output=${TASK_OUTPUT_ROOT:-/app/output}
tools_root=${TASK_TOOLS_ROOT:-/task-tools}
if [[ -n ${TASK_ROOT:-} ]]; then
  tools_root=${TASK_TOOLS_ROOT:-${TASK_ROOT}/environment/task-tools}
fi

attempt_id=oracle-smoke
hypothesis=${output}/hypotheses/${attempt_id}.json
run_dir=${output}/attempts/${attempt_id}
mkdir -p "$(dirname "${hypothesis}")"

python3 - "${hypothesis}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
path.write_text(json.dumps({
    "schema_version": 1,
    "attempt_id": "oracle-smoke",
    "hypothesis": "Exercise the redacted-source, task-tool, and artifact contracts.",
    "validation_mode": "layer1-smoke",
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

python3 "${tools_root}/preflight.py" \
  --attempt-id "${attempt_id}" --hypothesis-file "${hypothesis}" --output-root "${output}"
python3 "${tools_root}/run_candidate.py" \
  --attempt-id "${attempt_id}" --hypothesis-file "${hypothesis}" --output-root "${output}" --dry-run
python3 "${tools_root}/evaluate.py" --run-dir "${run_dir}"
python3 "${tools_root}/task_state.py" record --run-dir "${run_dir}" --output-root "${output}"
python3 "${tools_root}/task_state.py" stage --run-dir "${run_dir}" --output-root "${output}"

python3 - "${output}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
staged = json.loads((root / "staged_candidate.json").read_text(encoding="utf-8"))
run_dir = root / staged["run_dir"]
candidate = json.loads((run_dir / "candidate_result.json").read_text(encoding="utf-8"))
evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
submission = {
    "schema_version": 1,
    "task": "search-rl",
    "attempt_id": staged["attempt_id"],
    "run_dir": staged["run_dir"],
    "validation_mode": "layer1-smoke",
    "scientific_validation": "not-run",
    "source_tree_sha256": candidate["source_tree_sha256"],
    "smoke_evaluation_passed": evaluation["passed"],
}
(root / "submission.json").write_text(
    json.dumps(submission, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
