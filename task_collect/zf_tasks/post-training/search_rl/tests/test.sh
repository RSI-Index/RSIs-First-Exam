#!/usr/bin/env bash
set -euo pipefail

tests_root=${TASK_TESTS_ROOT:-/tests}
project=${TASK_PROJECT:-/app/project}
output=${TASK_OUTPUT_ROOT:-/app/output}
logs=${VERIFIER_LOGS_ROOT:-/logs/verifier}
policy=${TASK_POLICY_PATH:-}

if [[ -n ${TASK_ROOT:-} ]]; then
  tests_root=${TASK_TESTS_ROOT:-${TASK_ROOT}/tests}
  policy=${TASK_POLICY_PATH:-${TASK_ROOT}/policy.yaml}
fi

command=(python3 "${tests_root}/verify_layer1.py" \
  --project "${project}" \
  --output "${output}" \
  --logs "${logs}" \
  --task search-rl)
if [[ -n ${policy} ]]; then
  command+=(--policy "${policy}")
fi
"${command[@]}"
