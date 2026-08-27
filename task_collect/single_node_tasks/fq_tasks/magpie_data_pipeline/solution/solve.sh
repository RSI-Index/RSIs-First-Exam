#!/usr/bin/env bash
set -euo pipefail

runner="${MAGPIE_ASYNC_RUN:-/task-tools/magpie_async_run}"
attempt="filtered-prefix-001"
hypothesis="Filtering to the deterministic 50K baseline prefix trades coverage for data efficiency."
poll_seconds="${MAGPIE_POLL_SECONDS:-30}"
max_polls="${MAGPIE_MAX_POLLS:-900}"

"${runner}" submit --attempt-id "${attempt}" --hypothesis "${hypothesis}"

for ((poll = 0; poll < max_polls; poll += 1)); do
    status="$("${runner}" status --attempt-id "${attempt}")"
    status_kind="$(printf '%s' "${status}" | python3 -c '
import json
import sys

payload = json.load(sys.stdin)
value = payload.get("status") if isinstance(payload, dict) else None
if value not in {"queued", "running", "completed", "failed"}:
    raise ValueError("status payload has an invalid status")
print(value)
')"
    if [[ "${status_kind}" == "completed" ]]; then
        "${runner}" select --attempt-id "${attempt}"
        "${runner}" complete-control
        exit 0
    fi
    if [[ "${status_kind}" == "failed" ]]; then
        printf 'attempt %s failed:\n%s\n' "${attempt}" "${status}" >&2
        exit 1
    fi
    sleep "${poll_seconds}"
done

printf 'attempt %s did not complete after %s status polls\n' "${attempt}" "${max_polls}" >&2
exit 1
