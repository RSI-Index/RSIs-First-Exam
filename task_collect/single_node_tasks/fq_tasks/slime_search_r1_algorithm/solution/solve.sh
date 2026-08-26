#!/usr/bin/env bash
set -euo pipefail

attempt="baseline-grpo-001"
python3 /task-tools/slime_async_run.py submit \
  --attempt-id "$attempt" \
  --hypothesis "the pinned GRPO starter establishes the matched 1000-round Slime Search-R1 baseline"

while true; do
  status_json="$(python3 /task-tools/slime_async_run.py status --attempt-id "$attempt")"
  status="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])' <<<"$status_json")"
  case "$status" in
    completed) break ;;
    failed) printf '%s\n' "$status_json" >&2; exit 1 ;;
    queued|running) sleep 30 ;;
    *) printf 'unexpected attempt status: %s\n' "$status" >&2; exit 1 ;;
  esac
done

python3 /task-tools/slime_async_run.py select --attempt-id "$attempt"
python3 /task-tools/slime_async_run.py complete-control
