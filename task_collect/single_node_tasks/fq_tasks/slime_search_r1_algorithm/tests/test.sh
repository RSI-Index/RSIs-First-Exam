#!/usr/bin/env bash
set -uo pipefail
umask 077

reward_dir="${REWARD_DIR:-/logs/verifier}"
if [ -L "$reward_dir" ]; then
  exit 1
fi
mkdir -p "$reward_dir"
diagnostic_dir="$(mktemp -d "$reward_dir/diagnostics.XXXXXX")"
printf '{"reward":0.0}\n' > "$reward_dir/reward.json"

if ! python3 /tests/policy_check.py > "$diagnostic_dir/policy.log" 2>&1; then
  exit 0
fi
if ! python3 /tests/evaluate.py --output "$diagnostic_dir/aggregates.json" > "$diagnostic_dir/evaluate.log" 2>&1; then
  exit 0
fi
if ! python3 /tests/score.py --aggregates "$diagnostic_dir/aggregates.json" --reward "$reward_dir/reward.json" > "$diagnostic_dir/score.log" 2>&1; then
  printf '{"reward":0.0}\n' > "$reward_dir/reward.json"
fi
exit 0
