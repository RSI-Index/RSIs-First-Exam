#!/usr/bin/env bash
set -euo pipefail
stage="${1:-selector}"
if [[ "$stage" != "selector" && "$stage" != "full" ]]; then
  echo "usage: $0 [selector|full]" >&2
  exit 2
fi
exec python /task-tools/run_candidate.py --stage "$stage" --output-root /app/output
