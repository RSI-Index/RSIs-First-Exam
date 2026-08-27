#!/usr/bin/env bash
set -euo pipefail
stage="${1:-validate}"
exec python3 /task-tools/run_candidate.py --stage "$stage" --candidate-root /app/project/candidate --output-root /app/output

