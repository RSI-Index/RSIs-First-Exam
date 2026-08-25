#!/usr/bin/env bash
set -euo pipefail

# Remove all errored trial directories from a job folder so harbor can rerun them.
#
# Usage:
#   ./scripts/clean_errors.sh jobs/my_job_name
#   ./scripts/clean_errors.sh jobs/my_job_name AgentTimeoutError   # only remove specific error type
#
# With no error type filter, removes ALL trials that have a non-null exception_info.

JOB_DIR="${1:?Usage: $0 <job-dir> [error-type]}"
ERROR_TYPE="${2:-}"

if [ ! -d "$JOB_DIR" ]; then
    echo "Error: $JOB_DIR does not exist"
    exit 1
fi

count=0
for trial_dir in "$JOB_DIR"/*/; do
    result="$trial_dir/result.json"
    [ -f "$result" ] || continue

    if [ -n "$ERROR_TYPE" ]; then
        match=$(python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
ei = d.get('exception_info')
print('yes' if ei and ei.get('exception_type') == sys.argv[2] else 'no')
" "$result" "$ERROR_TYPE")
    else
        match=$(python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
print('yes' if d.get('exception_info') else 'no')
" "$result")
    fi

    if [ "$match" = "yes" ]; then
        name=$(basename "$trial_dir")
        echo "Removing $name"
        rm -rf "$trial_dir"
        count=$((count + 1))
    fi
done

echo "Removed $count errored trial(s) from $JOB_DIR"
