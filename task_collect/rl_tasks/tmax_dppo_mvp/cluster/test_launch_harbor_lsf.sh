#!/usr/bin/env bash
set -euo pipefail

cluster_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
launcher=${cluster_dir}/launch_harbor_lsf.sh

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

output=$(RUN_ID=tmax-ar-contract-test "${launcher}" --dry-run codex/gpt-5.6-sol) || \
    fail "TMAX Harbor dry-run failed"

for required in \
    "-n 8" \
    "-gpu num=8:mode=exclusive_process" \
    "-W 72:00" \
    "tmax-ar-contract-test" \
    "--agent codex" \
    "--model codex/gpt-5.6-sol" \
    "reasoning_effort=xhigh" \
    "web_search=disabled" \
    "--n-attempts 1" \
    "envs_backend.lsf_apptainer:LsfApptainerEnvironment"; do
    [[ ${output} == *"${required}"* ]] || fail "dry-run is missing: ${required}"
done

[[ ${output} != *"bsub"*"bsub"* ]] || fail "launcher rendered nested bsub submission"
echo "PASS: TMAX Harbor/LSF launcher contract"
