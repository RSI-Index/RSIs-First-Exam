#!/usr/bin/env bash
set -euo pipefail

contract=${1:-}
[[ -f ${contract} ]] || { echo "trusted attempt contract is missing: ${contract}" >&2; exit 2; }
run_root=${TMAX_RUN_ROOT:?trusted broker must set TMAX_RUN_ROOT}
runner=${TMAX_64GPU_RUNNER:?trusted broker must set TMAX_64GPU_RUNNER}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)

contract=$(realpath -e -- "${contract}")
run_root=$(realpath -e -- "${run_root}")
case ${contract} in
    "${run_root}"/attempts/*/run_contract.json) ;;
    *) echo "attempt contract is outside the trusted run root: ${contract}" >&2; exit 2 ;;
esac

attempt_id=$(jq -er '.attempt_id' "${contract}")
max_updates=$(jq -er '.max_updates' "${contract}")
[[ ${attempt_id} == "$(basename -- "$(dirname -- "${contract}")")" ]] || {
    echo "attempt_id does not match its trusted attempt directory: ${attempt_id}" >&2
    exit 2
}
[[ ${max_updates} =~ ^[0-9]+$ ]] && ((max_updates >= 1 && max_updates <= 200)) || {
    echo "max_updates must be an integer from 1 through 200" >&2
    exit 2
}

map_snapshot_path() {
    local raw=${1:?snapshot path is required}
    local label=${2:?snapshot label is required}
    local mapped resolved
    case ${raw} in
        /app/output/*) mapped="${run_root}/${raw#/app/output/}" ;;
        "${run_root}"/*) mapped=${raw} ;;
        *) echo "${label} is outside /app/output: ${raw}" >&2; return 2 ;;
    esac
    resolved=$(realpath -e -- "${mapped}") || {
        echo "${label} does not exist below the trusted run root: ${mapped}" >&2
        return 2
    }
    case ${resolved} in
        "${run_root}"/*) printf '%s\n' "${resolved}" ;;
        *) echo "${label} escapes the trusted run root: ${resolved}" >&2; return 2 ;;
    esac
}

project_root=$(map_snapshot_path "$(jq -er '.source_snapshot' "${contract}")" source_snapshot)
training_data=$(map_snapshot_path "$(jq -er '.training_data_snapshot' "${contract}")" training_data_snapshot)
training_prompt=$(map_snapshot_path "$(jq -er '.training_prompt_snapshot' "${contract}")" training_prompt_snapshot)
training_reward=$(map_snapshot_path "$(jq -er '.training_reward_snapshot' "${contract}")" training_reward_snapshot)
training_root=$(map_snapshot_path "$(jq -er '.training_artifact_snapshot' "${contract}")" training_artifact_snapshot)
rl_recipe=$(map_snapshot_path "$(jq -er '.rl_recipe_snapshot' "${contract}")" rl_recipe_snapshot)

python3 - "${script_dir}" \
    "${project_root}" "$(jq -er '.source_sha256' "${contract}")" \
    "${training_data}" "$(jq -er '.training_data_sha256' "${contract}")" \
    "${training_prompt}" "$(jq -er '.training_prompt_sha256' "${contract}")" \
    "${training_reward}" "$(jq -er '.training_reward_sha256' "${contract}")" \
    "${rl_recipe}" "$(jq -er '.rl_recipe_sha256' "${contract}")" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from tmax_contract import tree_digest

labels = (
    "source_sha256",
    "training_data_sha256",
    "training_prompt_sha256",
    "training_reward_sha256",
    "rl_recipe_sha256",
)
arguments = sys.argv[2:]
for label, path, expected in zip(labels, arguments[::2], arguments[1::2]):
    if tree_digest(Path(path)) != expected:
        raise SystemExit(f"scientific snapshot hash mismatch: {label}")
PY

export TMAX_ATTEMPT_ID=${attempt_id}
export TMAX_MAX_UPDATES=${max_updates}
export TMAX_SCHEDULER_HORIZON_UPDATES=200
export TMAX_PROJECT_ROOT=${project_root}
export TMAX_RUN_ROOT=${run_root}
export TMAX_TRAINING_DATA_DIR=${training_data}
export TMAX_TRAINING_PROMPT_FILE=${training_prompt}
export TMAX_TRAINING_REWARD_FILE=${training_reward}
export TMAX_TRAINING_ARTIFACT_ROOT=${training_root}
export TMAX_RL_RECIPE_FILE=${rl_recipe}

set +e
"${runner}"
runner_status=$?
set -e

python3 - "${contract}" "${runner_status}" <<'PY'
import json
import os
import sys
from pathlib import Path

contract_path = Path(sys.argv[1])
runner_status = int(sys.argv[2])
contract = json.loads(contract_path.read_text())
attempt = contract_path.parent
updates = int(contract["max_updates"]) if runner_status == 0 else 0
counters = {
    "optimizer_updates": updates,
    "training_trajectories": updates * 256,
    "generated_tokens": 0,
    "sandbox_steps": 0,
    "gpu_count": 64,
    "metrics": {},
}
path = attempt / "trusted_counters.json"
temporary = attempt / f".{path.name}.tmp.{os.getpid()}"
temporary.write_text(json.dumps(counters, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
exit "${runner_status}"
