#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run one rung inside the enclosing 256-GPU scaling-ladder allocation.

Candidate-visible training:
  run_parameterization_scaling_ladder.sh --scale E0 --attempt-id ID --hypothesis-file FILE [--resume-checkpoint DIR] [--dry-run]

Concurrent invocations atomically lease disjoint whole nodes. Only E0-E4 are
available here. E5 is held out and has no candidate-accessible launcher.
EOF
}

scale=
attempt_id=
hypothesis_file=
resume_checkpoint=
dry_run=0
while (( $# > 0 )); do
    case "$1" in
        --scale) scale=${2:-}; shift 2 ;;
        --attempt-id) attempt_id=${2:-}; shift 2 ;;
        --hypothesis-file) hypothesis_file=${2:-}; shift 2 ;;
        --resume-checkpoint) resume_checkpoint=${2:-}; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'error: unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ ${scale} =~ ^E[0-4]$ ]] || { printf 'error: --scale must be E0 through E4; E5 is held out\n' >&2; exit 2; }
export PARAMETERIZATION_SCALE=${scale}
task_tools=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=parameterization_scaling_ladder_profile.sh
source "${task_tools}/parameterization_scaling_ladder_profile.sh"

case "${scale}" in
    E0) minimum_full_run_seconds=18000 ;;
    E1|E2) minimum_full_run_seconds=27000 ;;
    E3) minimum_full_run_seconds=22000 ;;
    E4) minimum_full_run_seconds=36000 ;;
esac

if (( dry_run == 0 )); then
    deadline_file=/run-contract/RESEARCH_DEADLINE_UTC
    [[ -r ${deadline_file} ]] || {
        printf 'error: trusted research deadline is missing: %s\n' "${deadline_file}" >&2
        exit 2
    }
    deadline_epoch=$(date -u -d "$(tr -d '[:space:]' <"${deadline_file}")" +%s)
    remaining_seconds=$((deadline_epoch - $(date -u +%s)))
    (( remaining_seconds >= minimum_full_run_seconds )) || {
        printf 'error: %s seconds remain; %s requires at least %s\n' \
            "${remaining_seconds}" "${scale}" "${minimum_full_run_seconds}" >&2
        exit 2
    }
fi

[[ ${attempt_id} =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ ]] || {
    printf 'error: --attempt-id must contain 1-96 safe identifier characters\n' >&2
    exit 2
}
[[ -f ${hypothesis_file} && -s ${hypothesis_file} ]] || {
    printf 'error: --hypothesis-file must name a non-empty file\n' >&2
    exit 2
}
run_dir="${PARAMETERIZATION_OUTPUT_ROOT}/attempts/${attempt_id}"
[[ ! -e ${run_dir} ]] || { printf 'error: attempt already exists: %s\n' "${run_dir}" >&2; exit 2; }
if [[ -n ${resume_checkpoint} ]]; then
        [[ ${resume_checkpoint} == "${PARAMETERIZATION_OUTPUT_ROOT}/attempts/"*/checkpoints ]] || {
            printf 'error: --resume-checkpoint must be an attempt checkpoint directory below %s\n' "${PARAMETERIZATION_OUTPUT_ROOT}" >&2
            exit 2
        }
        [[ -f ${resume_checkpoint}/latest_checkpointed_iteration.txt ]] || {
            printf 'error: resume checkpoint tracker is missing: %s\n' "${resume_checkpoint}" >&2
            exit 2
        }
        resume_iteration=$(tr -d '[:space:]' <"${resume_checkpoint}/latest_checkpointed_iteration.txt")
        [[ ${resume_iteration} =~ ^[0-9]+$ ]] && (( resume_iteration > 0 && resume_iteration < TRAIN_ITERS )) || {
            printf 'error: invalid resume iteration %s for %s final iteration %s\n' \
                "${resume_iteration}" "${scale}" "${TRAIN_ITERS}" >&2
            exit 2
        }
        resume_attempt=$(dirname -- "${resume_checkpoint}")
        [[ -s ${resume_attempt}/run_contract.json ]] || {
            printf 'error: resume attempt contract is missing\n' >&2
            exit 2
        }
        resume_scale=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["scale"])' \
            "${resume_attempt}/run_contract.json")
        [[ ${resume_scale} == "${scale}" ]] || {
            printf 'error: resume checkpoint scale %s does not match %s\n' "${resume_scale}" "${scale}" >&2
            exit 2
        }
fi
lease_id=${attempt_id}
export EVAL_ONLY=0 EXPECT_FINAL_CHECKPOINT=1

if (( dry_run == 1 )); then
    cat <<EOF
mode: dry-run
scale: ${scale}
model: d${MODEL_HIDDEN_SIZE}-L${MODEL_NUM_LAYERS}, baseline parameters ${SOURCE_PARAMETER_COUNT}
run directory: ${run_dir}
training: ${TRAIN_ITERS} updates, ${TARGET_TOKENS} tokens, sequence length ${SEQUENCE_LENGTH}
batch: GBS=${GLOBAL_BATCH_SIZE}, MBS=${MICRO_BATCH_SIZE}
topology: ${EXPECTED_NODES} nodes x ${GPUS_PER_NODE} GPUs = ${LSF_SLOTS} GPUs, TP=1, PP=1
outer allocation: 32 nodes x 8 GPUs = 256 GPUs
EOF
    exit 0
fi

[[ -n ${LSB_JOBID:-} && -n ${LSB_MCPU_HOSTS:-} ]] || {
    printf 'error: the runner must execute inside the enclosing LSF allocation\n' >&2
    exit 2
}
[[ -r ${PARAMETERIZATION_HOST_OUTPUT_FILE} ]] || {
    printf 'error: host output-root contract is missing: %s\n' "${PARAMETERIZATION_HOST_OUTPUT_FILE}" >&2
    exit 2
}
host_output_root=$(tr -d '\r\n' <"${PARAMETERIZATION_HOST_OUTPUT_FILE}")
[[ ${host_output_root} == /proj/* ]] || { printf 'error: unsafe host output root\n' >&2; exit 2; }

mkdir -p -- "$(dirname -- "${run_dir}")"
prepare_args=(
        prepare
        --attempt "${run_dir}"
        --attempt-id "${attempt_id}"
        --scale "${scale}"
        --hypothesis-file "${hypothesis_file}"
        --project "${PARAMETERIZATION_PROJECT}"
    )
if [[ -n ${resume_checkpoint} ]]; then
        prepare_args+=(
            --resume-iteration "${resume_iteration}"
            --resume-attempt-id "$(basename -- "${resume_attempt}")"
        )
fi
python "${task_tools}/parameterization_scaling_task.py" "${prepare_args[@]}"

source_hash=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_inventory_sha256"])' \
    "${run_dir}/run_contract.json")
source_root="${PARAMETERIZATION_OUTPUT_ROOT}/source-worktrees/${source_hash}"
source_project="${source_root}/project"
source_lock="${PARAMETERIZATION_OUTPUT_ROOT}/source-worktrees/${source_hash}.lock"
mkdir -p -- "$(dirname -- "${source_root}")"
exec 9>"${source_lock}"
flock 9
if [[ ! -d ${source_project} ]]; then
    source_tmp="${source_root}.tmp.$$"
    [[ ! -e ${source_tmp} ]] || { printf 'error: source temp collision\n' >&2; exit 2; }
    mkdir -- "${source_tmp}"
    python "${task_tools}/parameterization_scaling_task.py" materialize-attempt-source \
            --attempt "${run_dir}" \
            --clean-project /opt/project \
            --project "${source_tmp}/project"
    copied_hash=$(python "${task_tools}/parameterization_scaling_task.py" hash --project "${source_tmp}/project")
    [[ ${copied_hash} == "${source_hash}" ]] || {
        printf 'error: candidate source changed while it was being snapshotted\n' >&2
        exit 2
    }
    mv -- "${source_tmp}" "${source_root}"
fi
flock -u 9

container_relative_run=${run_dir#${PARAMETERIZATION_OUTPUT_ROOT}/}
host_run_dir="${host_output_root}/${container_relative_run}"
host_source_project="${host_output_root}/source-worktrees/${source_hash}/project"
[[ -d ${host_source_project} ]] || { printf 'error: host cannot see source snapshot\n' >&2; exit 2; }
if [[ -n ${resume_checkpoint} ]]; then
    resume_source_hash=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_inventory_sha256"])' \
        "${resume_attempt}/run_contract.json")
    [[ ${resume_source_hash} == "${source_hash}" ]] || {
        printf 'error: resume checkpoint source %s does not match candidate source %s\n' \
            "${resume_source_hash}" "${source_hash}" >&2
        exit 2
    }
    host_checkpoint="${host_output_root}/${resume_checkpoint#${PARAMETERIZATION_OUTPUT_ROOT}/}"
    export LOAD_DIR=${host_checkpoint}
fi

lease_state="${PARAMETERIZATION_OUTPUT_ROOT}/host-leases.json"
lease_token=
cleanup() {
    status=$?
    trap - EXIT INT TERM
    if [[ -n ${lease_token:-} ]]; then
        python "${task_tools}/host_lease.py" release \
            --state "${lease_state}" --attempt-id "${lease_id}" --token "${lease_token}" || true
    fi
    exit "${status}"
}
trap cleanup EXIT INT TERM

lease_line=$(python "${task_tools}/host_lease.py" acquire \
    --state "${lease_state}" \
    --attempt-id "${lease_id}" \
    --slots "${LSF_SLOTS}" \
    --pid "$$" \
    --lsb-hosts "${LSB_MCPU_HOSTS}")
IFS=$'\t' read -r lease_token leased_lsf_hosts <<<"${lease_line}"
[[ -n ${lease_token} && -n ${leased_lsf_hosts} ]] || { printf 'error: invalid node lease\n' >&2; exit 2; }

export LSB_MCPU_HOSTS=${leased_lsf_hosts}
export RUN_DIR=${host_run_dir}
export PARAMETERIZATION_WANDB_RUN_NAME=${attempt_id:-verifier-${scale}}
export WANDB_RUN_NAME=${PARAMETERIZATION_WANDB_RUN_NAME}
# The enclosing Harbor allocation also exports MASTER_PORT for its own
# rendezvous.  Do not leak that port into a nested training attempt: the first
# leased host may be Harbor's master, where the inherited port is already in
# use.  Give every lease a deterministic, attempt-specific port below the
# ephemeral range instead.
read -r master_port_seed _ < <(printf '%s' "${LSB_JOBID}:${lease_id}:${lease_token}" | cksum)
export MASTER_PORT=$((20000 + master_port_seed % 30000))
launcher="${host_source_project}/examples/training/parameterization/run_marin_adamh_bluevela_lsf.sh"
[[ -f ${launcher} ]] || { printf 'error: shared launcher is missing: %s\n' "${launcher}" >&2; exit 2; }

set +e
bash "${launcher}" --run 2>&1 | tee "${run_dir}/run.log"
run_rc=${PIPESTATUS[0]}
set -e

python "${task_tools}/parameterization_scaling_task.py" finish \
    --attempt "${run_dir}" \
    --project "${source_project}" \
    --exit-code "${run_rc}" || run_rc=$?
exit "${run_rc}"
