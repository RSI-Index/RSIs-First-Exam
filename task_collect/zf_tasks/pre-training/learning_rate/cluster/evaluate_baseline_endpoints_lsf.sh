#!/usr/bin/env bash
set -euo pipefail

task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
asset_root=${LR_SCHEDULE_ASSET_ROOT:-/proj/datasets/interns/yuetai/agent_envs/more_task}
source_repo=${MBRIDGE_REPO:-${task_root}/../../../../pre_training/Megatron-Bridge}
image=${LR_SCHEDULE_CONTAINER_IMAGE:-${asset_root}/images/learning-rate-environment-accefd7b3a44-r1.sif}
default_lsf_resource="select[hname!='p1-r01-n1' && hname!='p1-r03-n2' && hname!='p1-r10-n4' && hname!='p1-r12-n3' && hname!='p2-r18-n1' && hname!='p2-r19-n3' && hname!='p3-r02-n4' && hname!='p3-r04-n2' && hname!='p3-r04-n4' && hname!='p3-r05-n1' && hname!='p3-r12-n4' && hname!='p3-r18-n2' && hname!='p3-r26-n4' && hname!='p4-r03-n3' && hname!='p4-r11-n2' && hname!='p4-r27-n1' && hname!='p5-r09-n3' && hname!='p6-r04-n3'] span[ptile=8]"
bluevela_env_file=${BLUEVELA_ENV_FILE:-${XDG_CONFIG_HOME:-${HOME:-}/.config}/megatron-bridge/bluevela.env}
if [[ -r ${bluevela_env_file} ]]; then
    # shellcheck source=/dev/null
    source "${bluevela_env_file}"
fi

usage() {
    cat <<'EOF'
Evaluate the existing locked WSD exact-final E4/E5 checkpoints with Paloma.

Usage:
  evaluate_baseline_endpoints_lsf.sh --dry-run [E4|E5|all]
  evaluate_baseline_endpoints_lsf.sh --submit [E4|E5|all]
  evaluate_baseline_endpoints_lsf.sh --run E4|E5 RUN_DIR
EOF
}

mode=${1:---dry-run}
selection=${2:-all}
case "${mode}" in
    --dry-run|--submit)
        (( $# <= 2 )) || { usage >&2; exit 2; }
        [[ ${selection} == E4 || ${selection} == E5 || ${selection} == all ]] \
            || { usage >&2; exit 2; }
        ;;
    --run)
        (( $# == 3 )) || { usage >&2; exit 2; }
        scale=$2
        run_dir=$3
        [[ ${scale} == E4 || ${scale} == E5 ]] || { usage >&2; exit 2; }
        ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

e4_checkpoint=${asset_root}/runs/marin-adamh-baseline/20260729T182645Z-marin-adamh-1p8e20-p1935m-t14p805b-d2048-l21-8gpu-mbs2-ga4-llama3-b64-norecompute-nooffload-gradaccfusion-pretraining/checkpoints/iter_0056477
e5_checkpoint=${asset_root}/runs/marin-adamh-baseline/20260729T182645Z-marin-adamh-3e20-p2545m-t18p617b-d2304-l23-16gpu-mbs1-ga8-llama3-b128-norecompute-nooffload-gradaccfusion-pretraining/checkpoints/iter_0035510

if [[ ${mode} != --run ]]; then
    [[ -s ${image} ]] || {
        printf 'error: learning-rate environment image is missing: %s\n' "${image}" >&2
        exit 2
    }
    [[ -f ${e4_checkpoint}/metadata.json && -f ${e5_checkpoint}/metadata.json ]] || {
        printf 'error: one or both exact-final baseline checkpoints are missing\n' >&2
        exit 2
    }
    timestamp=${RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
    scales=(E4 E5)
    [[ ${selection} == all ]] || scales=("${selection}")
    for scale in "${scales[@]}"; do
        case "${scale}" in
            E4) slots=8; job_name=mt-lr-base-e4-yuetai ;;
            E5) slots=16; job_name=mt-lr-base-e5-yuetai ;;
        esac
        run_dir=${asset_root}/baseline_evaluations/learning_rate/${timestamp}-${scale,,}
        command=(
            bsub -q "${LSF_QUEUE:-priority}" -G "${LSF_GROUP:-grp_models}"
            -n "${slots}" -M "${LSF_MEMORY:-32768}" -W "${LSF_WALLTIME:-04:00}"
            -R "${LSF_SPAN:-${default_lsf_resource}}" -gpu 'num=8:mode=exclusive_process'
            -J "${job_name}"
            -o "${run_dir}/lsf.%J.out" -e "${run_dir}/lsf.%J.err"
            bash "${task_root}/cluster/evaluate_baseline_endpoints_lsf.sh"
            --run "${scale}" "${run_dir}"
        )
        printf '%s run directory: %s\n' "${scale}" "${run_dir}"
        printf 'LSF command:'
        printf ' %q' "${command[@]}"
        printf '\n'
        if [[ ${mode} == --submit ]]; then
            [[ ! -e ${run_dir} ]] || { printf 'error: refusing to reuse %s\n' "${run_dir}" >&2; exit 2; }
            mkdir -p -- "${run_dir}"
            "${command[@]}"
        fi
    done
    exit 0
fi

: "${LSB_JOBID:?--run must execute inside LSF}"
[[ ${run_dir} == "${asset_root}/baseline_evaluations/learning_rate/"* ]] || {
    printf 'error: unsafe endpoint evaluation run directory: %s\n' "${run_dir}" >&2
    exit 2
}
[[ -d ${run_dir} ]] || { printf 'error: run directory is missing\n' >&2; exit 2; }
project=${run_dir}/project
evaluation=${run_dir}/evaluation
[[ ! -e ${project} && ! -e ${evaluation} ]] || {
    printf 'error: refusing to reuse materialized project or evaluation output\n' >&2
    exit 2
}

bridge_commit=3a6c6fb3811a10f532d26be7dd032457795a8e41
mcore_commit=58bf14e9e68915a5e0c7d70451620e6e713c07d5
bash "${task_root}/environment/materialize_project.sh" \
    "${source_repo}" "${project}" "${task_root}/environment/project-overlay" \
    "${bridge_commit}" "${mcore_commit}"
git -C "${project}" init -q
git -C "${project}" add --all
GIT_AUTHOR_DATE=2026-08-04T00:00:00Z GIT_COMMITTER_DATE=2026-08-04T00:00:00Z \
    git -C "${project}" -c user.name=Harbor -c user.email=harbor@localhost \
    commit -q -m 'Pinned learning-rate endpoint evaluation source'

export LR_SCHEDULE_SCALE=${scale}
# shellcheck source=/dev/null
source "${task_root}/environment/task-tools/lr_schedule_ladder_profile.sh"
source_hash=$(python3 "${task_root}/environment/task-tools/lr_schedule_task.py" \
    hash --project "${project}")
export LR_SCHEDULE_SOURCE_INVENTORY_SHA256=${source_hash}
export LR_SCHEDULE_CONTAINER_IMAGE=${image}
export CONTAINER_IMAGE=${image}
export EVAL_ONLY=1 EXPECT_FINAL_CHECKPOINT=0
export MARIN_LADDER_GPU_COUNT_OVERRIDE=${LSB_DJOB_NUMPROC}
export LSF_SLOTS=${MARIN_LADDER_GPU_COUNT_OVERRIDE}
export EXPECTED_NODES=$((LSF_SLOTS / GPUS_PER_NODE))
export RUN_DIR=${evaluation}
export WANDB_MODE=offline
export LR_SCHEDULE_WANDB_RUN_NAME=locked-wsd-${scale,,}-exact-final-eval
export WANDB_RUN_NAME=${LR_SCHEDULE_WANDB_RUN_NAME}
case "${scale}" in
    E4) export LOAD_DIR=$(dirname -- "${e4_checkpoint}") ;;
    E5) export LOAD_DIR=$(dirname -- "${e5_checkpoint}") ;;
esac
[[ $(tr -d '[:space:]' <"${LOAD_DIR}/latest_checkpointed_iteration.txt") == "${TRAIN_ITERS}" ]] || {
    printf 'error: checkpoint tracker does not match %s exact final update %s\n' \
        "${scale}" "${TRAIN_ITERS}" >&2
    exit 2
}

launcher=${project}/examples/training/lr_schedule/run_marin_adamh_bluevela_lsf.sh
bash "${launcher}" --run
result=${evaluation}/eval_harness/$(printf 'step_%08d' "${TRAIN_ITERS}")/results.json
[[ -s ${result} ]] || { printf 'error: exact-final Paloma result is missing: %s\n' "${result}" >&2; exit 1; }
printf 'exact-final %s result: %s\n' "${scale}" "${result}"
