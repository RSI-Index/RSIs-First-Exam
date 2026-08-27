#!/usr/bin/env bash
set -euo pipefail

task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
script_path="${task_root}/cluster/build_images.sh"
bluevela_env_file=${BLUEVELA_ENV_FILE:-${XDG_CONFIG_HOME:-${HOME:-}/.config}/megatron-bridge/bluevela.env}
if [[ -r ${bluevela_env_file} ]]; then
    # shellcheck source=/dev/null
    source "${bluevela_env_file}"
fi

usage() {
    cat <<'EOF'
Build the learning-rate Harbor environment and verifier SIFs.

Usage: build_images.sh [--dry-run|--submit|--run]

--dry-run is the default and performs no writes or submission.
--submit creates build/log directories and submits one CPU-only LSF job.
--run is the internal LSF entry point.
EOF
}

mode=${1:---dry-run}
(( $# <= 1 )) || { usage >&2; exit 2; }
case "${mode}" in
    --dry-run|--submit|--run) ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

asset_root=${LR_SCHEDULE_ASSET_ROOT:-/proj/datasets/interns/yuetai/agent_envs/more_task}
source_repo=${MBRIDGE_REPO:-${task_root}/../../../../pre_training/Megatron-Bridge}
apptainer=${APPTAINER:-/proj/datasets/interns/yuetai/rsi-nemotron/apptainer-env/bin/apptainer}
base_runtime_sif=${LR_SCHEDULE_BASE_RUNTIME_SIF:-${asset_root}/images/megatron-bridge-llama14b-accefd7b3a44-dbd0f1abbad5.sif}
environment_sif=${LR_SCHEDULE_ENVIRONMENT_SIF:-${asset_root}/images/learning-rate-environment-accefd7b3a44-r1.sif}
tests_sif=${LR_SCHEDULE_TESTS_SIF:-${asset_root}/images/learning-rate-tests-accefd7b3a44-r1.sif}
logs_dir=${asset_root}/logs/learning-rate-image-build
cache_dir=${APPTAINER_CACHEDIR:-${asset_root}/cache/apptainer}
lsf_queue=${LSF_QUEUE:-normal}
lsf_group=${LSF_GROUP:-}
lsf_cpus=${LSF_BUILD_CPUS:-32}
lsf_memory=${LSF_BUILD_MEMORY:-256G}
lsf_walltime=${LSF_BUILD_WALLTIME:-10:00}

[[ -d ${source_repo} && -f ${source_repo}/pyproject.toml ]] || {
    printf 'error: Megatron-Bridge checkout is missing: %s\n' "${source_repo}" >&2
    exit 2
}
[[ -f ${base_runtime_sif} ]] || {
    printf 'error: prepared base runtime SIF is missing: %s\n' "${base_runtime_sif}" >&2
    exit 2
}
[[ -f ${base_runtime_sif}.sha256 && -f ${base_runtime_sif}.build-info ]] || {
    printf 'error: prepared base runtime provenance is incomplete: %s\n' "${base_runtime_sif}" >&2
    exit 2
}
sha256sum -c "${base_runtime_sif}.sha256" >/dev/null
source_sha=$(git -C "${source_repo}" rev-parse HEAD)
mcore_sha=$(git -C "${source_repo}/3rdparty/Megatron-LM" rev-parse HEAD)
[[ ${source_sha} == 3a6c6fb3811a10f532d26be7dd032457795a8e41 ]] || {
    printf 'error: unexpected Bridge commit: %s\n' "${source_sha}" >&2
    exit 2
}
[[ ${mcore_sha} == 58bf14e9e68915a5e0c7d70451620e6e713c07d5 ]] || {
    printf 'error: unexpected MCore commit: %s\n' "${mcore_sha}" >&2
    exit 2
}
grep -Fx "source_commit=accefd7b3a448ab45b015edc04c9d3b70f7cb3b7" \
    "${base_runtime_sif}.build-info" >/dev/null || {
    printf 'error: base runtime does not match the locked upstream Bridge commit\n' >&2
    exit 2
}
grep -Fx "mcore_commit=${mcore_sha}" "${base_runtime_sif}.build-info" >/dev/null || {
    printf 'error: base runtime MCore commit does not match %s\n' "${mcore_sha}" >&2
    exit 2
}

bsub_command=(
    bsub -q "${lsf_queue}" -G "${lsf_group}" -n "${lsf_cpus}"
    -M "${lsf_memory}" -W "${lsf_walltime}" -R 'span[hosts=1]'
    -J mt-learning-rate-sif-yuetai
    -o "${logs_dir}/lr-schedule-sif.%J.out"
    -e "${logs_dir}/lr-schedule-sif.%J.err"
    bash "${script_path}" --run
)

if [[ ${mode} == --dry-run ]]; then
    printf 'mode: dry-run (no job submitted)\n'
    printf 'task root:       %s\n' "${task_root}"
    printf 'Bridge commit:   %s\n' "${source_sha}"
    printf 'MCore commit:    %s\n' "${mcore_sha}"
    printf 'project scope:   current Bridge working tree; training examples limited to lr_schedule\n'
    printf 'base runtime:    %s\n' "${base_runtime_sif}"
    printf 'environment SIF: %s\n' "${environment_sif}"
    printf 'tests SIF:       %s\n' "${tests_sif}"
    printf 'LSF command:'
    printf ' %q' "${bsub_command[@]}"
    printf '\n'
    exit 0
fi

[[ -n ${lsf_group} ]] || { printf 'error: LSF_GROUP is required\n' >&2; exit 2; }
[[ -x ${apptainer} ]] || { printf 'error: Apptainer is not executable: %s\n' "${apptainer}" >&2; exit 2; }

if [[ ${mode} == --submit ]]; then
    command -v bsub >/dev/null || { printf 'error: bsub is unavailable\n' >&2; exit 2; }
    [[ ! -e ${tests_sif} ]] || { printf 'error: tests SIF already exists: %s\n' "${tests_sif}" >&2; exit 2; }
    mkdir -p -- "${logs_dir}" "${cache_dir}" "$(dirname -- "${environment_sif}")"
    export LR_SCHEDULE_ASSET_ROOT=${asset_root} MBRIDGE_REPO=${source_repo} APPTAINER=${apptainer}
    export LR_SCHEDULE_BASE_RUNTIME_SIF=${base_runtime_sif}
    export LR_SCHEDULE_ENVIRONMENT_SIF=${environment_sif} LR_SCHEDULE_TESTS_SIF=${tests_sif}
    export APPTAINER_CACHEDIR=${cache_dir} LSF_GROUP=${lsf_group}
    "${bsub_command[@]}"
    exit 0
fi

[[ -n ${LSB_JOBID:-} ]] || { printf 'error: --run must execute inside LSF\n' >&2; exit 2; }
[[ ! -e ${tests_sif} ]] || { printf 'error: tests SIF already exists: %s\n' "${tests_sif}" >&2; exit 2; }
build_dir=$(mktemp -d "/tmp/lr-schedule-sif-${LSB_JOBID}.XXXXXXXX")
cleanup() {
    if [[ -n ${build_dir:-} && ${build_dir} == /tmp/lr-schedule-sif-${LSB_JOBID}.* ]]; then
        rm -rf -- "${build_dir}"
    fi
}
trap cleanup EXIT

context=${build_dir}/environment-context
mkdir -p -- "${context}/task-tools"
bash "${task_root}/environment/materialize_project.sh" \
    "${source_repo}" \
    "${context}/lr-schedule-project" \
    "${task_root}/environment/project-overlay" \
    "${source_sha}" \
    "${mcore_sha}"
cp -a -- "${task_root}/environment/task-tools/." "${context}/task-tools/"
sed "s|@BASE_RUNTIME_SIF@|${base_runtime_sif}|" \
    "${task_root}/environment/learning_rate.def" \
    >"${context}/learning_rate.def"
git -C "${context}/lr-schedule-project" init
git -C "${context}/lr-schedule-project" add --all
GIT_AUTHOR_DATE=2026-07-30T00:00:00Z GIT_COMMITTER_DATE=2026-07-30T00:00:00Z \
    git -C "${context}/lr-schedule-project" -c user.name=Harbor -c user.email=harbor@localhost \
    commit -m 'Pinned learning-rate task baseline'

export APPTAINER_BIND=/etc/resolv.conf
export APPTAINER_CACHEDIR=${cache_dir}
export APPTAINER_TMPDIR=${build_dir}/apptainer-tmp
mkdir -p -- "${APPTAINER_TMPDIR}"

if [[ ! -e ${environment_sif} ]]; then
    local_environment=${build_dir}/environment.sif
    (cd "${context}" && "${apptainer}" build --fakeroot \
        --mksquashfs-args "-processors ${lsf_cpus}" \
        "${local_environment}" learning_rate.def)
    environment_part=${environment_sif}.part.${LSB_JOBID}
    /usr/bin/dd if="${local_environment}" of="${environment_part}" bs=16M status=none conv=fsync
    [[ ! -e ${environment_sif} ]] || { printf 'error: environment SIF appeared during build\n' >&2; exit 2; }
    /usr/bin/mv -- "${environment_part}" "${environment_sif}"
    sha256sum "${environment_sif}" >"${environment_sif}.sha256"
else
    printf 'reusing existing environment SIF: %s\n' "${environment_sif}"
fi

tests_context=${build_dir}/tests-context
mkdir -p -- "${tests_context}/tests" "${tests_context}/task-tools"
cp -a -- "${task_root}/tests/." "${tests_context}/tests/"
cp -a -- "${task_root}/environment/task-tools/." "${tests_context}/task-tools/"
sed "s|@ENVIRONMENT_SIF@|${environment_sif}|" \
    "${task_root}/tests/learning_rate_tests.def" \
    >"${tests_context}/learning_rate_tests.def"
local_tests=${build_dir}/tests.sif
(cd "${tests_context}" && "${apptainer}" build --fakeroot \
    --mksquashfs-args "-processors ${lsf_cpus}" \
    "${local_tests}" learning_rate_tests.def)
tests_part=${tests_sif}.part.${LSB_JOBID}
/usr/bin/dd if="${local_tests}" of="${tests_part}" bs=16M status=none conv=fsync
[[ ! -e ${tests_sif} ]] || { printf 'error: tests SIF appeared during build\n' >&2; exit 2; }
/usr/bin/mv -- "${tests_part}" "${tests_sif}"
sha256sum "${tests_sif}" >"${tests_sif}.sha256"
printf 'built environment: %s\n' "${environment_sif}"
printf 'built verifier:    %s\n' "${tests_sif}"
