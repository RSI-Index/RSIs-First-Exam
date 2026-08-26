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
Build the parameterization scaling-ladder Harbor environment and verifier SIFs.

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

asset_root=${PARAMETERIZATION_ASSET_ROOT:-/proj/datasets/interns/yuetai/agent_envs/more_task}
source_repo=${MBRIDGE_REPO:-${task_root}/../../../pre_training/Megatron-Bridge}
apptainer=${APPTAINER:-/proj/datasets/interns/yuetai/rsi-nemotron/apptainer-env/bin/apptainer}
base_runtime_sif=${PARAMETERIZATION_BASE_RUNTIME_SIF:-${asset_root}/images/megatron-bridge-llama14b-accefd7b3a44-dbd0f1abbad5.sif}
environment_sif=${PARAMETERIZATION_ENVIRONMENT_SIF:-${asset_root}/images/parameterization-transfer-environment-accefd7b3a44-r2.sif}
tests_sif=${PARAMETERIZATION_TESTS_SIF:-${asset_root}/images/parameterization-transfer-tests-accefd7b3a44-r2.sif}
logs_dir=${asset_root}/logs/parameterization-transfer-image-build
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
[[ ${source_sha} == accefd7b3a448ab45b015edc04c9d3b70f7cb3b7 ]] || {
    printf 'error: unexpected Bridge commit: %s\n' "${source_sha}" >&2
    exit 2
}
[[ ${mcore_sha} == 58bf14e9e68915a5e0c7d70451620e6e713c07d5 ]] || {
    printf 'error: unexpected MCore commit: %s\n' "${mcore_sha}" >&2
    exit 2
}
grep -Fx "source_commit=${source_sha}" "${base_runtime_sif}.build-info" >/dev/null || {
    printf 'error: base runtime Bridge commit does not match %s\n' "${source_sha}" >&2
    exit 2
}
grep -Fx "mcore_commit=${mcore_sha}" "${base_runtime_sif}.build-info" >/dev/null || {
    printf 'error: base runtime MCore commit does not match %s\n' "${mcore_sha}" >&2
    exit 2
}

bsub_command=(
    bsub -q "${lsf_queue}" -G "${lsf_group}" -n "${lsf_cpus}"
    -M "${lsf_memory}" -W "${lsf_walltime}" -R 'span[hosts=1]'
    -J mt-param-xfer-sif-r2-yuetai
    -o "${logs_dir}/parameterization-transfer-sif.%J.out"
    -e "${logs_dir}/parameterization-transfer-sif.%J.err"
    bash "${script_path}" --run
)

if [[ ${mode} == --dry-run ]]; then
    printf 'mode: dry-run (no job submitted)\n'
    printf 'task root:       %s\n' "${task_root}"
    printf 'Bridge commit:   %s\n' "${source_sha}"
    printf 'MCore commit:    %s\n' "${mcore_sha}"
    printf 'project scope:   current Bridge working tree; training examples limited to parameterization\n'
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
    export PARAMETERIZATION_ASSET_ROOT=${asset_root} MBRIDGE_REPO=${source_repo} APPTAINER=${apptainer}
    export PARAMETERIZATION_BASE_RUNTIME_SIF=${base_runtime_sif}
    export PARAMETERIZATION_ENVIRONMENT_SIF=${environment_sif} PARAMETERIZATION_TESTS_SIF=${tests_sif}
    export APPTAINER_CACHEDIR=${cache_dir} LSF_GROUP=${lsf_group}
    "${bsub_command[@]}"
    exit 0
fi

[[ -n ${LSB_JOBID:-} ]] || { printf 'error: --run must execute inside LSF\n' >&2; exit 2; }
[[ ! -e ${tests_sif} ]] || { printf 'error: tests SIF already exists: %s\n' "${tests_sif}" >&2; exit 2; }
build_dir=$(mktemp -d "/tmp/parameterization-sif-${LSB_JOBID}.XXXXXXXX")
cleanup() {
    if [[ -n ${build_dir:-} && ${build_dir} == /tmp/parameterization-sif-${LSB_JOBID}.* ]]; then
        rm -rf -- "${build_dir}"
    fi
}
trap cleanup EXIT

context=${build_dir}/environment-context
mkdir -p -- "${context}/task-tools"
bash "${task_root}/environment/materialize_project.sh" \
    "${source_repo}" \
    "${context}/parameterization-project" \
    "${task_root}/environment/project-overlay" \
    "${source_sha}" \
    "${mcore_sha}"
cp -a -- "${task_root}/environment/task-tools/." "${context}/task-tools/"
sed "s|@BASE_RUNTIME_SIF@|${base_runtime_sif}|" \
    "${task_root}/environment/parameterization_transfer.def" \
    >"${context}/parameterization_transfer.def"
git -C "${context}/parameterization-project" init
git -C "${context}/parameterization-project" add --all
GIT_AUTHOR_DATE=2026-07-30T00:00:00Z GIT_COMMITTER_DATE=2026-07-30T00:00:00Z \
    git -C "${context}/parameterization-project" -c user.name=Harbor -c user.email=harbor@localhost \
    commit -m 'Pinned parameterization-encoding task baseline'

export APPTAINER_BIND=/etc/resolv.conf
export APPTAINER_CACHEDIR=${cache_dir}
export APPTAINER_TMPDIR=${build_dir}/apptainer-tmp
mkdir -p -- "${APPTAINER_TMPDIR}"

if [[ ! -e ${environment_sif} ]]; then
    local_environment=${build_dir}/environment.sif
    (cd "${context}" && "${apptainer}" build --fakeroot \
        --mksquashfs-args "-processors ${lsf_cpus}" \
        "${local_environment}" parameterization_transfer.def)
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
    "${task_root}/tests/parameterization_transfer_tests.def" \
    >"${tests_context}/parameterization_transfer_tests.def"
local_tests=${build_dir}/tests.sif
(cd "${tests_context}" && "${apptainer}" build --fakeroot \
    --mksquashfs-args "-processors ${lsf_cpus}" \
    "${local_tests}" parameterization_transfer_tests.def)
tests_part=${tests_sif}.part.${LSB_JOBID}
/usr/bin/dd if="${local_tests}" of="${tests_part}" bs=16M status=none conv=fsync
[[ ! -e ${tests_sif} ]] || { printf 'error: tests SIF appeared during build\n' >&2; exit 2; }
/usr/bin/mv -- "${tests_part}" "${tests_sif}"
sha256sum "${tests_sif}" >"${tests_sif}.sha256"
printf 'built environment: %s\n' "${environment_sif}"
printf 'built verifier:    %s\n' "${tests_sif}"
