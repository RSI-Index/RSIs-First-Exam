#!/usr/bin/env bash
set -euo pipefail

mode=${1:---dry-run}
case ${mode} in --dry-run|--submit|--run) ;; *) echo "usage: build_images.sh [--dry-run|--submit|--run]" >&2; exit 2 ;; esac
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
asset_root=${TMAX_ASSET_ROOT:-/proj/datasets/interns/yuetai/agent_envs/scale_autoresearch_tasks/others/rsi_tasks}
apptainer=${APPTAINER:-/proj/datasets/interns/yuetai/rsi-nemotron/apptainer-env/bin/apptainer}
base_sif=${asset_root}/assets/images/python-3.12-slim.sif
environment_sif=${asset_root}/assets/images/tmax-autoresearch-environment-7387d2f-r1.sif
tests_sif=${asset_root}/assets/images/tmax-autoresearch-tests-7387d2f-r1.sif
logs_dir=${asset_root}/logs/tmax-autoresearch-image-build
script_path=${task_root}/cluster/build_images.sh

bsub_command=(
    bsub -q "${LSF_QUEUE:-normal}" -G "${LSF_GROUP:-grp_models}" -n 32 -M "${LSF_BUILD_MEMORY:-262144}"
    -W "${LSF_BUILD_WALLTIME:-08:00}" -R 'span[hosts=1]'
    -J tmax-ar-images -o "${logs_dir}/build.%J.out" -e "${logs_dir}/build.%J.err"
    bash "${script_path}" --run
)
if [[ ${mode} == --dry-run ]]; then
    printf 'LSF command:'; printf ' %q' "${bsub_command[@]}"; printf '\n'
    printf 'environment image: %s\ntests image: %s\n' "${environment_sif}" "${tests_sif}"
    exit 0
fi
[[ -x ${apptainer} && -f ${base_sif} ]] || { echo "Apptainer or Python base SIF missing" >&2; exit 2; }
if [[ ${mode} == --submit ]]; then
    command -v bsub >/dev/null || { echo "bsub is unavailable" >&2; exit 2; }
    mkdir -p -- "${logs_dir}"
    "${bsub_command[@]}"
    exit 0
fi
[[ -n ${LSB_JOBID:-} ]] || { echo "--run must execute inside LSF" >&2; exit 2; }

build_dir=$(mktemp -d "/tmp/tmax-ar-images-${LSB_JOBID}.XXXXXXXX")
cleanup() {
    [[ ${build_dir:-} == /tmp/tmax-ar-images-${LSB_JOBID}.* ]] && rm -rf -- "${build_dir}"
}
trap cleanup EXIT

context=${build_dir}/environment
mkdir -p -- "${context}/project" "${context}/task-tools" "${context}/training-defaults" "${context}/training-data"
rsync -a --exclude .git --exclude .venv --exclude evaluation_assets --exclude __pycache__ --exclude '*.pyc' \
    "${task_root}/environment/project/" "${context}/project/"
rsync -a --exclude __pycache__ --exclude '*.pyc' "${task_root}/environment/task-tools/" "${context}/task-tools/"
rsync -a "${task_root}/environment/training-defaults/" "${context}/training-defaults/"
rsync -a "${asset_root}/assets/datasets/tmax-15k-open-instruct/data/" "${context}/training-data/"
sed "s|@PYTHON_BASE_SIF@|${base_sif}|" "${task_root}/environment/tmax_autoresearch.def" >"${context}/tmax_autoresearch.def"

export APPTAINER_BIND=/etc/resolv.conf
export APPTAINER_CACHEDIR=${asset_root}/apptainer-cache
export APPTAINER_TMPDIR=${build_dir}/tmp
mkdir -p -- "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"
if [[ ! -e ${environment_sif} ]]; then
    (cd -- "${context}" && "${apptainer}" build --fakeroot --mksquashfs-args "-processors 32" \
        "${build_dir}/environment.sif" tmax_autoresearch.def)
    install -m 0555 -- "${build_dir}/environment.sif" "${environment_sif}.part.${LSB_JOBID}"
    mv -- "${environment_sif}.part.${LSB_JOBID}" "${environment_sif}"
    sha256sum "${environment_sif}" >"${environment_sif}.sha256"
fi

tests_context=${build_dir}/tests
mkdir -p -- "${tests_context}/tests"
rsync -a --exclude __pycache__ --exclude '*.pyc' "${task_root}/tests/" "${tests_context}/tests/"
sed "s|@ENVIRONMENT_SIF@|${environment_sif}|" "${task_root}/tests/tmax_autoresearch_tests.def" >"${tests_context}/tests.def"
(cd -- "${tests_context}" && "${apptainer}" build --fakeroot --mksquashfs-args "-processors 32" \
    "${build_dir}/tests.sif" tests.def)
install -m 0555 -- "${build_dir}/tests.sif" "${tests_sif}.part.${LSB_JOBID}"
mv -- "${tests_sif}.part.${LSB_JOBID}" "${tests_sif}"
sha256sum "${tests_sif}" >"${tests_sif}.sha256"
echo "built ${environment_sif}"
echo "built ${tests_sif}"
