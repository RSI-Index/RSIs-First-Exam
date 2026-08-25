#!/usr/bin/env bash
set -euo pipefail

: "${LSB_JOBID:?run this smoke test inside an LSF allocation}"
: "${LSB_MCPU_HOSTS:?LSF host allocation is missing}"
(( $# == 1 )) || { printf 'usage: smoke_blaunch_proxy_lsf.sh OUTPUT_DIR\n' >&2; exit 2; }

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
output_dir=$1
[[ ${output_dir} == /proj/* ]] || {
    printf 'error: OUTPUT_DIR must be an absolute /proj path\n' >&2
    exit 2
}
mkdir -p -- "${output_dir}"
proxy_root=${output_dir}/proxy
python3 "${script_dir}/blaunch_proxy.py" \
    --root "${proxy_root}" \
    --blaunch "$(command -v blaunch)" \
    >"${output_dir}/broker.log" 2>&1 &
broker_pid=$!
cleanup() {
    status=$?
    trap - EXIT INT TERM
    kill -TERM "${broker_pid}" 2>/dev/null || true
    wait "${broker_pid}" 2>/dev/null || true
    exit "${status}"
}
trap cleanup EXIT INT TERM

for _ in {1..100}; do
    [[ -s ${proxy_root}/READY.json ]] && break
    sleep 0.1
done
[[ -s ${proxy_root}/READY.json ]] || {
    printf 'error: broker did not become ready\n' >&2
    exit 1
}

host=$(hostname -s)
host_home=$HOME
env \
    BLAUNCH_PROXY_ROOT="${proxy_root}" \
    HOME=/app/home \
    APPTAINER_BIND=/outer-runtime:/inner-runtime \
    APPTAINER_WORKDIR=/app/apptainer-workdir \
    SINGULARITY_WORKDIR=/app/singularity-workdir \
    LSF_SLOTS=128 \
    'LSF_SPAN=span[ptile=8]' \
    LSF_WALLTIME=08:00 \
    LSF_MEMORY=65536 \
    LSF_QUEUE=do-not-forward \
    LSB_JOBID=do-not-forward \
    "${script_dir}/blaunch_proxy_client.py" -z "${host}" env \
    >"${output_dir}/remote.env"

grep -Fx 'LSF_SLOTS=128' "${output_dir}/remote.env"
grep -Fx 'LSF_SPAN=span[ptile=8]' "${output_dir}/remote.env"
grep -Fx 'LSF_WALLTIME=08:00' "${output_dir}/remote.env"
grep -Fx 'LSF_MEMORY=65536' "${output_dir}/remote.env"
grep -Fx "HOME=${host_home}" "${output_dir}/remote.env"
! grep -F 'APPTAINER_BIND=' "${output_dir}/remote.env"
! grep -F 'APPTAINER_WORKDIR=' "${output_dir}/remote.env"
! grep -F 'SINGULARITY_WORKDIR=' "${output_dir}/remote.env"
! grep -Fx 'LSF_QUEUE=do-not-forward' "${output_dir}/remote.env"
! grep -Fx 'LSB_JOBID=do-not-forward' "${output_dir}/remote.env"
grep -Eq '^LSB_JOBID=[0-9]+$' "${output_dir}/remote.env"

apptainer=/proj/datasets/interns/yuetai/rsi-nemotron/apptainer-env/bin/apptainer
image=/proj/datasets/interns/yuetai/agent_envs/more_task/images/parameterization-transfer-environment-accefd7b3a44-r2.sif
[[ -x ${apptainer} && -s ${image} ]]
env \
    BLAUNCH_PROXY_ROOT="${proxy_root}" \
    HOME=/app/home \
    APPTAINER_BIND=/outer-runtime:/inner-runtime \
    APPTAINER_WORKDIR=/app/apptainer-workdir \
    SINGULARITY_WORKDIR=/app/singularity-workdir \
    "${script_dir}/blaunch_proxy_client.py" -z "${host}" \
    "${apptainer}" exec --containall --writable-tmpfs \
    --bind /proj:/proj "${image}" /bin/echo nested-apptainer-creation-passed \
    >"${output_dir}/container.out"
grep -Fx 'nested-apptainer-creation-passed' "${output_dir}/container.out"
env \
    BLAUNCH_PROXY_ROOT="${proxy_root}" \
    HOME=/app/home \
    "${script_dir}/blaunch_proxy_client.py" -z "${host}" \
    "${apptainer}" exec --nv --containall --writable-tmpfs \
    --bind /proj:/proj "${image}" /opt/venv/bin/python -c \
    'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() >= 1; print(torch.cuda.get_device_name(0))' \
    >"${output_dir}/gpu.out"
grep -F 'H100' "${output_dir}/gpu.out"
env \
    BLAUNCH_PROXY_ROOT="${proxy_root}" \
    HOME=/app/home \
    SINGULARITY_WORKDIR=/app/singularity-workdir \
    "${script_dir}/blaunch_proxy_client.py" -z "${host}" \
    "${apptainer}" exec --containall --writable-tmpfs \
    --bind /proj:/proj "${image}" /bin/grep -F parameterization \
    /opt/project/examples/training/parameterization/run_marin_adamh_bluevela_lsf.sh \
    >"${output_dir}/launcher-path.out"
grep -F 'container_script="$container_repo_dir/examples/training/parameterization"' \
    "${output_dir}/launcher-path.out"
printf 'real LSF broker profile forwarding passed\n'
