#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: launch_harbor_lsf.sh [--dry-run|--submit] [MODEL]" >&2
}

mode=${1:---dry-run}
case ${mode} in
    --dry-run|--submit)
        (( $# <= 2 )) || { usage; exit 2; }
        model=${2:-codex/gpt-5.6-sol}
        ;;
    --run)
        (( $# == 4 )) || { usage; exit 2; }
        model=$2
        run_dir=$3
        task_snapshot=$4
        ;;
    *) usage; exit 2 ;;
esac

live_task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
repository_root=$(cd -- "${live_task_root}/../../.." && pwd -P)
asset_root=${TMAX_ASSET_ROOT:-/proj/datasets/interns/yuetai/agent_envs/scale_autoresearch_tasks/others/rsi_tasks}
harbor_root=${HARBOR_ROOT:-/u/yuetai/Scale_AutoResearch/envs}
runs_root=${asset_root}/full_runs/tmax_autoresearch

if [[ ${mode} != --run ]]; then
    run_id=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-tmax-ar}
    [[ ${run_id} =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ ]] || {
        echo "RUN_ID must be a safe identifier" >&2
        exit 2
    }
    run_dir=${runs_root}/${run_id}
    task_snapshot=${run_dir}/control/task
    bsub_args=(
        -q "${LSF_QUEUE:-priority}"
        -G "${LSF_GROUP:-grp_models}"
        -n 8
        -M "${LSF_MEMORY:-1048576}"
        -W "${LSF_WALLTIME:-72:00}"
        -R "${LSF_SPAN:-span[ptile=8]}"
        -gpu "num=8:mode=exclusive_process"
        -J "${run_id}"
        -o "${run_dir}/lsf.%J.out"
        -e "${run_dir}/lsf.%J.err"
    )
    submitted_command=("${task_snapshot}/cluster/launch_harbor_lsf.sh" --run "${model}" "${run_dir}" "${task_snapshot}")

    "${live_task_root}/cluster/preflight.sh" --allow-missing-images --allow-missing-baseline
    echo "run directory: ${run_dir}"
    echo "web search: disabled"
    printf 'LSF command: bsub'
    printf ' %q' "${bsub_args[@]}" "${submitted_command[@]}"
    printf '\nHarbor command: harbor run -p %q --env envs_backend.lsf_apptainer:LsfApptainerEnvironment --agent codex --model %q --ak reasoning_effort=xhigh --ak web_search=disabled --n-attempts 1\n' \
        "${task_snapshot}" "${model}"
    [[ ${mode} == --dry-run ]] && exit 0

    "${live_task_root}/cluster/preflight.sh"
    command -v bsub >/dev/null || { echo "bsub is unavailable" >&2; exit 2; }
    [[ ! -e ${run_dir} ]] || { echo "run already exists: ${run_dir}" >&2; exit 2; }
    mkdir -p -- "${task_snapshot}" "${run_dir}/harbor" "${run_dir}/control"
    rsync -a --exclude __pycache__ --exclude '*.pyc' "${live_task_root}/" "${task_snapshot}/"
    install -m 0555 -- "${repository_root}/runs/tmax_lsf/run_tmax_64.sh" "${run_dir}/control/run_tmax_64.sh"
    bsub "${bsub_args[@]}" "${submitted_command[@]}"
    exit 0
fi

[[ -n ${LSB_JOBID:-} ]] || { echo "--run must execute inside LSF" >&2; exit 2; }
[[ -d ${task_snapshot} && -f ${task_snapshot}/task.toml ]] || { echo "frozen task snapshot missing" >&2; exit 2; }
"${task_snapshot}/cluster/preflight.sh"

python3 - "${LSB_MCPU_HOSTS:-}" <<'PY'
import sys
fields = sys.argv[1].split()
if len(fields) != 16:
    raise SystemExit("expected exactly eight allocated hosts")
hosts = [fields[index].split(".", 1)[0] for index in range(0, len(fields), 2)]
if len(set(hosts)) != 8:
    raise SystemExit("expected eight unique allocated hosts")
print("outer allocation: 8 nodes x 8 H100 = 64 GPUs")
PY

candidate_output=${run_dir}/output
run_contract_dir=${run_dir}/control/run-contract
proxy_root=${run_dir}/control/blaunch-proxy
mkdir -m 0700 -- "${candidate_output}" "${run_contract_dir}" "${proxy_root}"
research_deadline_epoch=$(( $(date -u +%s) + 237600 ))
date -u -d "@${research_deadline_epoch}" +%FT%TZ >"${run_contract_dir}/RESEARCH_DEADLINE_UTC"
date -u +%FT%TZ >"${run_contract_dir}/RESEARCH_STARTED_UTC"

proxy_client=${run_contract_dir}/blaunch_proxy_client.py
install -m 0555 -- "${task_snapshot}/cluster/blaunch_proxy_client.py" "${proxy_client}"
trusted_runner=${task_snapshot}/environment/task-tools/run_tmax_attempt.sh
tmax_64gpu_runner=${run_dir}/control/run_tmax_64.sh
[[ -x ${trusted_runner} && -x ${tmax_64gpu_runner} ]] || { echo "trusted runner is missing" >&2; exit 3; }
lsf_blaunch=${BLAUNCH:-$(command -v blaunch || true)}
[[ -x ${lsf_blaunch} ]] || { echo "blaunch is unavailable" >&2; exit 3; }

export TMAX_64GPU_RUNNER=${tmax_64gpu_runner}
export APPTAINERENV_LSB_JOBID=${LSB_JOBID}
export APPTAINERENV_LSB_MCPU_HOSTS=${LSB_MCPU_HOSTS}
export APPTAINERENV_BLAUNCH=/run-contract/blaunch_proxy_client.py
export APPTAINERENV_BLAUNCH_PROXY_ROOT=/blaunch-proxy
export APPTAINER_BIND=/etc/resolv.conf

dataset_root=${asset_root}/assets/datasets/tmax-15k-open-instruct
bind_paths="${dataset_root}/data:/datasets/tmax/train:ro,${candidate_output}:/app/output:rw,${candidate_output}:${candidate_output}:rw,${run_contract_dir}:/run-contract:ro,${proxy_root}:/blaunch-proxy:rw"

if [[ -z ${OPENAI_API_KEY:-} ]]; then
    export CODEX_FORCE_AUTH_JSON=${CODEX_FORCE_AUTH_JSON:-1}
    [[ -s ${HOME}/.codex/auth.json ]] || { echo "Codex auth is unavailable" >&2; exit 3; }
fi

proxy_pid=
cleanup() {
    status=$?
    trap - EXIT INT TERM
    if [[ -n ${proxy_pid:-} ]]; then
        kill -TERM "${proxy_pid}" 2>/dev/null || true
        wait "${proxy_pid}" 2>/dev/null || true
    fi
    exit "${status}"
}
trap cleanup EXIT INT TERM
python3 "${task_snapshot}/cluster/blaunch_proxy.py" \
    --root "${proxy_root}" \
    --blaunch "${lsf_blaunch}" \
    --runner "${trusted_runner}" \
    --full-run "${candidate_output}" \
    >"${run_dir}/control/blaunch-proxy.log" 2>&1 &
proxy_pid=$!
for _ in $(seq 1 100); do
    [[ -s ${proxy_root}/READY.json ]] && break
    kill -0 "${proxy_pid}" 2>/dev/null || { echo "blaunch proxy exited during startup" >&2; exit 3; }
    sleep 0.1
done
[[ -s ${proxy_root}/READY.json ]] || { echo "blaunch proxy did not become ready" >&2; exit 3; }

cd -- "${harbor_root}"
set +e
.venv/bin/harbor run \
    -p "${task_snapshot}" \
    --env envs_backend.lsf_apptainer:LsfApptainerEnvironment \
    --ek "bind_paths=${bind_paths}" \
    --ek agent_exec_timeout_sec=244800 \
    --agent codex \
    --model "${model}" \
    --ak reasoning_effort=xhigh \
    --ak web_search=disabled \
    --n-attempts 1 \
    --job-name "$(basename -- "${run_dir}")" \
    --jobs-dir "${run_dir}/harbor" \
    --yes
harbor_status=$?
set -e
exit "${harbor_status}"
