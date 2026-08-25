#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Launch one parameterization scaling-ladder Harbor trial in a Blue Vela allocation.

Usage:
  launch_harbor_lsf.sh [--dry-run|--submit] [HARBOR_MODEL]
  launch_harbor_lsf.sh --resume RUN_DIR [HARBOR_MODEL]

The default is --dry-run. --run and --run-resume are reserved for submitted
LSF jobs. --resume reuses one interrupted run's durable output and deadline.
EOF
}

mode=${1:---dry-run}
resume_existing=0
case "${mode}" in
    --dry-run|--submit)
        (( $# <= 2 )) || { usage >&2; exit 2; }
        model=${2:-codex/gpt-5.6-sol}
        ;;
    --run)
        (( $# == 4 )) || { printf 'error: internal --run requires MODEL RUN_DIR TASK_SNAPSHOT\n' >&2; exit 2; }
        model=$2
        run_dir=$3
        task_snapshot=$4
        ;;
    --resume)
        (( $# == 2 || $# == 3 )) || { printf 'error: --resume requires RUN_DIR [HARBOR_MODEL]\n' >&2; exit 2; }
        run_dir=$2
        model=${3:-codex/gpt-5.6-sol}
        task_snapshot=${run_dir}/control/task
        resume_existing=1
        ;;
    --run-resume)
        (( $# == 4 )) || { printf 'error: internal --run-resume requires MODEL RUN_DIR TASK_SNAPSHOT\n' >&2; exit 2; }
        model=$2
        run_dir=$3
        task_snapshot=$4
        resume_existing=1
        ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

live_task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
asset_root=/proj/datasets/interns/yuetai/agent_envs/more_task
harbor_root=/u/yuetai/Scale_AutoResearch/envs
bluevela_env_file=${BLUEVELA_ENV_FILE:-${XDG_CONFIG_HOME:-${HOME:-}/.config}/megatron-bridge/bluevela.env}
if [[ -r ${bluevela_env_file} ]]; then
    # shellcheck source=/dev/null
    source "${bluevela_env_file}"
fi

# Keep the Harbor control server and training leases off nodes with observed
# hard failures in this run: two rendezvous/transport failures and one local
# FastAPI readiness hang. LSF_SPAN remains an escape hatch for administrators.
default_lsf_resource="select[hname!='p1-r12-n3' && hname!='p2-r19-n3' && hname!='p3-r04-n2' && hname!='p3-r05-n1' && hname!='p4-r11-n2' && hname!='p4-r27-n1'] span[ptile=8]"
if [[ ${mode} == --resume ]]; then
    expected_runs_root=${asset_root}/full_runs/parameterization_transfer
    run_dir=$(realpath -e -- "${run_dir}")
    [[ ${run_dir} == "${expected_runs_root}/"* ]] || {
        printf 'error: resume run must be below %s: %s\n' "${expected_runs_root}" "${run_dir}" >&2
        exit 2
    }
    task_snapshot=${run_dir}/control/task
    [[ -d ${run_dir}/candidate-output && -d ${run_dir}/control/run-contract ]] || {
        printf 'error: interrupted run has no durable candidate output/contract: %s\n' "${run_dir}" >&2
        exit 2
    }
    [[ -x ${task_snapshot}/cluster/launch_harbor_lsf.sh ]] || {
        printf 'error: interrupted run has no frozen launcher: %s\n' "${task_snapshot}" >&2
        exit 2
    }
    deadline_file=${run_dir}/control/run-contract/RESEARCH_DEADLINE_UTC
    [[ -s ${deadline_file} ]] || { printf 'error: interrupted run deadline is missing\n' >&2; exit 2; }
    deadline_epoch=$(date -u -d "$(tr -d '[:space:]' <"${deadline_file}")" +%s)
    (( deadline_epoch > $(date -u +%s) )) || {
        printf 'error: interrupted run deadline has already passed\n' >&2
        exit 2
    }

    bsub_args=(
        -q "${LSF_QUEUE:-priority}"
        -G "${LSF_GROUP:-grp_models}"
        # A complete E0-E5 ladder uses exactly 27 nodes / 216 GPUs.  Reserve
        # one LSF CPU slot plus all eight GPUs per node; nested workers receive
        # a separate logical eight-slot inventory below.  Requiring eight LSF
        # CPU slots here needlessly makes GPU-idle nodes ineligible.
        -n 256
        # A resumed allocation reserves one LSF slot per node, but that slot
        # supervises eight Python/GPU workers.  LSF applies -M to the scheduler
        # task, so the old 32 GiB limit killed the whole allocation as soon as
        # two one-node screens initialized (~38 GiB aggregate RSS).  Size the
        # per-node supervisor for the eight-worker peak instead.
        -M "${LSF_MEMORY:-32768}"
        -W "${LSF_WALLTIME:-72:00}"
        -R "${LSF_SPAN:-${default_lsf_resource}}"
        -gpu 'num=8:mode=exclusive_process'
        -J mt-parameterization-transfer-resume-yuetai
        -o "${run_dir}/lsf.resume.%J.out"
        -e "${run_dir}/lsf.resume.%J.err"
    )
    submitted_launcher=${task_snapshot}/cluster/launch_harbor_lsf.sh
    submitted_command=("${submitted_launcher}" --run-resume "${model}" "${run_dir}" "${task_snapshot}")
    "${task_snapshot}/cluster/preflight.sh"
    command -v bsub >/dev/null || { printf 'error: bsub is unavailable\n' >&2; exit 2; }
    printf 'resuming run directory: %s\n' "${run_dir}"
    printf 'preserved deadline: %s\n' "$(tr -d '[:space:]' <"${deadline_file}")"
    printf 'LSF command: bsub'
    printf ' %q' "${bsub_args[@]}" "${submitted_command[@]}"
    printf '\n'
    bsub "${bsub_args[@]}" "${submitted_command[@]}"
    exit 0
fi

if [[ ${mode} != --run && ${mode} != --run-resume ]]; then
    run_id=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-parameterization-transfer-ladder}
    [[ ${run_id} =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ ]] || {
        printf 'error: RUN_ID must be a safe identifier\n' >&2
        exit 2
    }
    run_dir=${asset_root}/full_runs/parameterization_transfer/${run_id}
    task_snapshot=${run_dir}/control/task
    [[ ! -e ${run_dir} ]] || { printf 'error: run already exists: %s\n' "${run_dir}" >&2; exit 2; }

    bsub_args=(
        -q "${LSF_QUEUE:-priority}"
        -G "${LSF_GROUP:-grp_models}"
        -n 256
        -M "${LSF_MEMORY:-32768}"
        -W "${LSF_WALLTIME:-72:00}"
        -R "${LSF_SPAN:-${default_lsf_resource}}"
        # Blue Vela records H100s with a cluster-specific full model name.
        # Match the baseline launcher and avoid a fragile gmodel filter.
        -gpu 'num=8:mode=exclusive_process'
        -J mt-parameterization-transfer-yuetai
        -o "${run_dir}/lsf.%J.out"
        -e "${run_dir}/lsf.%J.err"
    )
    submitted_launcher=${task_snapshot}/cluster/launch_harbor_lsf.sh
    submitted_command=("${submitted_launcher}" --run "${model}" "${run_dir}" "${task_snapshot}")

    "${live_task_root}/cluster/preflight.sh" --allow-missing-images --allow-missing-assets
    printf 'run directory: %s\n' "${run_dir}"
    printf 'model: %s\n' "${model}"
    printf 'web search: disabled\n'
    printf 'LSF command: bsub'
    printf ' %q' "${bsub_args[@]}" "${submitted_command[@]}"
    printf '\n'
    if [[ ${mode} == --dry-run ]]; then
        exit 0
    fi

    "${live_task_root}/cluster/preflight.sh"
    command -v bsub >/dev/null || { printf 'error: bsub is unavailable\n' >&2; exit 2; }
    mkdir -p -- "${task_snapshot}" "${run_dir}/harbor"
    rsync -a --exclude '__pycache__' --exclude '*.pyc' "${live_task_root}/" "${task_snapshot}/"
    bsub "${bsub_args[@]}" "${submitted_command[@]}"
    exit 0
fi

[[ -n ${LSB_JOBID:-} ]] || { printf 'error: --run must execute inside LSF\n' >&2; exit 2; }
[[ -d ${task_snapshot} && -f ${task_snapshot}/task.toml ]] || {
    printf 'error: frozen task snapshot is missing: %s\n' "${task_snapshot}" >&2
    exit 2
}
"${task_snapshot}/cluster/preflight.sh"

if [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then
    IFS=',' read -r -a visible_gpu_ids <<<"${CUDA_VISIBLE_DEVICES}"
    visible_gpus=${#visible_gpu_ids[@]}
else
    visible_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
fi
(( visible_gpus == 8 )) || { printf 'error: expected 8 visible GPUs, found %s\n' "${visible_gpus}" >&2; exit 3; }

python3 "${task_snapshot}/cluster/validate_outer_allocation.py" "${LSB_MCPU_HOSTS:-}"

mapfile -t gpu_inventory < <(
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
)
(( ${#gpu_inventory[@]} == 8 )) || {
    printf 'error: expected inventory for 8 GPUs, found %s\n' "${#gpu_inventory[@]}" >&2
    exit 3
}
for gpu_row in "${gpu_inventory[@]}"; do
    gpu_name=${gpu_row%%,*}
    gpu_memory=${gpu_row##*,}
    gpu_name=${gpu_name#"${gpu_name%%[![:space:]]*}"}
    gpu_memory=${gpu_memory//[[:space:]]/}
    [[ ${gpu_memory} =~ ^[0-9]+$ ]] && (( gpu_memory >= 79000 )) || {
        printf 'error: expected an 80GB-class GPU, found %s (%s MiB)\n' "${gpu_name}" "${gpu_memory}" >&2
        exit 3
    }
done

export PATH="/proj/datasets/interns/yuetai/rsi-nemotron/apptainer-env/bin:${PATH}"
export PYTHONPATH="${harbor_root}:${PYTHONPATH:-}"
export APPTAINER_BIND=/etc/resolv.conf
if [[ -z ${OPENAI_API_KEY:-} ]]; then
    export CODEX_FORCE_AUTH_JSON=${CODEX_FORCE_AUTH_JSON:-1}
    [[ -s ${HOME}/.codex/auth.json ]] || {
        printf 'error: provide OPENAI_API_KEY or %s/.codex/auth.json\n' "${HOME}" >&2
        exit 3
    }
fi

# The project comes from the immutable environment image and Harbor gives each
# attempt its own writable /app/project. Only prepared data and caches are
# reused from the host; the live Megatron-Bridge checkout is never mounted.
# Candidate output is bound directly to this unique run's GPFS directory so
# checkpoints and the multi-candidate ledger survive Harbor cleanup or timeout.
candidate_output=${run_dir}/candidate-output
run_contract_dir=${run_dir}/control/run-contract
if (( resume_existing == 1 )); then
    [[ -d ${candidate_output} && -d ${run_contract_dir} ]] || {
        printf 'error: resume output/contract is missing below %s\n' "${run_dir}" >&2
        exit 3
    }
    [[ -s ${run_contract_dir}/RESEARCH_DEADLINE_UTC ]] || {
        printf 'error: resume deadline is missing\n' >&2
        exit 3
    }
    research_deadline_epoch=$(date -u -d "$(tr -d '[:space:]' <"${run_contract_dir}/RESEARCH_DEADLINE_UTC")" +%s)
    (( research_deadline_epoch > $(date -u +%s) )) || {
        printf 'error: resume deadline has already passed\n' >&2
        exit 3
    }
    lease_state=${candidate_output}/host-leases.json
    if [[ -s ${lease_state} ]]; then
        cp -a -- "${lease_state}" "${candidate_output}/host-leases.pre-resume-${LSB_JOBID}.json"
    fi
    lease_tmp=${candidate_output}/.host-leases.resume-${LSB_JOBID}.tmp
    printf '{\n  "leases": {},\n  "version": 1\n}\n' >"${lease_tmp}"
    mv -- "${lease_tmp}" "${lease_state}"
    date -u +%FT%TZ >"${run_contract_dir}/RESUMED_${LSB_JOBID}_UTC"
else
    [[ ! -e ${candidate_output} ]] || {
        printf 'error: candidate output already exists for this unique run: %s\n' "${candidate_output}" >&2
        exit 3
    }
    mkdir -- "${candidate_output}"
    [[ ! -e ${run_contract_dir} ]] || {
        printf 'error: trusted run contract already exists: %s\n' "${run_contract_dir}" >&2
        exit 3
    }
    mkdir -- "${run_contract_dir}"
    # End scientific search at 66h. The agent execution envelope is 68h and the
    # allocation is 72h, leaving protected time for handoff and parallel verifier
    # reloads of all six checkpoints.
    research_deadline_epoch=$(( $(date -u +%s) + 237600 ))
    date -u -d "@${research_deadline_epoch}" +%FT%TZ >"${run_contract_dir}/RESEARCH_DEADLINE_UTC"
    date -u +%FT%TZ >"${run_contract_dir}/RESEARCH_STARTED_UTC"
    printf '%s\n' "${candidate_output}" >"${run_contract_dir}/HOST_OUTPUT_ROOT"
fi
baseline_data=${task_snapshot}/baselines
[[ -s ${baseline_data}/scale_baselines.json ]] || {
    printf 'error: trusted six-scale baseline manifest is missing: %s\n' \
        "${baseline_data}/scale_baselines.json" >&2
    exit 3
}

# Harbor starts the agent with Apptainer --containall. The candidate runners
# still need the allocation identity to lease nodes, while LSF's blaunch must
# remain in the host job namespace. A trusted host broker accepts only one
# allocated target host per request; the contained client supplies the exact
# argv and sanitized runtime environment needed by the remote worker.
lsf_blaunch=${BLAUNCH:-$(command -v blaunch || true)}
[[ -x ${lsf_blaunch} ]] || {
    printf 'error: LSF blaunch executable is unavailable\n' >&2
    exit 3
}
proxy_broker=${task_snapshot}/cluster/blaunch_proxy.py
proxy_client_source=${task_snapshot}/cluster/blaunch_proxy_client.py
[[ -r ${proxy_broker} && -r ${proxy_client_source} ]] || {
    printf 'error: trusted blaunch proxy scripts are unavailable\n' >&2
    exit 3
}
proxy_client=${run_contract_dir}/blaunch_proxy_client.py
install -m 0555 -- "${proxy_client_source}" "${proxy_client}"
if (( resume_existing == 1 )); then
    proxy_root=${run_dir}/control/blaunch-proxy-${LSB_JOBID}
else
    proxy_root=${run_dir}/control/blaunch-proxy
fi
mkdir -m 0700 -- "${proxy_root}"
apptainer_binary=${APPTAINER:-$(command -v apptainer || true)}
[[ -x ${apptainer_binary} ]] || {
    printf 'error: Apptainer executable is unavailable\n' >&2
    exit 3
}
apptainer_root=$(cd -- "$(dirname -- "${apptainer_binary}")/.." && pwd -P)
export APPTAINERENV_LSB_JOBID=${LSB_JOBID}
# Host leases account in GPU slots, not scheduler CPU slots.  Resumed outer
# allocations intentionally reserve one LSF slot per eight-GPU node, so expose
# a normalized eight-slot inventory to the contained task tools while the host
# broker retains the real LSF allocation for target validation.
if (( resume_existing == 1 )); then
    logical_lsb_mcpu_hosts=$(python3 - "${LSB_MCPU_HOSTS}" <<'PY'
import sys

fields = sys.argv[1].split()
order = []
seen = set()
for index in range(0, len(fields), 2):
    host = fields[index]
    if host not in seen:
        seen.add(host)
        order.append(host)
print(" ".join(f"{host} 8" for host in order))
PY
    )
else
    logical_lsb_mcpu_hosts=${LSB_MCPU_HOSTS}
fi
export APPTAINERENV_LSB_MCPU_HOSTS=${logical_lsb_mcpu_hosts}
export APPTAINERENV_BLAUNCH=/run-contract/blaunch_proxy_client.py
export APPTAINERENV_BLAUNCH_PROXY_ROOT=/blaunch-proxy

# Keep the output available at both the public candidate path and its original
# host path.  The trusted nested launcher executes inside Harbor but must pass
# host-visible source/checkpoint paths to Apptainer processes started through
# the host blaunch broker.
bind_paths="${asset_root}/data:${asset_root}/data:ro,${asset_root}/cache:${asset_root}/cache:rw,${asset_root}/images:${asset_root}/images:ro,${apptainer_root}:${apptainer_root}:ro,${baseline_data}:/task-data:ro,${candidate_output}:/app/output:rw,${candidate_output}:${candidate_output}:rw,${run_contract_dir}:/run-contract:ro,${proxy_root}:/blaunch-proxy:rw"
if (( resume_existing == 1 )); then
    bind_paths="${bind_paths},${task_snapshot}/environment/task-tools:/task-tools:ro"
    job_name=$(basename -- "${run_dir}")-resume-${LSB_JOBID}
else
    job_name=$(basename -- "${run_dir}")
fi
cd -- "${harbor_root}"
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
python3 "${proxy_broker}" \
    --root "${proxy_root}" \
    --blaunch "${lsf_blaunch}" \
    >"${run_dir}/control/blaunch-proxy.log" 2>&1 &
proxy_pid=$!
for _attempt in $(seq 1 100); do
    [[ -s ${proxy_root}/READY.json ]] && break
    kill -0 "${proxy_pid}" 2>/dev/null || {
        printf 'error: blaunch proxy exited during startup\n' >&2
        exit 3
    }
    sleep 0.1
done
[[ -s ${proxy_root}/READY.json ]] || {
    printf 'error: blaunch proxy did not become ready\n' >&2
    exit 3
}

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
    --job-name "${job_name}" \
    --jobs-dir "${run_dir}/harbor" \
    --yes
harbor_status=$?
set -e
kill -TERM "${proxy_pid}" 2>/dev/null || true
wait "${proxy_pid}" 2>/dev/null || true
proxy_pid=
exit "${harbor_status}"
