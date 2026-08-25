#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: $0 pretrain.py [repository arguments ...]" >&2
    exit 2
fi

: "${LSB_JOBID:?the released 4x8 lane must run inside one LSF allocation}"
: "${GDN_SHARED_WORKSPACE:?missing host path for the shared candidate workspace}"
: "${GDN_SHARED_OUTPUT:?missing host path for the shared output directory}"
: "${GDN_DATA_ROOT:?missing host path for staged release-format data}"

APPTAINER=${GDN_APPTAINER_BINARY:-/proj/datasets/interns/yuetai/rsi-nemotron/apptainer-env/bin/apptainer}
SIF=${GDN_ENVIRONMENT_SIF:-/proj/datasets/interns/yuetai/agent_envs/scale_autoresearch_tasks/images/gated_deltanet-environment.sif}
BLAUNCH=${GDN_BLAUNCH_BINARY:-/opt/share/ETELSFSH/10.1/linux4.18-glibc2.28-x86_64/bin/blaunch}

hosts=()
if [ -n "${LSB_MCPU_HOSTS:-}" ]; then
    read -r -a host_slots <<< "$LSB_MCPU_HOSTS"
    for ((index = 0; index < ${#host_slots[@]}; index += 2)); do
        hosts+=("${host_slots[index]}")
    done
elif [ -n "${LSB_HOSTS:-}" ]; then
    read -r -a allocated_hosts <<< "$LSB_HOSTS"
    for host in "${allocated_hosts[@]}"; do
        if [[ " ${hosts[*]} " != *" $host "* ]]; then
            hosts+=("$host")
        fi
    done
fi

if [ "${#hosts[@]}" -ne 4 ]; then
    echo "released launch requires exactly 4 allocated hosts; found ${#hosts[@]}: ${hosts[*]-}" >&2
    exit 3
fi
for required in "$APPTAINER" "$SIF" "$BLAUNCH"; do
    if [ ! -e "$required" ]; then
        echo "missing multi-node launch dependency: $required" >&2
        exit 3
    fi
done

master_addr=${hosts[0]}
master_port=${GDN_MASTER_PORT:-$((20000 + LSB_JOBID % 20000))}
launch_label=${GDN_LAUNCH_LABEL:-release}
mkdir -p "$GDN_SHARED_OUTPUT/logs"

pids=()
for node_rank in 0 1 2 3; do
    host=${hosts[node_rank]}
    triton_cache="/app/output/triton/node-${node_rank}"
    hf_home="/app/output/hf-cache/node-${node_rank}"
    mkdir -p "$GDN_SHARED_OUTPUT/hf-cache/node-${node_rank}/transformers"
    command=(
        "$APPTAINER" exec --nv --containall --writable-tmpfs
        --pwd /app/project
        --bind "$GDN_SHARED_WORKSPACE:/app/project"
        --bind "$GDN_SHARED_OUTPUT:/app/output"
        --bind "$GDN_DATA_ROOT:/datasets:ro"
        --env SLURM_NNODES=4
        --env WANDB_MODE=offline
        --env "TRITON_CACHE_DIR=$triton_cache"
        --env "HF_HOME=$hf_home"
        --env "TRANSFORMERS_CACHE=$hf_home/transformers"
        --env HF_HUB_OFFLINE=1
        --env TRANSFORMERS_OFFLINE=1
        --env PYTHONPATH=/app/project
        --env "GDN_EVAL_ITERS=${GDN_EVAL_ITERS:-15}"
        --env "GDN_VALIDATION_DATA_DIR=${GDN_VALIDATION_DATA_DIR:-/datasets/gated_deltanet-official/slimpajama/packed/slim}"
        --env "GDN_SINGLE_GPU_VERIFIER_SMOKE=${GDN_SINGLE_GPU_VERIFIER_SMOKE:-0}"
        "$SIF"
        env -u LSB_DJOB_RANKFILE -u LSB_DJOB_HOSTFILE -u LSB_PJL_TASK_GEOMETRY
        torchrun
        --nnodes=4
        --nproc-per-node=8
        --node-rank="$node_rank"
        --master-addr="$master_addr"
        --master-port="$master_port"
        "$@"
    )
    printf -v remote_command '%q ' "${command[@]}"
    "$BLAUNCH" -z "$host" bash -lc "$remote_command" \
        >"$GDN_SHARED_OUTPUT/logs/${launch_label}-node-${node_rank}.log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
exit "$status"
