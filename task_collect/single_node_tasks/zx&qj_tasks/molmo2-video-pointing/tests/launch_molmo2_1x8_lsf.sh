#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <script.py> [repository arguments ...]" >&2
    exit 2
fi

: "${LSB_JOBID:?the released 1x8 lane must run inside one LSF allocation}"
: "${MVP_SHARED_WORKSPACE:?missing host path for the shared candidate workspace}"
: "${MVP_SHARED_OUTPUT:?missing host path for the shared output directory}"
: "${MVP_DATA_ROOT:?missing host path for staged release-format data}"

APPTAINER=${MVP_APPTAINER_BINARY:-apptainer}
SIF=${MVP_ENVIRONMENT_SIF:?missing environment sif image}
launch_label=${MVP_LAUNCH_LABEL:-release}
master_port=${MVP_MASTER_PORT:-$((20000 + LSB_JOBID % 20000))}
nproc=${MVP_NPROC_PER_NODE:-8}
mkdir -p "$MVP_SHARED_OUTPUT/logs" \
         "$MVP_SHARED_OUTPUT/hf-cache/transformers" \
         "$MVP_SHARED_OUTPUT/dataloader-cache"

"$APPTAINER" exec --nv --containall --writable-tmpfs \
    --pwd /app/project \
    --bind "$MVP_SHARED_WORKSPACE:/app/project" \
    --bind "$MVP_SHARED_OUTPUT:/app/output" \
    --bind "$MVP_DATA_ROOT:/datasets:ro" \
    --env WANDB_MODE=offline \
    --env HF_DATASETS_OFFLINE=1 \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env "MOLMO_DATA_DIR=${MOLMO_DATA_DIR:-/datasets/molmo2-official/molmo_data}" \
    --env "HF_HOME=${HF_HOME:-/datasets/molmo2-official/huggingface}" \
    --env "MOLMO2_START_CHECKPOINT=${MOLMO2_START_CHECKPOINT:-/datasets/molmo2-official/checkpoints/Molmo2-4B-Pretrain}" \
    --env OLMO_SHARED_FS=1 \
    --env OMP_NUM_THREADS=8 \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env NCCL_TIMEOUT_MINUTES=20 \
    --env PYTHONPATH=/app/project \
    "$SIF" \
    torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="$nproc" \
    --master-port="$master_port" \
    "$@" \
    >"$MVP_SHARED_OUTPUT/logs/${launch_label}.log" 2>&1
