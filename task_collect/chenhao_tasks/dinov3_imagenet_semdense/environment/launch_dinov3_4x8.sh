#!/usr/bin/env bash
# Multi-node launch translation for the dinov3-imagenet-semdense task —
# the analog of the sample task's launch_gdn_4x8_lsf.sh. This is task
# infrastructure, not an optimization variable.
#
# Usage:  launch_dinov3_4x8.sh <train-config.yaml> [extra train.py args...]
#
# Contract: released topology is 4 nodes x 8 GPUs = 32 ranks, exactly.
# The scheduler (LSF/Slurm/other) must export the standard torchrun
# rendezvous variables on every node before invoking this script:
#   DINO_NNODES (=4), DINO_NODE_RANK, DINO_MASTER_ADDR, DINO_MASTER_PORT
# On a single-node smoke run (DINO_SINGLE_NODE_SMOKE=1) it falls back to
# 1 node x 8 GPUs — for infrastructure debugging only; smoke runs are not
# comparable to the sealed anchors and never count as scientific evidence.
#
# The known upstream README bug (fast-setup command passes an ImageNet22k:
# dataset path against the IN1k config) is corrected here: this launcher
# always passes the staged IN1k path with the ImageNet: dataset class.
set -euo pipefail

CONFIG="${1:?usage: launch_dinov3_4x8.sh <train-config.yaml> [args...]}"
shift

REPO_ROOT="${DINO_REPO_ROOT:-/app/project/dinov3}"
IMAGENET_DIR="${DINO_IMAGENET_DIR:-/datasets/imagenet-1k}"

if [ "${DINO_SINGLE_NODE_SMOKE:-0}" = "1" ]; then
  NNODES=1
  NODE_RANK=0
  MASTER_ADDR=127.0.0.1
  MASTER_PORT="${DINO_MASTER_PORT:-29500}"
else
  NNODES="${DINO_NNODES:?DINO_NNODES not set (expected 4)}"
  NODE_RANK="${DINO_NODE_RANK:?DINO_NODE_RANK not set}"
  MASTER_ADDR="${DINO_MASTER_ADDR:?DINO_MASTER_ADDR not set}"
  MASTER_PORT="${DINO_MASTER_PORT:-29500}"
  if [ "${NNODES}" != "4" ]; then
    echo "error: released topology is 4 nodes; got DINO_NNODES=${NNODES}" >&2
    exit 2
  fi
fi

# BUILD-TIME VERIFY: exact train entrypoint/flags re-checked against the
# pinned commit when the image is first built (dinov3/train/train.py and
# its config-file/dataset-path options).
cd "${REPO_ROOT}"
exec torchrun \
  --nnodes "${NNODES}" \
  --nproc_per_node 8 \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  -m dinov3.train.train \
  --config-file "${CONFIG}" \
  train.dataset_path="ImageNet:split=TRAIN:root=${IMAGENET_DIR}:extra=${IMAGENET_DIR}" \
  "$@"
