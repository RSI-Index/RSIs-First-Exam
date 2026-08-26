#!/bin/bash
set -uxo pipefail
WORK=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/gracieh/task
pip install --no-cache-dir -q scipy scikit-learn 2>/dev/null
pip install --no-cache-dir -q --no-deps $WORK/gpic/gpic_eval
export TORCH_HOME=$WORK/torch_home HF_HUB_OFFLINE=1
export GPIC_QWEN_DIR=$WORK/hf_models/Qwen3-1.7B
python $WORK/cheat/run_battery.py
