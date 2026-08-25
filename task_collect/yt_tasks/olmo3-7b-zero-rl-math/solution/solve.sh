#!/bin/bash
#
# Reference solution outline: Olmo 3 RL-Zero (Math).
# RLVR from Olmo-3-7B base with the OlmoRL algorithm on Dolci-RL-Zero-Math via
# open-instruct, then evaluate on AIME 2024.
#
# This is an OUTLINE of the intended recipe (Olmo 3 report §6, Table 49). The exact
# open-instruct entry point / flag names should be taken from the open-instruct repo
# at run time; hyperparameter VALUES below are fixed by the paper.

set -euo pipefail

export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p /app/models /app/data /app/checkpoints /app/logs /app/eval

# 1. Download base model + RL-Zero math data ---------------------------------
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("allenai/Olmo-3-7B", local_dir="/app/models/Olmo-3-7B")
PY

python3 - <<'PY'
from datasets import load_dataset
ds = load_dataset("allenai/Dolci-RL-Zero-Math-7B", split="train")   # 13,314 prompts
ds.save_to_disk("/app/data/dolci-rl-zero-math")
PY

# 2. Prepare data: apply the RL-Zero Math prompt template (Figure 37) --------
#    "Solve the following math problem step by step. ... Answer: $Answer ..."
#    Emit open-instruct RLVR-format JSONL to /app/data/rlvr_math.jsonl with the
#    rule-based (SymPy) math verifier as the reward source.

# 3. Train with OlmoRL / GRPO via open-instruct ------------------------------
#    Fixed hyperparameters (Table 49, 7B RL-Zero column):
#      lr=1e-6 constant, steps=2000, minibatches=1
#      max_prompt_len=2048, response_len=16384
#      unique_prompts_per_batch=32, group_size=8, temperature=1.0
#      clip_lower=0.2, clip_higher=0.272, TIS cap=2.0
#      no KL, no entropy, no std-normalization (Dr.GRPO), token-level loss,
#      zero-gradient filtering + active sampling, NO overlong masking
#    Disaggregated layout (Table 49): 8 learner GPUs + 64 vLLM actor GPUs (TP=1),
#    max_asynchrony=8, launched via Ray across nodes.
#
#    (Use the open-instruct RLVR/GRPO launcher; write checkpoints + optimizer state
#     to /app/checkpoints and per-step metrics to /app/logs/train.jsonl.)

# 4. Evaluate on AIME 2024 (and base baseline) -------------------------------
#    32 samples/problem, temperature=1.0, max response length=32768.
#    pass@1 = bootstrapped avg over 32 samples; also compute pass@32.
#    Extract answer from the "Answer:" line; score with the SymPy math verifier.
#    Evaluate BOTH the trained checkpoint and the Olmo-3-7B base model.
#    Write /app/eval/aime2024.json:
#      {"trained_pass@1": ..., "base_pass@1": ..., "pass@32": ...,
#       "n_samples": 32, "temperature": 1.0}
#    Write the trained pass@1 to /app/result.txt.

echo "See comments above: this outline must be executed with the real open-instruct trainer."
