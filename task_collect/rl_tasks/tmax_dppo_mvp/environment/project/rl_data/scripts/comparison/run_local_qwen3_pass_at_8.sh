#!/bin/bash
#SBATCH --job-name=rl-qwen3-pass8
#SBATCH --output=logs/qwen3_pass8_%j.out
#SBATCH --error=logs/qwen3_pass8_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Single-job, single-node, single-vLLM orchestrator for the comparison ║
# ║  baselines under a LOCAL model (default: Qwen/Qwen3-8B on 8xH200).    ║
# ║                                                                        ║
# ║  Why this exists: each per-dataset run_generate_solutions_*.sh ALREADY ║
# ║  knows how to launch vLLM in-job (LAUNCH_VLLM=1).  But running four    ║
# ║  separate jobs would pay 4x vLLM cold-start + 4x weight-load.  This    ║
# ║  script brings vLLM up ONCE and runs the four solver scripts in        ║
# ║  sequence against it, all under one SLURM allocation.                 ║
# ║                                                                        ║
# ║  What you get per dataset: solutions/<MODEL_TAG>_summary.json          ║
# ║  containing pass_at_k = {1: ..., 2: ..., ..., 8: ...}, i.e. pass@1     ║
# ║  AND pass@8 in the same run.  The gemini summaries you already have    ║
# ║  live alongside (different MODEL_TAG) and are not overwritten.         ║
# ║                                                                        ║
# ║  Quick start (full run, all 4 baselines, pass@8 with Qwen3-8B):        ║
# ║                                                                        ║
# ║    APPTAINER_DOCKER_USERNAME=... APPTAINER_DOCKER_PASSWORD=... \       ║
# ║      sbatch rl_data/scripts/comparison/run_local_qwen3_pass_at_8.sh   ║
# ║                                                                        ║
# ║  Cost-bounded (250 random tasks per baseline):                         ║
# ║                                                                        ║
# ║    SAMPLE_SIZE=250 sbatch rl_data/scripts/comparison/run_local_qwen3_pass_at_8.sh
# ║                                                                        ║
# ║  Subset of datasets:                                                   ║
# ║                                                                        ║
# ║    DATASETS="et openthoughts" sbatch ...                               ║
# ║                                                                        ║
# ║  Different model (e.g. Qwen2.5-Coder-7B):                              ║
# ║                                                                        ║
# ║    VLLM_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct sbatch ...                ║
# ║                                                                        ║
# ║  Different pass@k (e.g. pass@16):                                      ║
# ║                                                                        ║
# ║    NUM_SOLUTIONS=16 sbatch ...                                         ║
# ║                                                                        ║
# ║  Env knobs forwarded to each child solve script:                       ║
# ║    NUM_SOLUTIONS, SAMPLE_SIZE, SAMPLE_SEED, FORCE_RERUN, LOG_COMMANDS, ║
# ║    DISABLE_TERMINAL_LOG.                                              ║
# ║                                                                        ║
# ║  vLLM knobs (forwarded to _vllm_local.sh):                             ║
# ║    VLLM_MODEL, VLLM_PORT, VLLM_TP, VLLM_DP, VLLM_MAX_LEN,              ║
# ║    VLLM_GPU_UTIL, VLLM_DTYPE, VLLM_TOOL_CALL_PARSER,                   ║
# ║    VLLM_REASONING_PARSER, VLLM_DISABLE_THINKING, VLLM_EXTRA_ARGS.      ║
# ╚═══════════════════════════════════════════════════════════════════════╝

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

# ---- Defaults (override via env) -----------------------------------------
DATASETS="${DATASETS:-et openthoughts termigen terminaltraj}"
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-8B}"
export NUM_SOLUTIONS="${NUM_SOLUTIONS:-8}"   # pass@1..pass@N in one shot

# Forward common solve-script knobs (no-ops if unset).
export SAMPLE_SIZE="${SAMPLE_SIZE:-0}"
export SAMPLE_SEED="${SAMPLE_SEED:-0}"
export FORCE_RERUN="${FORCE_RERUN:-0}"
export LOG_COMMANDS="${LOG_COMMANDS:-0}"
export DISABLE_TERMINAL_LOG="${DISABLE_TERMINAL_LOG:-0}"

# Apptainer/Docker passthrough — required by ET + OT solve scripts; passed
# through harmlessly otherwise.
export APPTAINER_DOCKER_USERNAME="${APPTAINER_DOCKER_USERNAME:-}"
export APPTAINER_DOCKER_PASSWORD="${APPTAINER_DOCKER_PASSWORD:-}"

# ---- 1. Bring up vLLM once for the whole job -----------------------------
export LAUNCH_VLLM=1
# shellcheck source=./_vllm_local.sh
source "$SCRIPT_DIR/_vllm_local.sh"
_vllm_start_local
_vllm_wait_ready_local

# After this point: MODEL, HOSTED_VLLM_API_BASE, OPENAI_API_KEY are set in
# this shell's env and will be inherited by the child solver scripts.
# IMPORTANT: turn LAUNCH_VLLM off so each child does NOT try to start its
# own vLLM server -- they'll inherit HOSTED_VLLM_API_BASE and route to ours.
export LAUNCH_VLLM=0

echo
echo "=== Running solvers under MODEL=${MODEL} ==="
echo "    HOSTED_VLLM_API_BASE=${HOSTED_VLLM_API_BASE}"
echo "    NUM_SOLUTIONS=${NUM_SOLUTIONS} (pass@1..pass@${NUM_SOLUTIONS})"
echo "    SAMPLE_SIZE=${SAMPLE_SIZE}, SAMPLE_SEED=${SAMPLE_SEED}"
echo "    DATASETS=${DATASETS}"
echo

# ---- 2. Run each requested dataset's solve script ------------------------
# We invoke them as plain `bash`, not `sbatch`, so they run in this same
# allocation rather than queuing as new jobs. Their #SBATCH directives are
# just comments at that point.
declare -A DS_SCRIPT=(
  [et]="$SCRIPT_DIR/run_generate_solutions_et.sh"
  [openthoughts]="$SCRIPT_DIR/run_generate_solutions_openthoughts.sh"
  [termigen]="$SCRIPT_DIR/run_generate_solutions_termigen.sh"
  [terminaltraj]="$SCRIPT_DIR/run_generate_solutions_terminaltraj.sh"
)

failed=()
for ds in $DATASETS; do
  script="${DS_SCRIPT[$ds]:-}"
  if [[ -z "$script" ]]; then
    echo "WARN: unknown dataset '$ds' (valid: et openthoughts termigen terminaltraj); skipping" >&2
    continue
  fi
  echo
  echo "=========================================================================="
  echo ">>> [$ds] $(date -u +%FT%TZ) -- running $script"
  echo "=========================================================================="
  if bash "$script"; then
    echo ">>> [$ds] OK"
  else
    rc=$?
    echo ">>> [$ds] FAILED (exit=$rc)" >&2
    failed+=("$ds")
    # Don't bail -- we want the other datasets to still run.
  fi
done

echo
if [[ "${#failed[@]}" -gt 0 ]]; then
  echo "=== One or more solvers failed: ${failed[*]} ==="
  exit 1
fi
echo "=== All datasets completed successfully ==="
