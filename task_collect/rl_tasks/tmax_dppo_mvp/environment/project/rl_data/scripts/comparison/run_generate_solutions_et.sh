#!/bin/bash
#SBATCH --job-name=rl-gen-sol-et
#SBATCH --output=logs/gen_sol_et_%j.out
#SBATCH --error=logs/gen_sol_et_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=960G

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Run our solution-generation harness on the (converted) Endless-     ║
# ║  Terminals dataset under the VANILLUX harness, using the uniform     ║
# ║  comparison config (NUM_SOLUTIONS=8, MAX_ACTIONS=64,                 ║
# ║  COMMAND_TIMEOUT=600, SAMPLE_SIZE=250, fixed SAMPLE_SEED=0). This is ║
# ║  the same config used by the rebased skill-tax reference run + every ║
# ║  other baseline in this directory, so head-to-head pass@1/4/8 is     ║
# ║  apples-to-apples across all 5 datasets.                              ║
# ║                                                                       ║
# ║  Output: per-task                                                     ║
# ║    <task>/solutions/<MODEL_TAG>_vanillux_summary.json                 ║
# ║  (the legacy `<MODEL_TAG>_summary.json` bash-harness files, if any,   ║
# ║  are preserved alongside since the filenames are disjoint).           ║
# ║                                                                       ║
# ║  Prerequisite: run ingest_endless_terminals.py to populate TASKS_DIR. ║
# ║  ET does NOT use shared base SIFs (its container.defs are self-       ║
# ║  contained), so per-task SIF builds happen lazily.                    ║
# ╚═══════════════════════════════════════════════════════════════════════╝

set -euo pipefail

# ---- Parameters (edit here) ----
TASKS_DIR="rl_data/output/tasks_endless_terminals"
# Override via env.  Examples:
#   API model:   MODEL="gemini/gemini-3-flash-preview"
#   Local vLLM:  MODEL="hosted_vllm/Qwen/Qwen2.5-Coder-7B-Instruct" \
#                HOSTED_VLLM_API_BASE="http://localhost:8000/v1"
#   Ollama:      MODEL="ollama_chat/qwen2.5-coder:7b" \
#                OLLAMA_API_BASE="http://localhost:11434"
MODEL="${MODEL:-gemini/gemini-3-flash-preview}"
# Solution-sampling harness. The 0515 comparison rebase pinned every baseline
# to the mini-swe-agent-style 'vanillux' harness so the head-to-head against
# our (also-vanillux) reference run is apples-to-apples. Override to 'bash'
# only if you specifically want to reproduce the pre-0515 results.
HARNESS="${HARNESS:-vanillux}"
# NUM_SOLUTIONS controls pass@k breadth. run_n_solutions{,_vanillux}() returns
# pass@k for every k in [1..N], so NUM_SOLUTIONS=8 lights up pass@1/4/8 in one
# run. Uniform across all five comparison datasets.
NUM_SOLUTIONS="${NUM_SOLUTIONS:-8}"
# 64 = vanillux convention (matches the mini-swe-agent "Recommended Workflow"
# budget and the upstream VanilluxAgent per_instance_call_limit). The legacy
# bash baselines used 16; raising it here is required for the vanillux prompts
# to play out, not just a nice-to-have.
MAX_ACTIONS="${MAX_ACTIONS:-64}"
# MAX_TOKENS is the per-turn generation cap.  65536 was chosen for gemini-3
# flash (1M-token context).  For a local vLLM with --max-model-len=32768 it
# would be rejected with `max_tokens > max_model_len`, so the helper's
# _vllm_wait_ready_local auto-caps this when LAUNCH_VLLM=1.  Override here
# via env (e.g. MAX_TOKENS=8192) when running against a local model.
MAX_TOKENS="${MAX_TOKENS:-65536}"
NUM_TASKS=999999
START_AT=0
SOLUTION_TEMPERATURE=0.7
# 600s matches the vanillux reference script. 60s was the old comparison
# default; with 8 parallel solutions hammering an ET container's setup it was
# producing spurious timeouts on the heavier tasks.
COMMAND_TIMEOUT="${COMMAND_TIMEOUT:-600}"
SHELL_INIT_TIMEOUT=240
SHELL_INIT_ATTEMPTS=3
BUILD_WORKERS=12
BUILD_RETRIES=3
# NOTE: no BASE_SIFS_DIR — ET tasks don't share the 9 prebuilt bases.
FORCE_RERUN=0
LOG_COMMANDS=0
DISABLE_TERMINAL_LOG=0

# Cost-bounded subsample: 250 tasks × 8 attempts = 2000 trajectories per
# dataset, uniform across all 5 comparison corpora. Fixed seed=0 so the same
# 250 ET tasks are picked on every rerun (and stay stable if anyone adds new
# ET tasks upstream).
SAMPLE_SIZE="${SAMPLE_SIZE:-250}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"

# WORKERS = concurrent TASKS processed at once.  NUM_POOL_WORKERS = concurrent
# solutions/env operations within a single task.  Both env-overridable so you
# can crank or shrink them per run (e.g. to verify across-task parallelism).
WORKERS="${WORKERS:-12}"
NUM_POOL_WORKERS="${NUM_POOL_WORKERS:-16}"
# --------------------------------

_RUN_TS=$(date -u +%Y%m%d_%H%M%S)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"
mkdir -p logs

# In-job vLLM bring-up. No-op unless LAUNCH_VLLM=1. We kick it off here so
# the slow SIF base build below overlaps with vLLM weight loading; we then
# block on readiness right before the solver call (see _vllm_wait_ready_local).
# When enabled, the helper will (later) set MODEL=hosted_vllm/$VLLM_MODEL +
# HOSTED_VLLM_API_BASE so the solver routes through the local server.
# shellcheck source=./_vllm_local.sh
source "$SCRIPT_DIR/_vllm_local.sh"
_vllm_start_local

export APPTAINER_DOCKER_USERNAME="${APPTAINER_DOCKER_USERNAME:?Set APPTAINER_DOCKER_USERNAME before running}"
export APPTAINER_DOCKER_PASSWORD="${APPTAINER_DOCKER_PASSWORD:?Set APPTAINER_DOCKER_PASSWORD before running}"

# Local-model support via litellm env passthrough.
# These are harmless if unset (API model is used instead).
export HOSTED_VLLM_API_BASE="${HOSTED_VLLM_API_BASE:-}"
export OLLAMA_API_BASE="${OLLAMA_API_BASE:-}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-}"
if [[ -n "${HOSTED_VLLM_API_BASE:-}${OLLAMA_API_BASE:-}${OPENAI_API_BASE:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="EMPTY"  # some local servers don't care but litellm checks
fi

export APPTAINER_CACHEDIR="/gpfs/projects/h2lab/osey/apptainer_cache"
export APPTAINER_TMPDIR="/tmp/apptainer_tmp"
mkdir -p "$APPTAINER_TMPDIR"

# Keep apptainer instance logs off GPFS (rootless cgroups aren't supported there).
# Unconditionally ensure the /tmp target dir exists -- it can disappear between
# runs (node reboots, /tmp cleanup) while the symlink on GPFS persists, leaving
# a dangling symlink that makes `apptainer instance start` fail with
# `mkdir ...: file exists`. Creating the target first heals that case cheaply.
mkdir -p /tmp/apptainer_instances
if [ ! -L "$HOME/.apptainer/instances" ]; then
  rm -rf "$HOME/.apptainer/instances"
  ln -s /tmp/apptainer_instances "$HOME/.apptainer/instances"
fi

# ---- ET-only base SIF bootstrap -----------------------------------------
# Every ET container.def uses `Bootstrap: localimage` + `From: ./ubuntu_22.04.sif`
# resolved against the CWD (PROJECT_ROOT). Build that base SIF once here so the
# 2.5k per-task builds can layer on top of it.
#
# LEVER 2: Pre-install python3 + pip + pytest + ca-certificates into the base
# SIF so the 2,488/2,492 ET defs that install those same packages turn into
# essentially-free no-ops. Without this, running N parallel builds thundering-
# herds archive.ubuntu.com + files.pythonhosted.org and pip times out.
#
# LEVER 3: Keep /var/lib/apt/lists/* populated in the base SIF. Each ET def
# still does `apt-get update` at the top of its %post; with cached lists,
# that becomes a handful of small HEAD/InRelease requests (~1-3s) instead of
# redownloading the full ~46MB package index (~20-30s) from archive.ubuntu.com.
# Adds ~20MB to the base SIF; saves >3 hours across 500 per-task builds.
#
# A version marker file (BASE_UBUNTU_SIF.version) lets us force a rebuild when
# the recipe below changes -- bump BASE_UBUNTU_SIF_VERSION to invalidate.
BASE_UBUNTU_SIF="$PROJECT_ROOT/ubuntu_22.04.sif"
BASE_UBUNTU_SIF_VERSION="v3-py+pytest+aptlists"
BASE_UBUNTU_SIF_MARK="${BASE_UBUNTU_SIF}.version"

_need_base_build=0
if [ ! -f "$BASE_UBUNTU_SIF" ]; then
  _need_base_build=1
  _reason="missing"
elif [ ! -f "$BASE_UBUNTU_SIF_MARK" ] || \
     [ "$(cat "$BASE_UBUNTU_SIF_MARK" 2>/dev/null)" != "$BASE_UBUNTU_SIF_VERSION" ]; then
  _need_base_build=1
  _reason="stale (want ${BASE_UBUNTU_SIF_VERSION}, got $(cat "$BASE_UBUNTU_SIF_MARK" 2>/dev/null || echo none))"
fi

if [ "$_need_base_build" = "1" ]; then
  echo "=== ET base SIF ${_reason}; building enriched base (python3 + pip + pytest) ==="
  BASE_UBUNTU_DEF="$(mktemp --suffix=.def)"
  trap 'rm -f "$BASE_UBUNTU_DEF"' EXIT
  cat > "$BASE_UBUNTU_DEF" <<'EOF'
Bootstrap: docker
From: ubuntu:22.04

%post
    set -e
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-setuptools \
        python3-wheel \
        ca-certificates
    pip3 install --no-cache-dir pytest
    apt-get clean
    # NOTE: Intentionally keep /var/lib/apt/lists/* populated (Lever 3) so
    # per-task `apt-get update` is nearly a no-op instead of re-fetching 46MB.

%labels
    Author rl_data-et-bootstrap
    Description "Ubuntu 22.04 + python3 + pip + pytest + prepopulated apt lists, prebaked for ET per-task builds."
EOF
  if ! apptainer build --force "$BASE_UBUNTU_SIF" "$BASE_UBUNTU_DEF"; then
    echo "ERROR: failed to build enriched $BASE_UBUNTU_SIF (ET per-task builds require this)." >&2
    exit 1
  fi
  echo "$BASE_UBUNTU_SIF_VERSION" > "$BASE_UBUNTU_SIF_MARK"
  rm -f "$BASE_UBUNTU_DEF"
  trap - EXIT
  echo "=== ET base SIF ready: $BASE_UBUNTU_SIF (version=${BASE_UBUNTU_SIF_VERSION}) ==="
else
  echo "=== ET base SIF already present: $BASE_UBUNTU_SIF (version=${BASE_UBUNTU_SIF_VERSION}, skipping build) ==="
fi

# Block until the in-job vLLM server is ready (no-op when LAUNCH_VLLM!=1).
# Also (re)exports MODEL/HOSTED_VLLM_API_BASE/OPENAI_API_KEY for the solver.
_vllm_wait_ready_local

# Derive model-tagged paths NOW (after the helper may have rewritten MODEL).
# Include $HARNESS in the log filename so a vanillux rerun doesn't clobber
# legacy bash-harness logs from earlier comparison runs.
_MODEL_TAG=$(echo "$MODEL" | tr '/' '_')
TERMINAL_LOG="${TASKS_DIR}/logs/${_MODEL_TAG}_${HARNESS}_${_RUN_TS}.log"

EXTRA_ARGS=()
if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--force-rerun)
fi
if [[ "${LOG_COMMANDS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--log-commands)
fi
if [[ "${SAMPLE_SIZE:-0}" != "0" ]]; then
  EXTRA_ARGS+=(--sample-size "$SAMPLE_SIZE" --sample-seed "$SAMPLE_SEED")
fi
if [[ "${DISABLE_TERMINAL_LOG:-0}" != "1" ]]; then
  TL="${TERMINAL_LOG}"
  if [[ "$TL" != /* ]]; then
    TL="$PROJECT_ROOT/$TL"
  fi
  mkdir -p "$(dirname "$TL")"
  EXTRA_ARGS+=(--terminal-log "$TL")
fi

echo "=== ET comparison run: MODEL=${MODEL}, HARNESS=${HARNESS}, WORKERS=${WORKERS}, NUM_SOLUTIONS=${NUM_SOLUTIONS}, SAMPLE_SIZE=${SAMPLE_SIZE} ==="
echo "=== Concurrent containers: $(( WORKERS * NUM_SOLUTIONS )) ==="

uv run python -m rl_data.generate_solutions \
    --tasks-dir "$TASKS_DIR" \
    --model "$MODEL" \
    --num-solutions "$NUM_SOLUTIONS" \
    --max-actions "$MAX_ACTIONS" \
    --max-tokens "$MAX_TOKENS" \
    --num-tasks "$NUM_TASKS" \
    --start-at "$START_AT" \
    --workers "$WORKERS" \
    --num-pool-workers "$NUM_POOL_WORKERS" \
    --solution-temperature "$SOLUTION_TEMPERATURE" \
    --command-timeout "$COMMAND_TIMEOUT" \
    --shell-init-timeout "$SHELL_INIT_TIMEOUT" \
    --shell-init-attempts "$SHELL_INIT_ATTEMPTS" \
    --build-workers "$BUILD_WORKERS" \
    --build-retries "$BUILD_RETRIES" \
    --harness "$HARNESS" \
    --verbose \
    "${EXTRA_ARGS[@]}"
