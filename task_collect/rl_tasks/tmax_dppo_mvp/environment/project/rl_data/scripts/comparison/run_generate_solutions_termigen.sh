#!/bin/bash
#SBATCH --job-name=rl-gen-sol-termigen
#SBATCH --output=logs/gen_sol_termigen_%j.out
#SBATCH --error=logs/gen_sol_termigen_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=960G

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Run our solution-generation harness on the (converted) TermiGen     ║
# ║  (ucsb-mlsec/terminal-bench-env, Harbor 2.0) dataset under the       ║
# ║  VANILLUX harness, using the uniform comparison config              ║
# ║  (NUM_SOLUTIONS=8, MAX_ACTIONS=64, COMMAND_TIMEOUT=600,             ║
# ║  SAMPLE_SIZE=250, fixed SAMPLE_SEED=0).                              ║
# ║                                                                        ║
# ║  Output: per-task                                                      ║
# ║    <task>/solutions/<MODEL_TAG>_vanillux_summary.json                  ║
# ║                                                                        ║
# ║  Prerequisite: run rl_data/scripts/comparison/run_ingest_termigen.sh  ║
# ║  once to populate TASKS_DIR (sparse-clones environments_harbor/).    ║
# ║                                                                        ║
# ║  Every TermiGen Dockerfile FROMs                                      ║
# ║    ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624              ║
# ║  and then installs python3-pip + pytest/pandas/scipy on top. Rather  ║
# ║  than re-doing that 3,500 times we prebuild a single shared base SIF ║
# ║  (tbench_ubuntu24_base.sif) here. The adapter has already rewritten  ║
# ║  each per-task container.def to use Bootstrap: localimage + From:    ║
# ║  ./tbench_ubuntu24_base.sif, so per-task builds layer cheap deltas   ║
# ║  (task payload + any task-specific apt/pip).                          ║
# ╚═══════════════════════════════════════════════════════════════════════╝

set -euo pipefail

# ---- Parameters (edit here) ----
TASKS_DIR="rl_data/output/tasks_termigen"
# Override via env.  Examples:
#   API model:   MODEL="gemini/gemini-3-flash-preview"
#   Local vLLM:  MODEL="hosted_vllm/Qwen/Qwen2.5-Coder-7B-Instruct" \
#                HOSTED_VLLM_API_BASE="http://localhost:8000/v1"
#   Ollama:      MODEL="ollama_chat/qwen2.5-coder:7b" \
#                OLLAMA_API_BASE="http://localhost:11434"
MODEL="${MODEL:-gemini/gemini-3-flash-preview}"
# Solution-sampling harness. See run_generate_solutions_et.sh for the 0515
# rebase rationale; every comparison baseline is now vanillux by default.
HARNESS="${HARNESS:-vanillux}"
# 8 attempts/task = pass@1/4/8 in one run. Uniform across all 5 datasets.
NUM_SOLUTIONS="${NUM_SOLUTIONS:-8}"
# 64 = vanillux convention; the mini-swe-agent prompts need the larger budget.
MAX_ACTIONS="${MAX_ACTIONS:-64}"
# MAX_TOKENS is the per-turn generation cap.  Gemini default 65536; auto-capped
# to VLLM_MAX_LEN-safety_margin when LAUNCH_VLLM=1 (see _vllm_wait_ready_local).
MAX_TOKENS="${MAX_TOKENS:-65536}"
NUM_TASKS=999999
START_AT=0
SOLUTION_TEMPERATURE=0.7
# 600s matches the vanillux reference script (was 60 in the pre-0515
# comparison; insufficient for the heavier TermiGen setup.sh under 8-way
# parallelism — apt + pip contend on the global lock and time out at 60s).
COMMAND_TIMEOUT="${COMMAND_TIMEOUT:-600}"
SHELL_INIT_TIMEOUT=240
SHELL_INIT_ATTEMPTS=3
BUILD_WORKERS=12
BUILD_RETRIES=3
FORCE_RERUN=0
LOG_COMMANDS=0
DISABLE_TERMINAL_LOG=0

# Cost-bounded subsample: 250 tasks (uniform across all 5 datasets).
# Fixed seed=0 so the same 250 TermiGen tasks are picked on every rerun.
SAMPLE_SIZE="${SAMPLE_SIZE:-250}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"

# WORKERS = concurrent TASKS processed at once.  NUM_POOL_WORKERS = concurrent
# solutions/env operations within a single task.  Both env-overridable.
WORKERS="${WORKERS:-12}"
NUM_POOL_WORKERS="${NUM_POOL_WORKERS:-16}"
# --------------------------------

_RUN_TS=$(date -u +%Y%m%d_%H%M%S)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"
mkdir -p logs

# In-job vLLM bring-up. No-op unless LAUNCH_VLLM=1. See run_generate_solutions_et.sh
# header for the rationale; we kick it off pre-SIF-build so weight loading
# overlaps with the (slow) base SIF build, then block on readiness right
# before the solver call.
# shellcheck source=./_vllm_local.sh
source "$SCRIPT_DIR/_vllm_local.sh"
_vllm_start_local

# Docker Hub creds are OPTIONAL for TermiGen because:
#   (a) the base SIF is pulled from ghcr.io, not Docker Hub (we explicitly
#       suppress these creds for that step -- see the `env -u ...` call below;
#       the upstream image is public and ghcr.io will REFUSE Docker-Hub creds
#       at its token endpoint with `DENIED: denied`); and
#   (b) per-task SIFs all use `Bootstrap: localimage` on top of the prebaked
#       base -- see rl_data.comparison.adapters.termigen._rewrite_container_def_to_localimage
#       -- so they hit no registry at all.
# We still keep them exported so `generate_solutions.py`'s inner
# `apptainer build` invocations inherit them harmlessly if they're set.
export APPTAINER_DOCKER_USERNAME="${APPTAINER_DOCKER_USERNAME:-}"
export APPTAINER_DOCKER_PASSWORD="${APPTAINER_DOCKER_PASSWORD:-}"

# Local-model support via litellm env passthrough.
export HOSTED_VLLM_API_BASE="${HOSTED_VLLM_API_BASE:-}"
export OLLAMA_API_BASE="${OLLAMA_API_BASE:-}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-}"
if [[ -n "${HOSTED_VLLM_API_BASE:-}${OLLAMA_API_BASE:-}${OPENAI_API_BASE:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="EMPTY"
fi

export APPTAINER_CACHEDIR="/gpfs/projects/h2lab/osey/apptainer_cache"
export APPTAINER_TMPDIR="/tmp/apptainer_tmp"
mkdir -p "$APPTAINER_TMPDIR"

# Keep apptainer instance logs off GPFS; heal dangling symlinks left behind
# when /tmp was cleaned between runs.
mkdir -p /tmp/apptainer_instances
if [ ! -L "$HOME/.apptainer/instances" ]; then
  rm -rf "$HOME/.apptainer/instances"
  ln -s /tmp/apptainer_instances "$HOME/.apptainer/instances"
fi

# ---- TermiGen shared base SIF bootstrap ---------------------------------
# Every TermiGen-derived container.def now uses `Bootstrap: localimage` +
# `From: ./tbench_ubuntu24_base.sif` (see termigen adapter). Build that base
# here once, pre-installing the common deps every task's Dockerfile asks for
# (python3-pip, pytest, pandas, scipy) on top of the upstream t-bench image.
#
# LEVER 3 (from the ET run): keep /var/lib/apt/lists/* populated so per-task
# `apt-get update` calls turn into fast HEAD/InRelease checks instead of
# re-downloading the full index every build.
#
# Bump BASE_TBENCH_SIF_VERSION to invalidate when the recipe below changes.
BASE_TBENCH_SIF="$PROJECT_ROOT/tbench_ubuntu24_base.sif"
BASE_TBENCH_SIF_VERSION="v1-tbench-ubuntu24.04+py+pytest+pandas+scipy+aptlists"
BASE_TBENCH_SIF_MARK="${BASE_TBENCH_SIF}.version"
TBENCH_UPSTREAM_IMAGE="ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624"

_need_base_build=0
if [ ! -f "$BASE_TBENCH_SIF" ]; then
  _need_base_build=1
  _reason="missing"
elif [ ! -f "$BASE_TBENCH_SIF_MARK" ] || \
     [ "$(cat "$BASE_TBENCH_SIF_MARK" 2>/dev/null)" != "$BASE_TBENCH_SIF_VERSION" ]; then
  _need_base_build=1
  _reason="stale (want ${BASE_TBENCH_SIF_VERSION}, got $(cat "$BASE_TBENCH_SIF_MARK" 2>/dev/null || echo none))"
fi

if [ "$_need_base_build" = "1" ]; then
  echo "=== TermiGen base SIF ${_reason}; building $BASE_TBENCH_SIF ==="
  BASE_TBENCH_DEF="$(mktemp --suffix=.def)"
  trap 'rm -f "$BASE_TBENCH_DEF"' EXIT
  cat > "$BASE_TBENCH_DEF" <<EOF
Bootstrap: docker
From: ${TBENCH_UPSTREAM_IMAGE}

%post
    set -e
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        ca-certificates
    # Create /usr/bin/python -> python3 symlink that most TermiGen Dockerfiles
    # redundantly add. Harmless if the base already has it (ln -sf is idempotent).
    ln -sf /usr/bin/python3 /usr/bin/python
    # Pre-install the pytest + scientific-python trio that roughly 90% of
    # TermiGen tasks re-install on top. Versions match the most common pins
    # observed in upstream Dockerfiles (pytest==8.4.1 pandas==2.3.2 scipy==1.16.1).
    # --break-system-packages is required on Ubuntu 24.04's PEP 668-flagged pip.
    python3 -m pip install --no-cache-dir --break-system-packages \
        pytest==8.4.1 pandas==2.3.2 scipy==1.16.1 numpy scikit-learn
    apt-get clean
    # tmax convention: every base SIF has /home/user present so the harness's
    # generate_solutions._patch_def_chmod can inject its chmod 755 /home/user
    # unconditionally across adapters.
    mkdir -p /home/user
    chmod 755 /home/user
    # Intentionally keep /var/lib/apt/lists/* populated (Lever 3).

%environment
    export DEBIAN_FRONTEND=noninteractive

%labels
    Author rl_data-termigen-bootstrap
    Description "t-bench/ubuntu-24-04 + python3-pip + pytest/pandas/scipy/numpy/sklearn, prebaked for per-task TermiGen builds."
EOF
  # The upstream image is PUBLIC but ghcr.io's token endpoint rejects any
  # Docker-Hub-shaped credentials we might have exported for other datasets
  # (e.g. APPTAINER_DOCKER_USERNAME=<dockerhub-user> + Docker-Hub PAT), so we
  # scrub APPTAINER_DOCKER_USERNAME/PASSWORD for this single build -- ghcr.io
  # happily serves the manifest anonymously. If you ever need to pull a
  # PRIVATE ghcr.io image, set APPTAINER_DOCKER_USERNAME=<github-user> and
  # APPTAINER_DOCKER_PASSWORD=<github-PAT-with-read:packages> before rerunning
  # and remove the `env -u` below.
  if ! env -u APPTAINER_DOCKER_USERNAME -u APPTAINER_DOCKER_PASSWORD \
        apptainer build --force "$BASE_TBENCH_SIF" "$BASE_TBENCH_DEF"; then
    echo "ERROR: failed to build $BASE_TBENCH_SIF (TermiGen per-task builds require this)." >&2
    echo "Hint: the base image (${TBENCH_UPSTREAM_IMAGE}) is public on ghcr.io;" >&2
    echo "      if the anonymous pull is rate-limited, retry after a minute, or" >&2
    echo "      set APPTAINER_DOCKER_USERNAME=<github-user> +" >&2
    echo "      APPTAINER_DOCKER_PASSWORD=<GitHub PAT with read:packages> and rerun." >&2
    exit 1
  fi
  echo "$BASE_TBENCH_SIF_VERSION" > "$BASE_TBENCH_SIF_MARK"
  rm -f "$BASE_TBENCH_DEF"
  trap - EXIT
  echo "=== TermiGen base SIF ready: $BASE_TBENCH_SIF (version=${BASE_TBENCH_SIF_VERSION}) ==="
else
  echo "=== TermiGen base SIF already present: $BASE_TBENCH_SIF (version=${BASE_TBENCH_SIF_VERSION}, skipping build) ==="
fi

_vllm_wait_ready_local

_MODEL_TAG=$(echo "$MODEL" | tr '/' '_')
# Include $HARNESS in the log filename so vanillux reruns don't clobber legacy
# bash-harness logs from earlier comparison runs.
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

echo "=== TermiGen comparison run: MODEL=${MODEL}, HARNESS=${HARNESS}, WORKERS=${WORKERS}, NUM_SOLUTIONS=${NUM_SOLUTIONS}, SAMPLE_SIZE=${SAMPLE_SIZE} ==="
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
