#!/bin/bash
#SBATCH --job-name=rl-cmp-our-vlx
#SBATCH --output=logs/cmp_our_vlx_%j.out
#SBATCH --error=logs/cmp_our_vlx_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Solution-generation under the VANILLUX harness against our rebased  ║
# ║  reference corpus, tasks_skill_tax_combined_20260506_legacy10k_new5k ║
# ║  (14,601 tasks = 9,465 legacy + 5,136 v2). This is the "ours" side   ║
# ║  of the 0515 5-way dataset comparison.                                ║
# ║                                                                       ║
# ║  Same uniform config as the four baseline scripts in this directory  ║
# ║  (run_generate_solutions_{et,openthoughts,termigen,terminaltraj}.sh) ║
# ║  so head-to-head pass@1/4/8 is apples-to-apples across all 5         ║
# ║  datasets:                                                            ║
# ║                                                                       ║
# ║    MODEL=gemini/gemini-3-flash-preview                                ║
# ║    HARNESS=vanillux                                                   ║
# ║    NUM_SOLUTIONS=8        MAX_ACTIONS=64                              ║
# ║    MAX_TOKENS=65536       SOLUTION_TEMPERATURE=0.7                    ║
# ║    COMMAND_TIMEOUT=600    SHELL_INIT_TIMEOUT=240                      ║
# ║    SAMPLE_SIZE=250        SAMPLE_SEED=0  (uniform i.i.d. subsample)   ║
# ║                                                                       ║
# ║  Naming: <task>/solutions/<MODEL_TAG>_vanillux_summary.json           ║
# ║  (legacy <MODEL_TAG>_summary.json bash-harness files, if any, stay   ║
# ║  alongside since filenames are disjoint).                            ║
# ║                                                                       ║
# ║  Sister scripts (same harness, different corpus):                     ║
# ║    * run_generate_solutions_skill_tax_combined_2.5k_vanillux_gemini.sh║
# ║      — the legacy 2.5k balanced combined corpus (pre-0515 reference). ║
# ║                                                                       ║
# ║  Required env vars:                                                   ║
# ║    GEMINI_API_KEY             — Google AI Studio key                  ║
# ║    APPTAINER_DOCKER_USERNAME  — Docker Hub creds                      ║
# ║    APPTAINER_DOCKER_PASSWORD                                          ║
# ║                                                                       ║
# ║  SBATCH allocation — pick the line that matches your wall budget.    ║
# ║  GPUs are reserved only because they bring CPUs+RAM with them on h200║
# ║  nodes; vanillux+Gemini does NOT use GPU at all.                     ║
# ║    8 GPUs: 64 CPUs / ~960 GB RAM  → WORKERS=24 NUM_SOLUTIONS=8 = 192 ║
# ║    4 GPUs: 32 CPUs / ~480 GB RAM  → WORKERS=12 NUM_SOLUTIONS=8 = 96  ║
# ║                                                                       ║
# ║  With SAMPLE_SIZE=250 and NUM_SOLUTIONS=8 the run is only 2000        ║
# ║  trajectories so 4 GPUs is fine; bump to 8 only if you also bump     ║
# ║  SAMPLE_SIZE for a deeper sweep.                                      ║
# ╚═══════════════════════════════════════════════════════════════════════╝

# ── Option A: 4 GPUs (default for the 250-task uniform comparison) ──
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
WORKERS="${WORKERS:-12}"

# ── Option B: 8 GPUs (faster; only worth it if you also bump SAMPLE_SIZE) ──
# #SBATCH --gres=gpu:h200:8
# #SBATCH --cpus-per-task=64
# #SBATCH --mem=960G
# WORKERS="${WORKERS:-24}"

set -euo pipefail

# ---- Parameters (edit here) ----
# Default to the 0515 rebased reference corpus (legacy 10k ∪ new v2 5k).
TASKS_DIR="${TASKS_DIR:-rl_data/output/tasks_skill_tax_combined_20260506_legacy10k_new5k}"

HARNESS="${HARNESS:-vanillux}"

MODEL="${MODEL:-gemini/gemini-3-flash-preview}"

# 8 attempts/task → pass@1/4/8 in one run. Uniform across all 5 datasets.
NUM_SOLUTIONS="${NUM_SOLUTIONS:-8}"

# 64 = vanillux convention (mini-swe-agent "Recommended Workflow" budget).
MAX_ACTIONS="${MAX_ACTIONS:-64}"

# Gemini context is 1M tokens; 65536 matches the legacy 10k Gemini script.
MAX_TOKENS="${MAX_TOKENS:-65536}"
NUM_TASKS="${NUM_TASKS:-999999}"     # cap on tasks processed (sharding hook)
START_AT="${START_AT:-0}"            # skip first N tasks (sharding hook)
SOLUTION_TEMPERATURE=0.7
COMMAND_TIMEOUT="${COMMAND_TIMEOUT:-600}"
SHELL_INIT_TIMEOUT=240
SHELL_INIT_ATTEMPTS=3
BUILD_WORKERS="${BUILD_WORKERS:-8}"
BUILD_RETRIES=3
BASE_SIFS_DIR="${BASE_SIFS_DIR:-rl_data/containers}"
FORCE_RERUN="${FORCE_RERUN:-0}"
LOG_COMMANDS=0
DISABLE_TERMINAL_LOG=0

# Uniform-comparison subsample: 250 tasks i.i.d. with fixed seed=0, matching
# the four baseline scripts in this directory.
SAMPLE_SIZE="${SAMPLE_SIZE:-250}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"

# NUM_POOL_WORKERS = concurrent solutions / shell ops within a single task.
# Must be >= NUM_SOLUTIONS for full parallelism within a task.
NUM_POOL_WORKERS="${NUM_POOL_WORKERS:-16}"
# --------------------------------

_RUN_TS=$(date -u +%Y%m%d_%H%M%S)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"
mkdir -p logs

# ---- Pre-flight: dangling-symlink check ----
# Combined corpora are SYMLINK-VIEWS over their source dirs (legacy 10k +
# v2 5k). A rename of either source dir leaves dangling symlinks here that
# generate_solutions silently filters out (is_dir() returns False on
# dangling links), making the run process FEWER tasks than NUM_TASKS
# suggests. Aborting here is cheap; the alternative is wasting hours of
# Gemini-API spend on a partial corpus.
if [[ -d "$TASKS_DIR" ]]; then
  _broken=$(find "$TASKS_DIR" -maxdepth 1 -type l ! -exec test -e {} \; -print 2>/dev/null | wc -l)
  if (( _broken > 0 )); then
    echo "ERROR: $_broken dangling task_* symlink(s) in $TASKS_DIR." >&2
    echo "       A source corpus dir was likely renamed/moved after the combine." >&2
    echo "       Fix: re-run combine with --force pointing at the new source path." >&2
    exit 2
  fi
fi

# Gemini API key — required.
: "${GEMINI_API_KEY:?Set GEMINI_API_KEY before running (Google AI Studio key)}"
export GEMINI_API_KEY

# Apptainer Docker Hub creds — required for skill-tax per-task base pulls.
export APPTAINER_DOCKER_USERNAME="${APPTAINER_DOCKER_USERNAME:?Set APPTAINER_DOCKER_USERNAME before running}"
export APPTAINER_DOCKER_PASSWORD="${APPTAINER_DOCKER_PASSWORD:?Set APPTAINER_DOCKER_PASSWORD before running}"

# Clear local-vLLM passthrough vars so litellm uses Gemini's native endpoint,
# not whatever stale HOSTED_VLLM_API_BASE is in the shell.
unset HOSTED_VLLM_API_BASE OLLAMA_API_BASE OPENAI_API_BASE OPENAI_API_KEY 2>/dev/null || true

export APPTAINER_CACHEDIR="/gpfs/projects/h2lab/osey/apptainer_cache"
export APPTAINER_TMPDIR="/tmp/apptainer_tmp"
mkdir -p "$APPTAINER_TMPDIR"

mkdir -p /tmp/apptainer_instances
if [ ! -L "$HOME/.apptainer/instances" ]; then
  rm -rf "$HOME/.apptainer/instances"
  ln -s /tmp/apptainer_instances "$HOME/.apptainer/instances"
fi

_MODEL_TAG=$(echo "$MODEL" | tr '/' '_')
TERMINAL_LOG="${TASKS_DIR}/logs/${_MODEL_TAG}_${HARNESS}_${_RUN_TS}.log"

EXTRA_ARGS=()
if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--force-rerun)
fi
if [[ "${LOG_COMMANDS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--log-commands)
fi
if [[ -n "${BASE_SIFS_DIR:-}" ]]; then
  EXTRA_ARGS+=(--base-sifs-dir "$BASE_SIFS_DIR")
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

echo "=== Skill-tax (combined legacy10k+new5k) vanillux comparison run ==="
echo "===   MODEL=${MODEL}, HARNESS=${HARNESS}, MAX_ACTIONS=${MAX_ACTIONS}, NUM_SOLUTIONS=${NUM_SOLUTIONS}"
echo "===   Tasks dir: ${TASKS_DIR}  (SAMPLE_SIZE=${SAMPLE_SIZE}, SAMPLE_SEED=${SAMPLE_SEED})"
echo "===   Concurrent containers: $(( WORKERS * NUM_SOLUTIONS ))  (WORKERS=${WORKERS} × NUM_SOLUTIONS=${NUM_SOLUTIONS})"

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
