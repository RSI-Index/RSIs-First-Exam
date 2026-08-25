#!/usr/bin/env bash
#SBATCH --job-name=rl-gen-stx-comb-2.5k-thinking
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
# This cluster binds CPUs to GPUs (cpus-per-gpu policy); --cpus-per-task=8 with
# --gpus-per-node=8 yields 64 CPUs / node total. 8 is the per-GPU max — going
# higher conflicts with the binding. Confirmed via sacct on a prior run:
#   AllocTRES: cpu=64,gres/gpu:h200=8,mem=...
#SBATCH --cpus-per-task=8
#SBATCH --mem=1440G
#SBATCH --time=24:00:00
# Single combined output file (stderr merges into stdout) — matches the
# proven SFT-script pattern. %j: jobid for plain submits,
# "<arrayjobid>_<arraytaskid>" for `--array=...` submits — works for both
# without leaking the uint32_t-(-1) sentinel (4294967294) into filenames.
#SBATCH --output=/gpfs/scrubbed/osey/tmax/logs/gen_sol_stx_comb_2.5k_thinking_%j.out

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Solution-generation on the COMBINED 2.5k SFT corpus, THINKING-MODE   ║
# ║  variant.                                                             ║
# ║                                                                       ║
# ║  Identical corpus + harness + model to                                ║
# ║    rl_data/scripts/generate_solutions/run_generate_solutions_skill_tax_combined_2.5k.sh
# ║  but with Qwen3's reasoning traces (`<think>...</think>`) ENABLED at  ║
# ║  sample time. Per-task summary files are written under a SEPARATE     ║
# ║  filename so thinking-mode and non-thinking trajectories coexist on   ║
# ║  the same task dirs without overwriting:                              ║
# ║                                                                       ║
# ║    non-thinking : <MODEL_TAG>_vanillux_summary.json                    ║
# ║    thinking-on  : <MODEL_TAG>_vanillux_thinking_summary.json          ║
# ║                                                                       ║
# ║  This is critical because the legacy 2.5k corpus already has          ║
# ║  *_vanillux_summary.json files in place from the non-thinking run;    ║
# ║  re-running here with the same filename would clobber them.           ║
# ║                                                                       ║
# ║  How thinking is turned on:                                           ║
# ║    1. VLLM_DISABLE_THINKING=0 (overrides _vllm_local.sh's default-1   ║
# ║       for Qwen3-family models so the helper does NOT export           ║
# ║       LITELLM_EXTRA_BODY_JSON='{...enable_thinking:false}').          ║
# ║    2. After the helper finishes, we explicitly export                 ║
# ║       LITELLM_EXTRA_BODY_JSON='{"chat_template_kwargs":               ║
# ║       {"enable_thinking": true, "preserve_thinking": true}}'          ║
# ║       so litellm's chat-completions request body carries both kwargs ║
# ║       through to vLLM:                                                ║
# ║         * enable_thinking=true   — vLLM emits <think>...</think>      ║
# ║           blocks; the qwen3 reasoning-parser (already enabled in      ║
# ║           _vllm_local.sh) splits them into a separate                 ║
# ║           reasoning_content field on the response.                    ║
# ║         * preserve_thinking=true — Qwen3.6's chat template KEEPS      ║
# ║           historical assistant turns' thinking blocks in the prompt   ║
# ║           when rendering subsequent turns. Without it, only the most  ║
# ║           recent assistant turn keeps its <think> in the prompt       ║
# ║           ("interleaved thinking" — Qwen3.6's default), which for     ║
# ║           multi-turn agent loops means the model loses its own        ║
# ║           chain-of-thought across turns. The Qwen3.6 model card       ║
# ║           explicitly recommends preserve_thinking=true for agent      ║
# ║           scenarios; the Qwen-Agent example sets both kwargs.         ║
# ║           NOTE: preserve_thinking is a Qwen3.6 CHAT-TEMPLATE feature  ║
# ║           (not Qwen3 / Qwen3.5). Setting it on a non-Qwen3.6 model    ║
# ║           is a silent no-op (chat template ignores unknown kwargs).   ║
# ║           Our harness already round-trips reasoning_content via       ║
# ║           response.choices[0].message.model_dump() into messages      ║
# ║           history, so the data is on the wire — preserve_thinking    ║
# ║           is the chat-template-side switch that actually renders it. ║
# ║    3. --thinking is passed to rl_data.generate_solutions so the per-  ║
# ║       task summary filename gets the `_thinking` infix.               ║
# ║                                                                       ║
# ║  Throughput notes:                                                    ║
# ║    Reasoning traces inflate per-turn output by ~3-10× (Qwen3's        ║
# ║    "Recommended Workflow" thinking traces routinely run 4-8K tokens). ║
# ║    We therefore:                                                      ║
# ║      - bump MAX_TOKENS from 4096 -> 16384 (per-turn cap)              ║
# ║      - drop WORKERS from 24 -> 12 to halve concurrency, since each   ║
# ║        thinking-on turn uses meaningfully more KV (and each thinking  ║
# ║        turn is significantly slower wall-clock).                      ║
# ║    Even with these knobs, expect ~3-4× the wall time of the non-     ║
# ║    thinking run, hence the 4-shard layout (vs the 2-shard layout     ║
# ║    documented on the non-thinking sister script).                    ║
# ║                                                                       ║
# ║  4-node sharding (default config — submit ALL FOUR sbatch commands   ║
# ║  to start the run; each shard has its own job ID and log files):     ║
# ║                                                                       ║
# ║    NUM_SHARDS=4 SHARD_INDEX=0 sbatch ./run_generate_solutions_skill_tax_combined_2.5k_thinking.sh
# ║    NUM_SHARDS=4 SHARD_INDEX=1 sbatch ./run_generate_solutions_skill_tax_combined_2.5k_thinking.sh
# ║    NUM_SHARDS=4 SHARD_INDEX=2 sbatch ./run_generate_solutions_skill_tax_combined_2.5k_thinking.sh
# ║    NUM_SHARDS=4 SHARD_INDEX=3 sbatch ./run_generate_solutions_skill_tax_combined_2.5k_thinking.sh
# ║                                                                       ║
# ║  Or (cleaner) as a SLURM array job — one submit, four scheduled       ║
# ║  child jobs, SHARD_INDEX/NUM_SHARDS auto-derived from                 ║
# ║  SLURM_ARRAY_TASK_ID/SLURM_ARRAY_TASK_COUNT:                          ║
# ║                                                                       ║
# ║    sbatch --array=0-3 ./run_generate_solutions_skill_tax_combined_2.5k_thinking.sh
# ║                                                                       ║
# ║  The script computes START_AT/NUM_TASKS deterministically from        ║
# ║  SHARD_INDEX/NUM_SHARDS and the corpus size (2500 tasks / 4 = 625 per ║
# ║  shard, with the last shard absorbing any remainder). Each task's    ║
# ║  solutions/ subdir is written by exactly one shard — no contention.   ║
# ║                                                                       ║
# ║  Required env vars (must be set BEFORE `sbatch` so SLURM's default   ║
# ║  --export=ALL propagates them, OR placed in $TMAX_SECRETS_FILE which ║
# ║  the script auto-sources at start):                                  ║
# ║    APPTAINER_DOCKER_USERNAME, APPTAINER_DOCKER_PASSWORD              ║
# ║    HF_TOKEN                  (for hf model pull if cache miss)        ║
# ║    GEMINI_API_KEY            (only if you swap MODEL to gemini/...)   ║
# ║    OPENAI_API_KEY            (only if you swap MODEL to openai/...)   ║
# ║                                                                       ║
# ║  Recommended workflow:                                                ║
# ║      # one-time setup: store secrets in a chmod-600 file              ║
# ║      cat > ~/.tmax_secrets <<'EOF'                                    ║
# ║      export APPTAINER_DOCKER_USERNAME=...                             ║
# ║      export APPTAINER_DOCKER_PASSWORD=...                             ║
# ║      export HF_TOKEN=hf_...                                           ║
# ║      export GEMINI_API_KEY=...                                        ║
# ║      EOF                                                              ║
# ║      chmod 600 ~/.tmax_secrets                                        ║
# ║                                                                       ║
# ║      # every submit: source secrets THEN sbatch (default --export=ALL ║
# ║      # propagates exported vars into the job's env)                  ║
# ║      source ~/.tmax_secrets                                           ║
# ║      sbatch --array=0-3 \                                            ║
# ║          ./run_generate_solutions_skill_tax_combined_2.5k_thinking.sh ║
# ║                                                                       ║
# ║  Override TMAX_SECRETS_FILE to point at a different file:            ║
# ║      TMAX_SECRETS_FILE=~/.my_secrets sbatch --array=0-3 ...           ║
# ║                                                                       ║
# ║  Disabling thinking (use the non-thinking sister script instead, but  ║
# ║  THINKING=0 here is supported as an opt-out for A/B comparisons —    ║
# ║  output filename then loses the `_thinking` infix and clobbers the    ║
# ║  non-thinking run, so prefer the sister script unless you know what  ║
# ║  you're doing):                                                       ║
# ║      THINKING=0 sbatch ./run_generate_solutions_skill_tax_combined_2.5k_thinking.sh
# ╚═══════════════════════════════════════════════════════════════════════╝

set -euo pipefail

# ── Auto-source secrets file (best-effort) ─────────────────────────────
# When the user remembered to `source ~/.tmax_secrets` before sbatch and
# kept SLURM's default --export=ALL, the env vars are already in scope and
# this is a no-op (re-sourcing is idempotent for `export VAR=value` lines).
# When the user forgot, this rescues the run as long as the file exists on
# the compute node's filesystem (GPFS-mounted $HOME on this cluster).
_TMAX_SECRETS_FILE="${TMAX_SECRETS_FILE:-$HOME/.tmax_secrets}"
if [[ -r "$_TMAX_SECRETS_FILE" ]]; then
  echo "=== Sourcing secrets from $_TMAX_SECRETS_FILE ==="
  # shellcheck disable=SC1090
  set +u
  source "$_TMAX_SECRETS_FILE"
  set -u
fi

# ---- Parameters (edit here) ----
TASKS_DIR="${TASKS_DIR:-rl_data/output/tasks_skill_tax_20260505_2.2k_combined_balanced}"

# This script is local-vLLM ONLY.
export LAUNCH_VLLM="${LAUNCH_VLLM:-1}"
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3.6-27B}"
MODEL="${MODEL:-hosted_vllm/${VLLM_MODEL}}"

# THINKING=1 (default) tells the solver to:
#   1. send `chat_template_kwargs={"enable_thinking": true, "preserve_thinking": true}`
#      in every chat request body (so vLLM emits <think>...</think> blocks
#      AND the chat template keeps historical thinking traces in the prompt
#      across turns — see the header for why preserve_thinking matters
#      for agent loops), AND
#   2. tag the per-task summary filename with a `_thinking` infix so it
#      doesn't clobber the non-thinking sister run on the same task dir.
# THINKING=0 reverts to the non-thinking behaviour (and reverts the
# filename — risk of clobbering the non-thinking sister run; A/B only).
THINKING="${THINKING:-1}"

# PRESERVE_THINKING controls Qwen3.6's chat-template `preserve_thinking`
# kwarg independently of THINKING. Defaults to whatever THINKING is, since
# preserving thinking only matters when thinking is enabled in the first
# place. Set to 0 to A/B test interleaved thinking (Qwen3.6 default) vs
# preserved thinking on the same corpus.
# Silent no-op for non-Qwen3.6 models (their chat templates ignore the
# unknown kwarg).
PRESERVE_THINKING="${PRESERVE_THINKING:-$THINKING}"

NUM_SOLUTIONS="${NUM_SOLUTIONS:-8}"

# Vanillux harness (matches the non-thinking sister script). Kept as an
# env override only — the bash harness has not been validated with thinking
# mode here.
HARNESS="${HARNESS:-vanillux}"
MAX_ACTIONS="${MAX_ACTIONS:-64}"

# MAX_TOKENS: per-turn output cap. Thinking traces inflate per-turn output
# by ~3-10× (Qwen3's "Recommended Workflow" reasoning traces routinely run
# 4-8K tokens BEFORE the actual tool-call payload). 16384 leaves headroom
# for trace + final action without bumping into the auto-cap (vllm_max_len
# / 4 = 131072 / 4 = 32768).
MAX_TOKENS="${MAX_TOKENS:-16384}"

# ── Sharding ───────────────────────────────────────────────────────────
# Honour SLURM array indexing first, fall back to env-only sharding so
# the script also works when launched manually (4 separate sbatch
# invocations, each with NUM_SHARDS=4 SHARD_INDEX=N).
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] && [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]]; then
  : "${NUM_SHARDS:=$SLURM_ARRAY_TASK_COUNT}"
  : "${SHARD_INDEX:=$SLURM_ARRAY_TASK_ID}"
fi
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"

if (( SHARD_INDEX < 0 || SHARD_INDEX >= NUM_SHARDS )); then
  echo "ERROR: SHARD_INDEX=$SHARD_INDEX must be in [0,$NUM_SHARDS)" >&2
  exit 2
fi

# Compute START_AT / NUM_TASKS from shard layout, but allow explicit env
# overrides to win (lets a user re-run a single tail-end shard with a
# tighter window without recomputing).
if [[ -z "${START_AT:-}" || -z "${NUM_TASKS:-}" ]]; then
  if (( NUM_SHARDS > 1 )); then
    # Count tasks once on rank 0 — combined corpus is a flat dir of
    # task_*/ entries (some symlinks; that's fine, find -maxdepth 1 -type
    # l counts those too because they resolve to dirs after deref).
    _TOTAL_TASKS=$(find "$TASKS_DIR" -mindepth 1 -maxdepth 1 \
                    \( -type d -o -type l \) -name 'task_*' 2>/dev/null | wc -l)
    if (( _TOTAL_TASKS == 0 )); then
      echo "ERROR: no task_* entries under $TASKS_DIR (cwd=$(pwd))" >&2
      exit 2
    fi
    _PER_SHARD=$(( (_TOTAL_TASKS + NUM_SHARDS - 1) / NUM_SHARDS ))   # ceil
    START_AT="${START_AT:-$(( SHARD_INDEX * _PER_SHARD ))}"
    # last shard absorbs the remainder
    if (( SHARD_INDEX == NUM_SHARDS - 1 )); then
      NUM_TASKS="${NUM_TASKS:-$(( _TOTAL_TASKS - START_AT ))}"
    else
      NUM_TASKS="${NUM_TASKS:-$_PER_SHARD}"
    fi
    echo "=== Shard $SHARD_INDEX/$NUM_SHARDS: START_AT=$START_AT NUM_TASKS=$NUM_TASKS (of $_TOTAL_TASKS total)"
  else
    START_AT="${START_AT:-0}"
    NUM_TASKS="${NUM_TASKS:-999999}"
  fi
fi

SOLUTION_TEMPERATURE=0.7
COMMAND_TIMEOUT=600
SHELL_INIT_TIMEOUT=240
SHELL_INIT_ATTEMPTS=3
BUILD_WORKERS=4
BUILD_RETRIES=3
BASE_SIFS_DIR="${BASE_SIFS_DIR:-rl_data/containers}"
FORCE_RERUN=0
LOG_COMMANDS=0
DISABLE_TERMINAL_LOG=0

SAMPLE_SIZE="${SAMPLE_SIZE:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"

# WORKERS = concurrent TASKS at once. Lowered from 24 -> 12 vs the non-
# thinking sister script: thinking traces 3-4× the per-turn token output
# AND lengthen each turn's wall-clock, so the vLLM throughput ceiling
# moves down. 12 × 8 sols = 96 concurrent containers (1.5× CPU on 64-CPU
# nodes), still keeps the apptainer side healthy without thrashing.
WORKERS="${WORKERS:-12}"
NUM_POOL_WORKERS="${NUM_POOL_WORKERS:-16}"

export VLLM_TP="${VLLM_TP:-1}"
export VLLM_MAX_LEN="${VLLM_MAX_LEN:-131072}"
export VLLM_PREFIX_CACHE="${VLLM_PREFIX_CACHE:-1}"

# ── Thinking-mode toggle for the vLLM helper ───────────────────────────
# _vllm_local.sh defaults VLLM_DISABLE_THINKING=1 for Qwen3 models; we
# override here so its post-readiness hook does NOT export
# LITELLM_EXTRA_BODY_JSON='{...enable_thinking:false}'. We then set the
# enable_thinking=true variant ourselves below (after _vllm_wait_ready_local
# returns), guarding against the helper's logic running first.
if [[ "$THINKING" == "1" ]]; then
  export VLLM_DISABLE_THINKING="${VLLM_DISABLE_THINKING:-0}"
else
  export VLLM_DISABLE_THINKING="${VLLM_DISABLE_THINKING:-1}"
fi
# --------------------------------

_RUN_TS=$(date -u +%Y%m%d_%H%M%S)

# Resolve PROJECT_ROOT in a way that works under BOTH sbatch and `bash
# <script>`. Under sbatch, SLURM copies the script body to
# /var/spool/slurmd/job<id>/slurm_script and runs it from there, so
# ${BASH_SOURCE[0]} no longer points at the original file location ⇒
# SCRIPT_DIR/../../.. resolves to /var/spool (or /) and every subsequent
# relative path breaks. SLURM sets $SLURM_SUBMIT_DIR to the directory
# `sbatch` / `salloc` was invoked from — that IS the tmax root in our
# canonical workflow, so we prefer it *when it actually contains the
# repo* (marker: rl_data/scripts/comparison/_vllm_local.sh).
#
# Interactive pitfall: if you `salloc` from e.g. open-instruct, then
# SLURM_SUBMIT_DIR points there forever inside the allocation even after
# you `cd` to tmax. Blindly trusting it would break. We therefore only
# adopt SLURM_SUBMIT_DIR when the marker file exists; otherwise we fall
# back to the path derived from this script's real location on disk.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PROJECT_ROOT_FROM_SCRIPT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
if [[ -n "${TMAX_PROJECT_ROOT:-}" ]] && [[ -r "$TMAX_PROJECT_ROOT/rl_data/scripts/comparison/_vllm_local.sh" ]]; then
  PROJECT_ROOT="$TMAX_PROJECT_ROOT"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && [[ -r "$SLURM_SUBMIT_DIR/rl_data/scripts/comparison/_vllm_local.sh" ]]; then
  PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
  PROJECT_ROOT="$_PROJECT_ROOT_FROM_SCRIPT"
fi

# Sanity check: bail loudly if the resolved root doesn't actually contain
# the helper script we're about to source — much friendlier than a cryptic
# `source: file not found` deeper in the run.
if [[ ! -r "$PROJECT_ROOT/rl_data/scripts/comparison/_vllm_local.sh" ]]; then
  echo "ERROR: PROJECT_ROOT=$PROJECT_ROOT does not contain rl_data/scripts/comparison/_vllm_local.sh." >&2
  echo "       * sbatch/salloc: launch from the tmax repo root, OR unset SLURM_SUBMIT_DIR and run again from tmax," >&2
  echo "         OR set TMAX_PROJECT_ROOT=/path/to/tmax explicitly (advanced)." >&2
  exit 2
fi

COMPARISON_DIR="$PROJECT_ROOT/rl_data/scripts/comparison"

cd "$PROJECT_ROOT"
mkdir -p logs

# ── Pre-flight: dangling-symlink check ─────────────────────────────────
if [[ -d "$TASKS_DIR" ]]; then
  _broken=$(find "$TASKS_DIR" -maxdepth 1 -type l ! -exec test -e {} \; -print 2>/dev/null | wc -l)
  if (( _broken > 0 )); then
    echo "ERROR: $_broken dangling task_* symlink(s) in $TASKS_DIR." >&2
    echo "       A source corpus dir was likely renamed/moved after the combine." >&2
    echo "       Fix: re-run combine with --force pointing at the new source path." >&2
    exit 2
  fi
fi

# In-job vLLM bring-up.
# shellcheck source=../comparison/_vllm_local.sh
source "$COMPARISON_DIR/_vllm_local.sh"
_vllm_start_local

export APPTAINER_DOCKER_USERNAME="${APPTAINER_DOCKER_USERNAME:?Set APPTAINER_DOCKER_USERNAME before running (or put it in \$TMAX_SECRETS_FILE)}"
export APPTAINER_DOCKER_PASSWORD="${APPTAINER_DOCKER_PASSWORD:?Set APPTAINER_DOCKER_PASSWORD before running (or put it in \$TMAX_SECRETS_FILE)}"

export HOSTED_VLLM_API_BASE="${HOSTED_VLLM_API_BASE:-}"
export OLLAMA_API_BASE="${OLLAMA_API_BASE:-}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-}"
if [[ -n "${HOSTED_VLLM_API_BASE:-}${OLLAMA_API_BASE:-}${OPENAI_API_BASE:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="EMPTY"
fi

export APPTAINER_CACHEDIR="/gpfs/projects/h2lab/osey/apptainer_cache"
export APPTAINER_TMPDIR="/tmp/apptainer_tmp"
mkdir -p "$APPTAINER_TMPDIR"

mkdir -p /tmp/apptainer_instances
if [ -L "$HOME/.apptainer/instances" ] && [ ! -d "$HOME/.apptainer/instances" ]; then
  mkdir -p /tmp/apptainer_instances
fi
if [ ! -L "$HOME/.apptainer/instances" ]; then
  rm -rf "$HOME/.apptainer/instances"
  ln -s /tmp/apptainer_instances "$HOME/.apptainer/instances"
fi

( while kill -0 $$ 2>/dev/null; do
    mkdir -p /tmp/apptainer_instances 2>/dev/null
    sleep 5
  done ) >/dev/null 2>&1 &
_APPTAINER_INSTANCES_WATCHDOG_PID=$!
disown "$_APPTAINER_INSTANCES_WATCHDOG_PID" 2>/dev/null || true

# Block until vLLM is ready. May (re)export LITELLM_EXTRA_BODY_JSON with
# enable_thinking=false based on VLLM_DISABLE_THINKING; we re-set it
# ourselves immediately after to enforce thinking ON.
_vllm_wait_ready_local

# Force enable_thinking=true (the helper would have set it to false if
# VLLM_DISABLE_THINKING=1 and possibly cleared it if =0; we want a single
# source of truth here regardless of helper version). The vLLM server
# was already started with --reasoning-parser qwen3 by the helper, so
# the <think>...</think> blocks land in response.choices[].message.reasoning_content
# and get persisted by the harness alongside content.
#
# We assemble chat_template_kwargs as a small object built from the
# THINKING / PRESERVE_THINKING knobs so each kwarg appears in the request
# body iff the user actually opted into it. This avoids accidentally
# rendering preserve_thinking=false (which is the chat template's
# default; we just don't want to send it explicitly when the user opted
# out, to keep the request body minimal and to leave room for the chat
# template's own default behavior to evolve).
if [[ "$THINKING" == "1" ]]; then
  if [[ "$PRESERVE_THINKING" == "1" ]]; then
    export LITELLM_EXTRA_BODY_JSON='{"chat_template_kwargs": {"enable_thinking": true, "preserve_thinking": true}}'
  else
    export LITELLM_EXTRA_BODY_JSON='{"chat_template_kwargs": {"enable_thinking": true}}'
  fi
fi

_MODEL_TAG=$(echo "$MODEL" | tr '/' '_')
_THINKING_TAG=""
if [[ "$THINKING" == "1" ]]; then
  _THINKING_TAG="_thinking"
fi
TERMINAL_LOG="${TASKS_DIR}/logs/${_MODEL_TAG}_${HARNESS}${_THINKING_TAG}_shard${SHARD_INDEX}of${NUM_SHARDS}_${_RUN_TS}.log"

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
if [[ "$THINKING" == "1" ]]; then
  EXTRA_ARGS+=(--thinking)
fi
if [[ "${DISABLE_TERMINAL_LOG:-0}" != "1" ]]; then
  TL="${TERMINAL_LOG}"
  if [[ "$TL" != /* ]]; then
    TL="$PROJECT_ROOT/$TL"
  fi
  mkdir -p "$(dirname "$TL")"
  EXTRA_ARGS+=(--terminal-log "$TL")
fi

echo "=== skill_tax combined 2.5k SFT-data run (THINKING=$THINKING PRESERVE_THINKING=$PRESERVE_THINKING) ==="
echo "    MODEL=${MODEL}"
echo "    HARNESS=${HARNESS}"
echo "    WORKERS=${WORKERS} NUM_SOLUTIONS=${NUM_SOLUTIONS} (= $(( WORKERS * NUM_SOLUTIONS )) concurrent containers)"
echo "    MAX_TOKENS=${MAX_TOKENS} (per-turn output cap)"
echo "    Shard ${SHARD_INDEX}/${NUM_SHARDS}: START_AT=${START_AT} NUM_TASKS=${NUM_TASKS}"
echo "    LITELLM_EXTRA_BODY_JSON=${LITELLM_EXTRA_BODY_JSON:-<unset>}"
echo "    Output filename suffix: ${HARNESS}${_THINKING_TAG}_summary.json"

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
