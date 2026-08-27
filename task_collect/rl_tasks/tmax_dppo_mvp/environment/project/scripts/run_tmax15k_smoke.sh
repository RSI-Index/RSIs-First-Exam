#!/usr/bin/env bash
set -euo pipefail

# Smoke-test the converted TMax-15K-Harbor dataset with Terminus-2 + Claude on
# the Daytona cloud sandbox (Docker is unavailable on the HPC login nodes).
#
# The subset is built to cover BOTH conversion flavours so a single run
# validates the whole converter:
#   * legacy tasks      (self-contained container.def %post)
#   * intricate tasks   (inlined base_intricate layer + %files fixtures)
#                         detected by the presence of environment/base_install.sh
#
# Usage:
#   bash scripts/run_tmax15k_smoke.sh
#   N_LEGACY=3 N_INTRICATE=3 bash scripts/run_tmax15k_smoke.sh
#
# Keys are read from the environment, or from .harbor.env at the repo root if
# present (gitignored). Required:
#   DAYTONA_API_KEY    - Daytona API key
#   ANTHROPIC_API_KEY  - Anthropic API key
#
# Optional env vars:
#   MODEL          - Model name (default: anthropic/claude-sonnet-4-20250514)
#   ENV            - Backend: "daytona" (default) or "docker"
#   N_LEGACY       - legacy tasks in the subset (default: 3)
#   N_INTRICATE    - intricate tasks in the subset (default: 3)
#   N_CONCURRENT   - concurrent trials (default: 6)
#   JOB_NAME       - job name / resume key (default: tmax15k_smoke_claude)
#   FULL_DATASET   - converted Harbor dataset path

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------- Load keys (~/.tmax_secrets then .harbor.env; latter wins) ----------
set -a
# shellcheck disable=SC1090,SC1091
[ -f "$HOME/.tmax_secrets" ] && source "$HOME/.tmax_secrets"
# shellcheck disable=SC1091
[ -f .harbor.env ] && source .harbor.env
set +a

MODEL="${MODEL:-anthropic/claude-sonnet-4-5-20250929}"
ENV="${ENV:-daytona}"
N_LEGACY="${N_LEGACY:-3}"
N_INTRICATE="${N_INTRICATE:-3}"
N_CONCURRENT="${N_CONCURRENT:-6}"
JOB_NAME="${JOB_NAME:-tmax15k_smoke_claude}"
FULL_DATASET="${FULL_DATASET:-rl_data/output/tasks_skill_tax_combined_20260506_legacy10k_new5k_harbor}"
TEST_DATASET="rl_data/output/tmax15k_smoke_${N_LEGACY}l_${N_INTRICATE}i_harbor"
JOB_DIR="jobs/${JOB_NAME}"

# ---------- Validate keys ----------
if [ "$ENV" = "daytona" ] && [ -z "${DAYTONA_API_KEY:-}" ]; then
    echo "ERROR: DAYTONA_API_KEY is required (export it or put it in .harbor.env)."
    exit 1
fi
# Require the model provider's API key based on the MODEL string.
case "$MODEL" in
    anthropic/*|*claude*)
        [ -z "${ANTHROPIC_API_KEY:-}" ] && { echo "ERROR: ANTHROPIC_API_KEY required for model '$MODEL'."; exit 1; } ;;
    gemini/*|google/*|*gemini*)
        [ -z "${GEMINI_API_KEY:-}${GOOGLE_API_KEY:-}" ] && { echo "ERROR: GEMINI_API_KEY (or GOOGLE_API_KEY) required for model '$MODEL'."; exit 1; } ;;
    openai/*|gpt-*|o[0-9]*)
        [ -z "${OPENAI_API_KEY:-}" ] && { echo "ERROR: OPENAI_API_KEY required for model '$MODEL'."; exit 1; } ;;
    *)
        echo "WARN: could not infer the required API key for model '$MODEL'; assuming it is already set." ;;
esac

# ---------- Validate source dataset ----------
if [ ! -d "$FULL_DATASET" ]; then
    echo "ERROR: Harbor dataset not found at $FULL_DATASET"
    echo "Convert it first with rl_data/scripts/analyze/convert_to_harbor.py"
    exit 1
fi

# ---------- Build a mixed legacy+intricate subset (symlinks) ----------
if [ ! -d "$TEST_DATASET" ]; then
    echo "Building smoke subset: ${N_LEGACY} legacy + ${N_INTRICATE} intricate tasks..."
    mkdir -p "$TEST_DATASET"
    legacy=0
    intricate=0
    for task_dir in "$FULL_DATASET"/task_*; do
        [ -d "$task_dir" ] || continue
        if [ -f "$task_dir/environment/base_install.sh" ]; then
            # intricate (has the inlined base layer + fixtures)
            if [ "$intricate" -lt "$N_INTRICATE" ]; then
                ln -sfn "$(realpath "$task_dir")" "$TEST_DATASET/$(basename "$task_dir")"
                intricate=$((intricate + 1))
            fi
        else
            if [ "$legacy" -lt "$N_LEGACY" ]; then
                ln -sfn "$(realpath "$task_dir")" "$TEST_DATASET/$(basename "$task_dir")"
                legacy=$((legacy + 1))
            fi
        fi
        [ "$legacy" -ge "$N_LEGACY" ] && [ "$intricate" -ge "$N_INTRICATE" ] && break
    done
    echo "Subset ready at $TEST_DATASET (legacy=$legacy, intricate=$intricate)"
else
    echo "Using existing smoke subset at $TEST_DATASET"
fi

# ---------- Run or resume ----------
if [ -d "$JOB_DIR" ]; then
    echo "Resuming job from $JOB_DIR"
    if [ "$ENV" = "daytona" ]; then
        uv run harbor jobs resume --job-path "$JOB_DIR" --filter-error-type DaytonaError
    else
        uv run harbor jobs resume --job-path "$JOB_DIR"
    fi
else
    echo "Starting smoke job: $JOB_NAME (env=$ENV, model=$MODEL, n=$N_CONCURRENT)"
    uv run harbor run \
        --path "$TEST_DATASET" \
        --agent terminus-2 \
        --model "$MODEL" \
        --env "$ENV" \
        --n-concurrent "$N_CONCURRENT" \
        --job-name "$JOB_NAME"
fi
