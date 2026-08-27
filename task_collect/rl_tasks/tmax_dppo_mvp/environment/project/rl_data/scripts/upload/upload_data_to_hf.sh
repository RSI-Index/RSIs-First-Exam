#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."

# ── Upload RL task dataset to Hugging Face ───────────────────────────
#
# Uploads task_* trees (plus analysis/, data/*.parquet for the viewer).
# ``container.sif`` is excluded (local Apptainer only; RL training uses Docker
# from container.def — see rl_data.upload_to_hf.ALWAYS_IGNORE).
#
# Default below targets the **combined RL corpus** (~14.6k tasks: legacy 10k
# union v2 5k). Override for the standalone 10k or 5k trees as needed.
#
# Usage:
#   bash rl_data/scripts/upload/upload_data_to_hf.sh
#   bash rl_data/scripts/upload/upload_data_to_hf.sh --input-dir rl_data/output/tasks_skill_tax_20260401_10k
#   bash rl_data/scripts/upload/upload_data_to_hf.sh --repo osieosie/my-dataset --private
#   bash rl_data/scripts/upload/upload_data_to_hf.sh --no-parquet
#
# Requirements:
#   - huggingface-cli login  (or HF_TOKEN env var)
#   - Python with huggingface_hub, pandas, pyarrow

REPO_ID="osieosie/tmax-tasks-skill-taxonomy-20260506-legacy10k-new5k-rl"
INPUT_DIR="/gpfs/scrubbed/osey/tmax/rl_data/output/tasks_skill_tax_combined_20260506_legacy10k_new5k"
PRIVATE=""
# These are "opt-out" flags — empty by default (feature ON), set to the
# corresponding CLI flag string when the user passes the option.
#   NO_PARQUET=""        → parquet generation is enabled (default)
#   NO_PARQUET="--no-parquet" → parquet generation is skipped
#   NO_CLEAN=""          → stale upload cache is cleared before upload (default)
#   NO_CLEAN="--no-clean"    → cache is kept, allowing resume of interrupted uploads
#   FAST=""              → use resilient multi-commit upload (default)
#   FAST="--fast"        → use single-commit upload (faster, no resume)
#   COMPACT=""           → upload raw files (default)
#   COMPACT="--compact"  → zip task folders + upload parquet & zip (fastest)
NO_PARQUET=""
NO_CLEAN=""
FAST=""
COMPACT="--compact"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)         REPO_ID="$2"; shift 2 ;;
        --input-dir)    INPUT_DIR="$2"; shift 2 ;;
        --private)      PRIVATE="--private"; shift ;;
        --public)       PRIVATE=""; shift ;;
        --no-parquet)   NO_PARQUET="--no-parquet"; shift ;;
        --no-clean)     NO_CLEAN="--no-clean"; shift ;;
        --fast)         FAST="--fast"; shift ;;
        --compact)      COMPACT="--compact"; shift ;;
        *)              echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== Upload RL Dataset to Hugging Face ==="
echo "  Repo:       ${REPO_ID}"
echo "  Input dir:  ${INPUT_DIR}"
echo ""

exec uv run python -m rl_data.upload_to_hf \
    --repo "${REPO_ID}" \
    --input-dir "${INPUT_DIR}" \
    ${PRIVATE} ${NO_PARQUET} ${NO_CLEAN} ${FAST} ${COMPACT}
