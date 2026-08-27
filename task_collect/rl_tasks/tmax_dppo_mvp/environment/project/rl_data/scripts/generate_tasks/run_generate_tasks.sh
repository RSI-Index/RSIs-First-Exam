#!/bin/bash
#SBATCH --job-name=rl-gen-tasks
#SBATCH --output=logs/gen_tasks_%j.out
#SBATCH --error=logs/gen_tasks_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=960G

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Generic task-generation launcher (v2 pipeline).                          ║
# ║                                                                            ║
# ║  Runs the 4-stage pipeline (template -> initial test -> final test ->     ║
# ║  container.def build+smoke) and writes surviving tasks to $OUT_DIR.       ║
# ║                                                                            ║
# ║  Every parameter below is env-overridable, so you can launch without      ║
# ║  editing this file, e.g.:                                                  ║
# ║      NUM_TASKS=50 OUT_DIR=rl_data/output/tasks_smoke \                     ║
# ║          bash rl_data/scripts/generate_tasks/run_generate_tasks.sh        ║
# ║                                                                            ║
# ║  CORPUS_KIND controls the v2 sampling axes:                               ║
# ║    * legacy  — byte-identical to the pre-v2 pipeline; only exact_text     ║
# ║                verifiers, text_only fixtures, 3 complexity buckets.       ║
# ║    * sft_v2  — upweights non-legacy verifier_kind / fixture_kind /        ║
# ║                intricate complexity (M=2; ~67%% intricate).               ║
# ║    * rl_v2   — same axes, tuned per-axis multipliers for the RL mix.      ║
# ║                                                                            ║
# ║  v2 corpora (sft_v2 / rl_v2) need rl_data/containers/base_intricate.sif.  ║
# ║  Build it once on a build node:                                           ║
# ║      apptainer build rl_data/containers/base_intricate.sif \              ║
# ║                      rl_data/containers/base_intricate.def               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

set -euo pipefail

# ---- Parameters (edit here or override via env) ----
NUM_TASKS="${NUM_TASKS:-10}"
OUT_DIR="${OUT_DIR:-rl_data/output/tasks_skill_tax_toy}"
MODEL="${MODEL:-gemini/gemini-3.1-pro-preview}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
BATCH_SIZE="${BATCH_SIZE:-10}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-128}"
DEF_BUILD_WORKERS="${DEF_BUILD_WORKERS:-16}"
TASK_TEMPERATURE="${TASK_TEMPERATURE:-1.0}"
TEST_TEMPERATURE="${TEST_TEMPERATURE:-0.6}"
CORPUS_KIND="${CORPUS_KIND:-legacy}"   # legacy | sft_v2 | rl_v2

# ---- Resume behaviour ----
# Stages 1-3 are checkpointed to <OUT_DIR>/_intermediates.jsonl and stage 4
# progress to <OUT_DIR>/_stage4_done.jsonl. Delete those to force a full
# regeneration.
# --------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"
mkdir -p logs

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/gpfs/projects/h2lab/osey/apptainer_cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/tmp/apptainer_tmp}"
mkdir -p "$APPTAINER_TMPDIR"

uv run python -c "
from pathlib import Path
from rl_data.generate_tasks import AsyncBatchConfig, run_pipeline
import json

cfg = AsyncBatchConfig(
    num_tasks=$NUM_TASKS,
    out_dir=Path('$OUT_DIR'),
    model='$MODEL',
    max_tokens=$MAX_TOKENS,
    task_temperature=$TASK_TEMPERATURE,
    test_temperature=$TEST_TEMPERATURE,
    batch_size=$BATCH_SIZE,
    max_concurrency=$MAX_CONCURRENCY,
    def_build_workers=$DEF_BUILD_WORKERS,
    corpus_kind='$CORPUS_KIND',
    verbose=True,
)

summary = run_pipeline(cfg)
print(json.dumps(summary, indent=4))
"
