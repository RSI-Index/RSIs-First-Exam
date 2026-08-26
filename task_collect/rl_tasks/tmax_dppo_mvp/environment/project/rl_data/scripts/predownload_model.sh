#!/bin/bash
# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Pre-download a HuggingFace model into the shared HF cache.           ║
# ║                                                                        ║
# ║  Why:                                                                  ║
# ║   * Compute nodes have flaky / slow outbound network compared to the  ║
# ║     login node, so cold-pulling 50+ GB at job start often saturates    ║
# ║     wall budget or stalls indefinitely.                               ║
# ║   * The HF cache lives on shared GPFS (HF_HOME=/gpfs/scrubbed/osey/   ║
# ║     .cache/huggingface), so a download from the login node is        ║
# ║     immediately visible to every compute node we sbatch into.        ║
# ║                                                                        ║
# ║  Usage (from the LOGIN NODE):                                          ║
# ║    bash rl_data/scripts/predownload_model.sh Qwen/Qwen3.6-27B          ║
# ║    bash rl_data/scripts/predownload_model.sh Qwen/Qwen3.5-9B           ║
# ║    MODEL=Qwen/Qwen3.6-27B bash rl_data/scripts/predownload_model.sh    ║
# ║                                                                        ║
# ║  Optional env:                                                         ║
# ║    HF_TOKEN              -- needed for gated/private repos.            ║
# ║    HF_HOME               -- shared cache root (already preset to      ║
# ║                             /gpfs/scrubbed/osey/.cache/huggingface).  ║
# ║    HF_HUB_ENABLE_HF_TRANSFER=1  -- use the rust-based hf_transfer     ║
# ║                             accelerator (10x faster on fat pipes;    ║
# ║                             needs `pip install hf_transfer`).        ║
# ║                                                                        ║
# ║  After this finishes, the model is cached at:                          ║
# ║    $HF_HOME/hub/models--<org>--<name>/                                ║
# ║  and any subsequent vLLM server in this cluster picks it up offline.  ║
# ╚═══════════════════════════════════════════════════════════════════════╝

set -euo pipefail

# Usage / model resolution.
MODEL="${1:-${MODEL:-}}"
if [[ -z "$MODEL" ]]; then
  cat >&2 <<EOF
ERROR: model id required.

Usage:
  bash $0 <huggingface/model-id>
  MODEL=<huggingface/model-id> bash $0

Examples:
  bash $0 Qwen/Qwen3.6-27B
  bash $0 Qwen/Qwen3.5-9B
EOF
  exit 2
fi

# Sanity guard: the whole point of this helper is to use the login node's
# fat pipe instead of the compute node's flaky one.  We can't reliably tell
# from inside the job alone, but we can warn loudly when SLURM_JOB_ID is
# set (which only happens inside an sbatch / srun).
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "WARN: \$SLURM_JOB_ID=${SLURM_JOB_ID} is set -- you are on a compute node." >&2
  echo "WARN: This helper is designed for LOGIN-NODE pre-downloads where the" >&2
  echo "WARN: outbound network is much faster.  Continuing anyway..." >&2
fi

# Locate hf -- the venv copy preferred (we know it's there); else try PATH.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HF="${HF:-}"
if [[ -z "$HF" ]]; then
  if [[ -x "$PROJECT_ROOT/.venv/bin/hf" ]]; then
    HF="$PROJECT_ROOT/.venv/bin/hf"
  elif command -v hf >/dev/null 2>&1; then
    HF="$(command -v hf)"
  else
    echo "ERROR: 'hf' CLI not found.  Activate the project venv first:" >&2
    echo "    source $PROJECT_ROOT/.venv/bin/activate" >&2
    exit 3
  fi
fi

# Default HF_HOME to the shared GPFS cache so all compute nodes share the
# download.  Honour an explicit override (e.g. someone debugging in /tmp).
export HF_HOME="${HF_HOME:-/gpfs/scrubbed/osey/.cache/huggingface}"
mkdir -p "$HF_HOME/hub"

# hf_transfer is dramatically faster (Rust + multi-stream) for big repos
# but only kicks in when the package is installed.  Probe and enable.
if [[ -z "${HF_HUB_ENABLE_HF_TRANSFER:-}" ]]; then
  if "$HF" env 2>/dev/null | grep -q "hf_transfer.*installed"; then
    export HF_HUB_ENABLE_HF_TRANSFER=1
  elif python -c "import hf_transfer" 2>/dev/null; then
    export HF_HUB_ENABLE_HF_TRANSFER=1
  fi
fi

CACHE_DIR_NAME="models--$(echo "$MODEL" | sed 's|/|--|g')"
CACHE_DIR="$HF_HOME/hub/$CACHE_DIR_NAME"

echo "=== HuggingFace pre-download ==="
echo "  model     : $MODEL"
echo "  hf binary : $HF"
echo "  HF_HOME   : $HF_HOME"
echo "  cache dir : $CACHE_DIR"
echo "  hf_transfer: ${HF_HUB_ENABLE_HF_TRANSFER:-0}"
echo "  hostname  : $(hostname)"
echo ""

# Show pre-download cache state so users see progress on a re-run.
if [[ -d "$CACHE_DIR" ]]; then
  echo "Existing cache size: $(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1)"
else
  echo "No existing cache for $MODEL -- starting fresh."
fi
echo ""

# Show free space on the cache filesystem; bail early if obviously
# insufficient (heuristic: warn under 100 GB, error under 10 GB).
avail_kb=$(df -P "$HF_HOME" 2>/dev/null | awk 'NR==2 {print $4}')
if [[ -n "$avail_kb" ]]; then
  avail_gb=$(( avail_kb / 1024 / 1024 ))
  echo "Free space on \$HF_HOME filesystem: ${avail_gb} GB"
  if [[ "$avail_gb" -lt 10 ]]; then
    echo "ERROR: less than 10 GB free; aborting." >&2
    exit 4
  elif [[ "$avail_gb" -lt 100 ]]; then
    echo "WARN: less than 100 GB free; large models may not fit." >&2
  fi
  echo ""
fi

# Run the download.  --quiet would hide the progress bar; we want to see
# bytes/s on the login node so the user knows whether to ctrl-C and retry.
echo "=== Downloading (this may take a while for large models) ==="
echo "+ $HF download $MODEL"
"$HF" download "$MODEL"

echo ""
echo "=== Post-download ==="
post_size=$(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1 || echo "?")
echo "  cache dir : $CACHE_DIR ($post_size)"

# Sanity: vLLM needs config.json + at least one *.safetensors (or *.bin)
# to load.  Surface a clear failure if the snapshot looks half-baked.
# We use `find` rather than glob+ls because the latter exits non-zero
# whenever ANY of its glob args doesn't match (e.g. weights are .safetensors
# only, no .bin) -- which combined with `set -e -o pipefail` would silently
# abort the script even though the download is fine.
snap=$(find "$CACHE_DIR/snapshots" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -n 1) || true
if [[ -z "$snap" ]]; then
  echo "ERROR: no snapshot dir under $CACHE_DIR/snapshots/" >&2
  exit 5
fi
if [[ ! -e "$snap/config.json" ]]; then
  echo "ERROR: snapshot missing config.json -- download incomplete." >&2
  echo "       run again to resume." >&2
  exit 6
fi
weight_count=$(find "$snap" -maxdepth 1 \( -name '*.safetensors' -o -name '*.bin' \) 2>/dev/null | wc -l)
if [[ "$weight_count" -eq 0 ]]; then
  echo "ERROR: snapshot has no *.safetensors or *.bin -- download incomplete." >&2
  echo "       run again to resume." >&2
  exit 7
fi
echo "  snapshot  : $snap"
echo "  weights   : $weight_count file(s)"
echo ""
echo "Done.  vLLM will pick this up automatically on the next sbatch."
