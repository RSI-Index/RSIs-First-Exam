#!/usr/bin/env bash
# Deferred fetcher for environment/assets.yaml. SHIPPED UNEXECUTED, ON PURPOSE.
#
# Nothing in this package downloads data or weights at build time. This script is the single
# place where that happens, later, deliberately, with one operator watching ~21 GB (GUESS) of
# transfer. It is a dry run unless --yes is passed.
#
# Its real job is not the download -- curl can do that. Its job is to RECORD what arrived:
# the resolved HF revision, the archive sha256, the byte count. Every URL in assets.yaml
# resolves through `main`, which moves, so an unrecorded fetch leaves the chain unreproducible
# and nothing downstream can recover the information. See assets.yaml's header.
#
# Usage:
#   environment/fetch_assets.sh --stage-root /path/to/assets            # dry run, prints plan
#   environment/fetch_assets.sh --stage-root /path/to/assets --yes      # actually fetch
set -euo pipefail

STAGE_ROOT=""
CONFIRM=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage-root) STAGE_ROOT="$2"; shift 2 ;;
    --yes)        CONFIRM=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$STAGE_ROOT" ]] || { echo "--stage-root is required" >&2; exit 2; }

PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF=https://huggingface.co
DS="$HF/datasets/omlab/VLM-R1/resolve/main"

MODEL_DIR="$STAGE_ROOT/models/Qwen2.5-VL-3B-Instruct"
TRAIN_JSONL="$STAGE_ROOT/datasets/vlmr1/rec_jsons_train"
TRAIN_IMG="$STAGE_ROOT/datasets/vlmr1/coco"
EVAL_JSON="$STAGE_ROOT/eval-data/jsons"
EVAL_LISA="$STAGE_ROOT/eval-data/lisa"
WORK="$STAGE_ROOT/_archives"
LOCK="$PKG/environment/staged.lock.yaml"

cat <<PLAN
fetch plan  (stage root: $STAGE_ROOT)

  MODEL
    $HF/Qwen/Qwen2.5-VL-3B-Instruct            -> $MODEL_DIR

  TRAIN  (agent-visible, read-only)
    $DS/rec_jsons_processed.zip                -> $TRAIN_JSONL   (3 *_train.jsonl ONLY)
    $DS/train2014.zip                          -> $TRAIN_IMG

  EVAL   (verifier-only mount; NEVER mounted in the agent container)
    $DS/rec_jsons_processed.zip                -> $EVAL_JSON     (4 eval json ONLY)
    $DS/lisa-test.zip                          -> $EVAL_LISA
    (bind-mount $TRAIN_IMG read-only at /eval-data/coco -- the in-domain guard sets are
     annotated on COCO train2014; do not fetch or copy those ~13 GB twice)

  ONE archive feeds both TRAIN and EVAL. The split is the containment boundary for
  RH-RSI-003; unpacking it wholesale into one directory is the upstream layout and is
  exactly what this package refuses to inherit.

  NOT fetched, ever -- already-RL-trained releases, assets.yaml must_not_stage:
    omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps, omlab/VLM-R1-Qwen2.5VL-3B-OVD-0321,
    omlab/VLM-R1-Qwen2.5VL-3B-Math-0305, rec_jsons_internvl.zip
PLAN

if [[ "$CONFIRM" != "1" ]]; then
  cat <<'DRY'

DRY RUN -- nothing fetched. Re-run with --yes to transfer.
Sizes are GUESSES (~21 GB total); the repository states none. Have the disk before you start.
DRY
  exit 0
fi

mkdir -p "$MODEL_DIR" "$TRAIN_JSONL" "$TRAIN_IMG" "$EVAL_JSON" "$EVAL_LISA" "$WORK"

get() {  # url -> local archive, resumable
  local url="$1" dest="$2"
  echo ">>> $url"
  curl -fL --retry 5 --retry-delay 10 -C - -o "$dest" "$url"
}

# ---- model ---------------------------------------------------------------------------------
# hf CLI so the revision is resolvable; --revision left to `main` here ONLY because the
# resolved sha is captured below. If you already know the sha, pass it and pin it.
python -m pip install --quiet "huggingface_hub[cli]"
python - "$MODEL_DIR" <<'PY'
import json, sys
from huggingface_hub import snapshot_download, HfApi
dest = sys.argv[1]
info = HfApi().model_info("Qwen/Qwen2.5-VL-3B-Instruct")
snapshot_download("Qwen/Qwen2.5-VL-3B-Instruct", revision=info.sha, local_dir=dest)
print(json.dumps({"resolved_revision": info.sha}))
PY

# ---- archives ------------------------------------------------------------------------------
get "$DS/rec_jsons_processed.zip" "$WORK/rec_jsons_processed.zip"
get "$DS/train2014.zip"            "$WORK/train2014.zip"
get "$DS/lisa-test.zip"            "$WORK/lisa-test.zip"

# ---- unpack, SPLIT ------------------------------------------------------------------------
TMP="$WORK/_rec_unpacked"; rm -rf "$TMP"; mkdir -p "$TMP"
unzip -q -o "$WORK/rec_jsons_processed.zip" -d "$TMP"

stage_one() {  # basename dest_dir
  local name="$1" dest="$2" src
  src="$(find "$TMP" -type f -name "$name" -print -quit)"
  if [[ -z "$src" ]]; then
    echo "MISSING in archive: $name -- fix the staging list in assets.yaml, NOT the evaluator" >&2
    return 1
  fi
  install -m 0444 "$src" "$dest/$name"
}
for f in refcoco_train.jsonl refcocop_train.jsonl refcocog_train.jsonl; do stage_one "$f" "$TRAIN_JSONL"; done
for f in lisa_test.json refcoco_val.json refcocop_val.json refcocog_val.json; do stage_one "$f" "$EVAL_JSON"; done

unzip -q -o "$WORK/train2014.zip" -d "$TRAIN_IMG"
unzip -q -o "$WORK/lisa-test.zip" -d "$EVAL_LISA"

# Anything from the rec archive that is neither a staged train jsonl nor a staged eval json is
# discarded rather than parked next to the pool. The upstream single-directory layout is the
# hazard; leaving the leftovers around reintroduces it.
rm -rf "$TMP"

# ---- record what arrived ------------------------------------------------------------------
{
  echo "# WRITTEN BY environment/fetch_assets.sh. Do not hand-edit."
  echo "manifest_version: 1"
  echo "staged_at_utc: \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\""
  echo "stage_root: \"$STAGE_ROOT\""
  echo "archives:"
  for a in rec_jsons_processed.zip train2014.zip lisa-test.zip; do
    printf '  - name: %s\n    sha256: "%s"\n    bytes: %s\n' \
      "$a" "$(sha256sum "$WORK/$a" | cut -d" " -f1)" "$(stat -c%s "$WORK/$a")"
  done
  echo "staged_files:"
  for d in "$TRAIN_JSONL" "$EVAL_JSON"; do
    for f in "$d"/*; do
      printf '  - path: %s\n    sha256: "%s"\n' "$f" "$(sha256sum "$f" | cut -d" " -f1)"
    done
  done
  echo "model:"
  echo "  hf_repo: Qwen/Qwen2.5-VL-3B-Instruct"
  echo "  dir: \"$MODEL_DIR\""
  # SAME code path the evaluator's CK-1 check uses -- tests/dir_hash.py, stdlib only, no torch.
  echo "  dir_manifest_sha256: \"$(python3 "$PKG/tests/dir_hash.py" "$MODEL_DIR")\""
} > "$LOCK"

cat <<NEXT

staged. lockfile: $LOCK

Now copy these into the contract BY HAND, once, and re-run the drift gate:
  contract.yaml  artifact.base_weight_sha256  <- model.dir_manifest_sha256 from the lockfile
  contract.yaml  data.*.sha256                <- the archive sha256s
  assets.yaml    state: NOT_STAGED -> STAGED
  python tests/contract_check.py --package .

contract_check.py rejects state: STAGED while any integrity field still reads
PENDING_DOWNLOAD -- a flag flip is not a recorded bitstream.
NEXT
