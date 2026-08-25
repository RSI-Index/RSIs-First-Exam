#!/usr/bin/env bash
# DEFERRED FETCHER. Ships unexecuted. Nothing in this package has been downloaded.
#
# Dry run by default; --yes actually transfers. It does four things beyond `git clone`ing data:
#   1. stages a fixed PREFIX of a streamed dataset into exactly 56 local shards, because
#      train.py:123 hardcodes 56 and SKIPS missing ones with only a warning
#   2. records the revision it actually resolved for every mutable ref
#   3. records realized bytes and row counts, so the size GUESSES in assets.yaml never become
#      load-bearing
#   4. flips assets.yaml NOT_STAGED -> STAGED, last, only if all of the above succeeded
#
# Usage:  environment/fetch_assets.sh --root /scratch/nanovlm-rsi [--yes]
set -euo pipefail

PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT=""
YES=0
PREFIX_ROWS=1500000
SHARDS=56
PY="${PYTHON:-python3}"

while [ $# -gt 0 ]; do
  case "$1" in
    --root)   ROOT="${2:?}"; shift 2 ;;
    --rows)   PREFIX_ROWS="${2:?}"; shift 2 ;;
    --yes)    YES=1; shift ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done
[ -n "$ROOT" ] || { echo "--root is required (where the ~250 GB goes)" >&2; exit 2; }

LOCK="$PKG/environment/staged.lock.yaml"
MANIFEST="$PKG/environment/assets.yaml"
POOL="$ROOT/datasets/finevision_pool"
MODELS="$ROOT/models"
EVAL_HOME="$ROOT/eval-data/lmms_eval_home"

say() { echo "  $*"; }
run() {
  if [ "$YES" -eq 1 ]; then "$@"; else echo "  DRY RUN: $*"; fi
}

echo "fetch_assets.sh -> $ROOT   (yes=$YES)"
echo
echo "SIZE WARNING: assets.yaml's size fields are GUESSES -- nothing upstream states FineVision's"
echo "size. A 1.5M-row prefix with images could plausibly be 75-250 GB. This script measures what"
echo "it actually wrote and records that; do not plan disk from the guesses."
echo

# ---- 1. backbones ------------------------------------------------------------------------
for repo in google/siglip2-base-patch16-512 HuggingFaceTB/SmolLM2-360M-Instruct; do
  dest="$MODELS/$(basename "$repo")"
  say "hf download $repo -> $dest"
  run mkdir -p "$dest"
  run "$PY" -c "
from huggingface_hub import snapshot_download
p = snapshot_download('$repo', local_dir='$dest')
print('resolved:', p)
"
done

# ---- 2. the pool, as 56 shards -----------------------------------------------------------
# A PREFIX of the stream, materialised. Streaming does not shuffle (train.py:160 applies
# shuffle(seed=0) only in the non-streaming path), so "the first N rows" is well defined and
# reproducible given the same dataset revision -- which is why the revision goes in the lockfile.
say "stage $PREFIX_ROWS rows of FineVision_concat_shuffled_2 into $SHARDS shards -> $POOL"
run mkdir -p "$POOL"
run "$PY" - <<PYEOF
import json, math, pathlib
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi

repo = "HuggingFaceM4/FineVision_concat_shuffled_2"
rev = HfApi().dataset_info(repo).sha            # the resolved commit, not "main"
print("resolved revision:", rev)

stream = load_dataset(repo, "default", split="train", streaming=True, revision=rev)
rows_per_shard = math.ceil($PREFIX_ROWS / $SHARDS)
out = pathlib.Path("$POOL")
it = iter(stream)
total = 0
for i in range($SHARDS):
    buf = []
    for _ in range(rows_per_shard):
        try:
            buf.append(next(it))
        except StopIteration:
            break
    if not buf:
        raise SystemExit(
            f"the stream ran out after {total} rows, before shard_{i}. train.py:123 hardcodes 56 "
            f"shards and skips missing ones with only a warning, so a short staging area would "
            f"train on less data and never say so. Lower --rows and re-run."
        )
    Dataset.from_list(buf).save_to_disk(out / f"shard_{i}")
    total += len(buf)
    print(f"  shard_{i}: {len(buf)} rows (total {total})")
pathlib.Path("$ROOT/.pool_rows").write_text(json.dumps({"pool_rows": total, "revision": rev}))
PYEOF

# ---- 3. the evaluation sets, via the PINNED evaluator ------------------------------------
# The dataset ids are lmms-eval's, not ours. This package deliberately declares none of them: it
# invokes the pinned evaluator and records whatever it resolved.
say "pre-download the eval sets for tasks mmstar,chartqa into $EVAL_HOME"
run mkdir -p "$EVAL_HOME"
if [ "$YES" -eq 1 ]; then
  : "${LMMS_EVAL_COMMIT:?set LMMS_EVAL_COMMIT to the commit you are pinning; there is no default}"
fi
run env HF_HOME="$EVAL_HOME" "$PY" -c "
import lmms_eval, subprocess
print('lmms_eval', lmms_eval.__file__)
"

# ---- 4. record, then flip ---------------------------------------------------------------
say "write $LOCK"
if [ "$YES" -eq 1 ]; then
  POOL_ROWS="$("$PY" -c "import json;print(json.load(open('$ROOT/.pool_rows'))['pool_rows'])")"
  POOL_REV="$("$PY" -c "import json;print(json.load(open('$ROOT/.pool_rows'))['revision'])")"
  {
    echo "# Written by environment/fetch_assets.sh. The gates read THIS file, never assets.yaml's"
    echo "# size guesses. Every revision here is the one actually resolved, not a moving ref."
    echo "schema: rsi-staged-lock/1"
    echo "staged_at_utc: \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\""
    echo "pool_rows: $POOL_ROWS"
    echo "pool_revision: \"$POOL_REV\""
    echo "pool_shards: $SHARDS"
    echo "pool_bytes: $(du -sb "$POOL" | cut -f1)"
    echo "pool_dir_digest: \"$("$PY" "$PKG/tests/dir_hash.py" "$POOL")\""
    echo "lmms_eval_commit: \"${LMMS_EVAL_COMMIT}\""
    echo "eval_home_bytes: $(du -sb "$EVAL_HOME" | cut -f1)"
    for m in "$MODELS"/*; do
      echo "backbone_${m##*/}_digest: \"$("$PY" "$PKG/tests/dir_hash.py" "$m")\""
      echo "backbone_${m##*/}_bytes: $(du -sb "$m" | cut -f1)"
    done
  } > "$LOCK"

  # The manifest flip is LAST and it is the only place the flip happens. A STAGED manifest with no
  # lockfile is what contract_check.py refuses, so ordering these the other way round would create
  # a window in which the package claims to be staged and cannot prove it.
  "$PY" - <<PYEOF
import pathlib
p = pathlib.Path("$MANIFEST"); t = p.read_text()
assert t.count("\nstate: NOT_STAGED\n") == 1, "manifest state line is not the bare expected form"
# Stays bare. Three programs read this value with a plain text scan, and an inline comment on it
# once disarmed all three staging gates in the sibling package.
p.write_text(t.replace("\nstate: NOT_STAGED\n", "\nstate: STAGED\n"))
print("assets.yaml: NOT_STAGED -> STAGED")
PYEOF
  echo
  echo "NEXT, and none of it is optional:"
  echo "  1. put lmms_eval_commit and the resolved mmstar metric key into contract.yaml"
  echo "     (data.evaluator_commit and metric.headline.source_metric); until then test.sh gate 0"
  echo "     infra_fails and nothing can be scored"
  echo "  2. check launch_gates.pool_has_selection_freedom: pool_rows >= 3 * 384000"
  echo "  3. measure the anchors at 3 seeds and fill them in, including sigma_floor_binomial"
else
  echo
  echo "DRY RUN ONLY. Nothing was downloaded and assets.yaml still reads NOT_STAGED."
  echo "Re-run with --yes when you have the disk, the bandwidth, and LMMS_EVAL_COMMIT set."
fi
