#!/usr/bin/env bash
# DEFERRED FETCHER. Ships unexecuted. Nothing in this package has been downloaded.
#
# Dry run by default; --yes actually transfers. ~528 GB by upstream's own table (README.md:68) and a
# multi-day crawl of 12.8M URLs.
#
# THE PART THAT MATTERS IS STEP 4. The pool is not distributed as data -- img2dataset crawls the live
# web -- so link rot makes the realized pool site-local and time-local. This script therefore records
# the REALIZED UID SET as a fingerprint. Without it, a re-crawl between rounds would be invisible and
# a chain's slope would partly measure the internet.
#
# Usage:  environment/fetch_assets.sh --root /scratch/datacomp-rsi [--yes]
set -euo pipefail

PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT=""
YES=0
SCALE=small
PY="${PYTHON:-python3}"
PROJECT="${DATACOMP_PROJECT:-/opt/project}"

while [ $# -gt 0 ]; do
  case "$1" in
    --root)    ROOT="${2:?}"; shift 2 ;;
    --project) PROJECT="${2:?}"; shift 2 ;;
    --yes)     YES=1; shift ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done
[ -n "$ROOT" ] || { echo "--root is required (~528 GB goes there)" >&2; exit 2; }

LOCK="$PKG/environment/staged.lock.yaml"
POOL_LOCK="$PKG/environment/pool.lock.yaml"
MANIFEST="$PKG/environment/assets.yaml"

say() { echo "  $*"; }
run() { if [ "$YES" -eq 1 ]; then "$@"; else echo "  DRY RUN: $*"; fi }

echo "fetch_assets.sh -> $ROOT   (scale=$SCALE, yes=$YES)"
echo
echo "SIZE, from upstream's own table (README.md:68): metadata 3 GB + npzs 75 GB + tars 450 GB."
echo "Those describe a COMPLETE crawl. Link rot means today's crawl returns less, which is exactly"
echo "why step 4 fingerprints what you actually got."
echo

# ---- 1. metadata ---------------------------------------------------------------------------
say "download_upstream.py --scale $SCALE --skip_shards  (parquets + npzs, metadata only)"
run "$PY" "$PROJECT/download_upstream.py" --scale "$SCALE" \
    --data_dir "$ROOT/datasets/commonpool" --download_npz --skip_shards

# ---- 2. the crawl --------------------------------------------------------------------------
say "download_upstream.py --scale $SCALE  (img2dataset crawls 12.8M urls; expect days, and losses)"
run "$PY" "$PROJECT/download_upstream.py" --scale "$SCALE" \
    --data_dir "$ROOT/datasets/commonpool"

# ---- 3. evaluation sets, VERIFIER SIDE ONLY -------------------------------------------------
say "download_evalsets.py -> $ROOT/eval-data/datacomp_evalsets  (NEVER mounted in the agent image)"
run "$PY" "$PROJECT/download_evalsets.py" --output_dir "$ROOT/eval-data/datacomp_evalsets"

# ---- 4. FINGERPRINT the realized pool ------------------------------------------------------
say "fingerprint the realized uid set -> $POOL_LOCK and $ROOT/assets/pool_uids.npy"
if [ "$YES" -eq 1 ]; then
  mkdir -p "$ROOT/assets"
  "$PY" - <<PYEOF
import hashlib, json, pathlib, tarfile
import numpy as np

shards = pathlib.Path("$ROOT/datasets/commonpool/shards")
tars = sorted(shards.glob("*.tar"))
if not tars:
    raise SystemExit(f"no .tar shards under {shards}; the crawl produced nothing")

uids = set()
for t in tars:
    with tarfile.open(t) as tf:
        for name in tf.getnames():
            stem = name.split("/")[-1]
            if "." not in stem:
                raise SystemExit(
                    f"{t} member {name!r} has no extension. The uid is the part before the first "
                    f"dot; this layout is not what the fingerprint assumes, and returning a digest "
                    f"anyway would silently change what the fingerprint MEANS."
                )
            uids.add(stem.split(".", 1)[0])

ordered = sorted(uids)
digest = hashlib.sha256()
for u in ordered:
    digest.update(u.encode()); digest.update(b"\n")
fingerprint = digest.hexdigest()

# The uid array the verifier checks every submitted subset against, and the array the reference
# identity filter IS. uint64 where possible; DataComp uids are hex strings, so this keeps them as
# fixed-width bytes rather than inventing a numeric mapping that nothing else would share.
arr = np.array(ordered)
np.save("$ROOT/assets/pool_uids.npy", arr, allow_pickle=False)

nominal = 12_800_000
rate = len(ordered) / nominal
total_bytes = sum(p.stat().st_size for p in tars)
pathlib.Path("$POOL_LOCK").write_text(
    "# Written by environment/fetch_assets.sh. THE POOL IDENTITY OF THIS CHAIN.\n"
    "# The pool is crawled from the live web, so re-crawling gives a different pool. Every round of\n"
    "# every chain must verify against this file (tests/pool_check.py); a mismatch is an\n"
    "# INFRASTRUCTURE FAILURE, never a reward of zero.\n"
    "schema: rsi-pool-lock/1\n"
    f'crawled_at_utc: "{__import__("datetime").datetime.utcnow().isoformat()}Z"\n'
    f"realized_uids: {len(ordered)}\n"
    f'uid_set_sha256: "{fingerprint}"\n'
    f"download_rate: {rate:.6f}\n"
    f"nominal_pool: {nominal}\n"
    f"n_shards: {len(tars)}\n"
    f"shards_bytes: {total_bytes}\n"
)
print(f"realized {len(ordered)} uids ({rate:.2%} of nominal), fingerprint {fingerprint[:16]}")
if rate < 0.7:
    raise SystemExit(
        f"LAUNCH GATE FAILED: download_rate {rate:.2%} is below the preregistered 70% "
        f"(contract.yaml launch_gates.pool_is_large_enough). A crawl that lost more than 30% of the "
        f"pool is a different benchmark, and the filters being compared were designed against a full "
        f"one. This threshold was set before the download; do not lower it now."
    )
PYEOF
else
  echo "  DRY RUN: would tar-index every shard, write pool_uids.npy and $POOL_LOCK"
fi

# ---- 5. record, then flip ------------------------------------------------------------------
say "write $LOCK"
if [ "$YES" -eq 1 ]; then
  : "${OPEN_CLIP_VERSION:?set OPEN_CLIP_VERSION to the version you are pinning; there is no default}"
  {
    echo "# Written by environment/fetch_assets.sh. The gates read THIS file, not assets.yaml's"
    echo "# size fields."
    echo "schema: rsi-staged-lock/1"
    echo "staged_at_utc: \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\""
    echo "open_clip_version: \"${OPEN_CLIP_VERSION}\""
    echo "metadata_bytes: $(du -sb "$ROOT/datasets/commonpool/metadata" | cut -f1)"
    echo "shards_bytes: $(du -sb "$ROOT/datasets/commonpool/shards" | cut -f1)"
    echo "evalsets_bytes: $(du -sb "$ROOT/eval-data/datacomp_evalsets" | cut -f1)"
    echo "evalsets_digest: \"$("$PY" "$PKG/tests/dir_hash.py" "$ROOT/eval-data/datacomp_evalsets")\""
  } > "$LOCK"

  # The manifest flip is LAST, and this is the only place it happens. A STAGED manifest with no
  # lockfile is what contract_check.py refuses, so the other ordering would create a window in which
  # the package claims to be staged and cannot prove it.
  "$PY" - <<PYEOF
import pathlib
p = pathlib.Path("$MANIFEST"); t = p.read_text()
assert t.count("\nstate: NOT_STAGED\n") == 1, "manifest state line is not the bare expected form"
# Stays bare: three programs read this value with a plain text scan, and an inline comment on it
# once disarmed all three staging gates in the VLM-R1 sibling.
p.write_text(t.replace("\nstate: NOT_STAGED\n", "\nstate: STAGED\n"))
print("assets.yaml: NOT_STAGED -> STAGED")
PYEOF
  echo
  echo "NEXT, and none of it is optional:"
  echo "  1. copy realized_uids / uid_set_sha256 / download_rate from pool.lock.yaml into"
  echo "     contract.yaml data.*, and open_clip_version too; until then test.sh gate 0 infra_fails"
  echo "  2. measure the anchors at 3 seeds: pristine (untrained) and no_filter, then the reported"
  echo "     baselines (clip_score, basic_filter, text_based) at seed 0"
  echo "  3. launch gates: anchors_separated, then recursion_channel_has_signal -- the second one"
  echo "     decides whether the recursive claim is testable at this scale, and it may fail"
else
  echo
  echo "DRY RUN ONLY. Nothing was downloaded and assets.yaml still reads NOT_STAGED."
  echo "Re-run with --yes when you have ~528 GB, days of crawl budget, and OPEN_CLIP_VERSION set."
fi
