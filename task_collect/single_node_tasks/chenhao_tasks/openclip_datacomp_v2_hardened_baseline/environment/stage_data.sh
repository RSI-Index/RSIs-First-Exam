#!/usr/bin/env bash
# One-time data staging for the openclip-datacomp task family. Run by task
# infrastructure on a machine with good egress, NOT inside the task container.
# Produces the sealed snapshot that mounts read-only at
#   /datasets/datacomp-medium-snapshot/<DATE>   (training pool + metadata)
#   /datasets/datacomp-eval                     (38-task evaluation suite)
#
# The crawl is inherently lossy (2026 link rot vs the 2023 pool). The sealed
# manifest written at the end IS the task's canonical data definition; the
# baseline is re-trained on this snapshot, so paper numbers are only
# provisional anchors.
set -euo pipefail

DATACOMP_COMMIT=4a8df1992566ef8334773f7152e1855b1f716162
SNAPSHOT_DATE="${SNAPSHOT_DATE:-$(date +%Y%m%d)}"
STAGE_ROOT="${STAGE_ROOT:-/datasets}"
SNAP="${STAGE_ROOT}/datacomp-medium-snapshot/${SNAPSHOT_DATE}"
EVAL_DIR="${STAGE_ROOT}/datacomp-eval"
WORK="${WORK:-/tmp/datacomp-staging}"
NPROC="${NPROC:-16}"

mkdir -p "${SNAP}/shards" "${SNAP}/metadata" "${EVAL_DIR}" "${WORK}"

if [ ! -d "${WORK}/datacomp" ]; then
  git clone https://github.com/mlfoundations/datacomp.git "${WORK}/datacomp"
fi
git -C "${WORK}/datacomp" checkout "${DATACOMP_COMMIT}"

# 1) Official medium-scale metadata (urls, captions, precomputed CLIP scores).
#    ~817 GB of parquet from HF; kept in the snapshot as curation features.
python "${WORK}/datacomp/download_upstream.py" \
  --scale medium \
  --data_dir "${SNAP}/metadata" \
  --skip_shards \
  --processes "${NPROC}"

# 2) Crawl the images (img2dataset under the hood via the repo tooling),
#    then reshard into training-ready webdataset tars.
python "${WORK}/datacomp/download_upstream.py" \
  --scale medium \
  --data_dir "${SNAP}/raw" \
  --processes "${NPROC}"
python "${WORK}/datacomp/resharder.py" \
  -i "${SNAP}/raw/shards" \
  -o "${SNAP}/shards" \
  -s "${SNAP}/metadata"

# 3) Evaluation suite (38 tasks), staged fully offline.
python "${WORK}/datacomp/download_evalsets.py" "${EVAL_DIR}"

# 4) Seal: content manifest + success accounting. The manifest sha256 is
#    recorded in task.toml [metadata.baseline] at task-build time.
python3 - "$SNAP" <<'PY'
import hashlib, json, os, sys
snap = sys.argv[1]
entries = []
for root, _, files in os.walk(os.path.join(snap, "shards")):
    for name in sorted(files):
        path = os.path.join(root, name)
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        entries.append({"path": os.path.relpath(path, snap),
                        "bytes": os.path.getsize(path),
                        "sha256": digest.hexdigest()})
manifest = {"snapshot": snap, "num_files": len(entries),
            "total_bytes": sum(e["bytes"] for e in entries), "files": entries}
out = os.path.join(snap, "MANIFEST.json")
with open(out, "w") as handle:
    json.dump(manifest, handle, indent=2)
top = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
print(f"snapshot manifest: {out}")
print(f"manifest sha256: {top}")
PY

chmod -R a-w "${SNAP}" "${EVAL_DIR}"
echo "Staging complete: ${SNAP} (training) and ${EVAL_DIR} (eval)."
echo "Mount both read-only into agent and verifier containers."
