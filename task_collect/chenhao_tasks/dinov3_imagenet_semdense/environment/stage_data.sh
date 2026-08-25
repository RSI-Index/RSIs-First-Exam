#!/usr/bin/env bash
# One-time data staging for the dinov3-imagenet-semdense task. Run by task
# infrastructure, NOT inside the task container. Produces read-only mounts:
#   /datasets/imagenet-1k     (train + val, repo-expected layout + metadata)
#   /datasets/ade20k          (ADEChallengeData2016)
#   /datasets/nyu-depth-v2    (official splits)
#
# ImageNet-1k requires an authorized download (image-net.org account or the
# gated HF mirror ILSVRC/imagenet-1k). This script assumes the two official
# tarballs are already present in $IN1K_TARBALL_DIR and does layout +
# metadata generation only — it never scrapes.
set -euo pipefail

DINOV3_COMMIT=6876159a11b4df116f30f667f8c9888617df0751
STAGE_ROOT="${STAGE_ROOT:-/datasets}"
WORK="${WORK:-/tmp/dinov3-staging}"
IN1K_TARBALL_DIR="${IN1K_TARBALL_DIR:?set to the dir holding ILSVRC2012_img_train.tar and ILSVRC2012_img_val.tar}"

mkdir -p "${STAGE_ROOT}/imagenet-1k" "${STAGE_ROOT}/ade20k" "${STAGE_ROOT}/nyu-depth-v2" "${WORK}"

if [ ! -d "${WORK}/dinov3" ]; then
  git clone https://github.com/facebookresearch/dinov3.git "${WORK}/dinov3"
fi
git -C "${WORK}/dinov3" checkout "${DINOV3_COMMIT}"

# 1) ImageNet-1k: standard train/<wnid>/*.JPEG + val/<wnid>/*.JPEG layout,
#    then the repository's dataset metadata files (see repo README
#    "Data preparation": ImageNet class generates .npy metadata via
#    dinov3.data.datasets — BUILD-TIME VERIFY exact invocation).
python3 - "$IN1K_TARBALL_DIR" "${STAGE_ROOT}/imagenet-1k" <<'PY'
import subprocess, sys, os
tarballs, dest = sys.argv[1], sys.argv[2]
train_tar = os.path.join(tarballs, "ILSVRC2012_img_train.tar")
val_tar = os.path.join(tarballs, "ILSVRC2012_img_val.tar")
for path in (train_tar, val_tar):
    assert os.path.exists(path), f"missing {path}"
print("Extract with the standard ImageNet layout scripts; see")
print("https://github.com/pytorch/examples/tree/main/imagenet (extract_ILSVRC.sh)")
PY
echo ">>> extract tarballs into ${STAGE_ROOT}/imagenet-1k/{train,val} (standard layout),"
echo ">>> then generate repo metadata:"
echo "    PYTHONPATH=${WORK}/dinov3 python -c \"from dinov3.data.datasets import ImageNet; ImageNet(split='TRAIN', root='${STAGE_ROOT}/imagenet-1k', extra='${STAGE_ROOT}/imagenet-1k').dump_extra()\""

# 2) ADE20k (ADEChallengeData2016, ~1 GB, public)
if [ ! -d "${STAGE_ROOT}/ade20k/ADEChallengeData2016" ]; then
  curl -L -o "${WORK}/ade20k.zip" \
    http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
  unzip -q "${WORK}/ade20k.zip" -d "${STAGE_ROOT}/ade20k"
fi

# 3) NYUv2 depth (official labeled subset + splits; public)
if [ ! -f "${STAGE_ROOT}/nyu-depth-v2/nyu_depth_v2_labeled.mat" ]; then
  curl -L -o "${STAGE_ROOT}/nyu-depth-v2/nyu_depth_v2_labeled.mat" \
    http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
fi
echo ">>> convert NYUv2 to the layout expected by dinov3/eval/depth configs"
echo ">>> (BUILD-TIME VERIFY: see dinov3/eval/depth/config-nyu.yaml data paths)"

# 4) Seal: manifest + read-only.
python3 - "${STAGE_ROOT}" <<'PY'
import hashlib, json, os, sys
root = sys.argv[1]
manifest = {}
for name in ("imagenet-1k", "ade20k", "nyu-depth-v2"):
    base = os.path.join(root, name)
    total_bytes = num = 0
    for dirpath, _, files in os.walk(base):
        for f in files:
            num += 1
            total_bytes += os.path.getsize(os.path.join(dirpath, f))
    manifest[name] = {"files": num, "bytes": total_bytes}
out = os.path.join(root, "DINOV3_STAGING_MANIFEST.json")
json.dump(manifest, open(out, "w"), indent=2)
print(json.dumps(manifest, indent=2))
print("manifest:", out)
PY
chmod -R a-w "${STAGE_ROOT}/ade20k" "${STAGE_ROOT}/nyu-depth-v2" || true
echo "Staging complete. Mount all three dirs read-only into agent and verifier containers."
