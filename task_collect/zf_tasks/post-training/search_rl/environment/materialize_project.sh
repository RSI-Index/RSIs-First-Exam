#!/usr/bin/env bash
set -euo pipefail

destination=""
while (($#)); do
  case "$1" in
    --destination)
      [[ $# -ge 2 ]] || { echo "--destination requires a value" >&2; exit 2; }
      destination=$2
      shift 2
      ;;
    --help|-h)
      echo "usage: materialize_project.sh --destination PATH"
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

[[ -n ${destination} ]] || { echo "--destination is required" >&2; exit 2; }
[[ ! -e ${destination} ]] || { echo "destination already exists: ${destination}" >&2; exit 2; }

source_url=https://github.com/OSU-NLP-Group/QUEST.git
source_commit=962e10b6b99f53efc8fd45229ed80728b9ee9a05
expected_tree=9238a96476fa1607fc572f8d0e62d52cdd441c851a9d51b31c5d76000a7e780c
destination_parent=$(dirname "${destination}")
mkdir -p "${destination_parent}"
staging=$(mktemp -d "${destination_parent}/.search-rl-materialize.XXXXXX")
trap 'rm -rf -- "${staging}"' EXIT

git clone --filter=blob:none --no-checkout "${source_url}" "${staging}/source"
git -C "${staging}/source" checkout --detach "${source_commit}"
git -C "${staging}/source" submodule update --init --depth 1 training_scripts/sft/LlamaFactory

mkdir "${staging}/project"
tar -C "${staging}/source" \
  --exclude='./.git' \
  --exclude='./evaluation' \
  --exclude='*/.git' \
  -cf - . | tar -C "${staging}/project" -xf -

[[ ! -e ${staging}/project/evaluation ]] || {
  echo "redaction failed: evaluation/ remains" >&2
  exit 1
}

actual_tree=$(python3 - "${staging}/project" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
records = []
for path in sorted(root.rglob("*")):
    relative = path.relative_to(root)
    if ".git" in relative.parts or "__pycache__" in relative.parts or path.is_dir():
        continue
    if path.is_symlink():
        records.append([relative.as_posix(), "symlink", os.readlink(path)])
    elif path.is_file():
        records.append([relative.as_posix(), "file", hashlib.sha256(path.read_bytes()).hexdigest()])
payload = json.dumps(records, separators=(",", ":"), ensure_ascii=False).encode()
print(hashlib.sha256(payload).hexdigest())
PY
)

[[ ${actual_tree} == "${expected_tree}" ]] || {
  echo "redacted source tree hash mismatch: ${actual_tree}" >&2
  exit 1
}

mv "${staging}/project" "${destination}"
echo "materialized redacted QUEST source at ${destination} (${actual_tree})"
