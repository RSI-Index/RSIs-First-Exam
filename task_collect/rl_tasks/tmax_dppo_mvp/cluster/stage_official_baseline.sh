#!/usr/bin/env bash
set -euo pipefail

repo_id=allenai/tmax-9b
revision=ecb24e8c608870ee5fc776d85f84500375768f15
asset_root=${TMAX_ASSET_ROOT:-/proj/datasets/interns/yuetai/agent_envs/scale_autoresearch_tasks/others/rsi_tasks}
destination=${asset_root}/assets/models/tmax-step-200
python_bin=${TMAX_STAGE_PYTHON:-/u/yuetai/Scale_AutoResearch/others/RSI-tasks/bunddle/samples/tmax_dppo_mvp/environment/project/training/open-instruct/.venv/bin/python}
mode=${1:---stage}
[[ ${mode} == --stage || ${mode} == --check ]] || { echo "usage: stage_official_baseline.sh [--stage|--check]" >&2; exit 2; }

verify() {
    "${python_bin}" - "${destination}" "${repo_id}" "${revision}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
repo_id, revision = sys.argv[2:]
index = json.loads((root / "model.safetensors.index.json").read_text())
shards = sorted(set(index["weight_map"].values()))
if not shards or any(not (root / shard).is_file() for shard in shards):
    raise SystemExit("official baseline has incomplete safetensor shards")
config = json.loads((root / "config.json").read_text())
model_type = str(config.get("model_type", "")).lower()
if "qwen3_5" not in model_type and "qwen3.5" not in model_type:
    raise SystemExit(f"official baseline is not Qwen3.5: {model_type}")
files = {}
for path in sorted(root.iterdir()):
    if path.is_file() and path.name != "manifest.json":
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        files[path.name] = digest.hexdigest()
manifest = {"repository": repo_id, "revision": revision, "branch": "step_200", "files": files}
manifest_path = root / "manifest.json"
if manifest_path.is_file():
    if json.loads(manifest_path.read_text()) != manifest:
        raise SystemExit("official baseline manifest does not match staged files")
else:
    temporary = root / ".manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
print(f"official baseline verified: {root} ({len(shards)} shards)")
PY
}

if [[ -d ${destination} ]]; then
    verify
    exit 0
fi
[[ ${mode} == --stage ]] || { echo "official baseline is not staged: ${destination}" >&2; exit 1; }
mkdir -p -- "$(dirname -- "${destination}")"
exec 9>"${destination}.stage.lock"
flock -n 9 || { echo "official baseline staging is already running" >&2; exit 2; }
temporary=${destination}.part
if [[ ! -e ${temporary} ]]; then
    shopt -s nullglob
    legacy_partials=("${destination}.part."*)
    shopt -u nullglob
    if ((${#legacy_partials[@]} > 1)); then
        echo "multiple interrupted baseline downloads require manual inspection" >&2
        printf '  %s\n' "${legacy_partials[@]}" >&2
        exit 2
    elif ((${#legacy_partials[@]} == 1)); then
        mv -- "${legacy_partials[0]}" "${temporary}"
    fi
fi
[[ ! -e ${temporary} || -d ${temporary} ]] || { echo "temporary baseline path is not a directory: ${temporary}" >&2; exit 2; }
"${python_bin}" - "${repo_id}" "${revision}" "${temporary}" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=sys.argv[1],
    revision=sys.argv[2],
    local_dir=sys.argv[3],
    allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.jinja", "*.txt"],
)
PY
mv -- "${temporary}" "${destination}"
verify
