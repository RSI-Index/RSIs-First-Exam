import importlib.util
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("datacomp_prepare_assets", ROOT / "setup/prepare_assets.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_dry_run_plan_is_official_and_offline_at_runtime(tmp_path: Path) -> None:
    lock = json.loads((ROOT / "setup/asset_sources.lock.json").read_text())
    value = MODULE.plan(ROOT, tmp_path, lock)
    assert value["mode"] == "official_task_owned_bootstrap"
    assert "download_upstream.py" in value["steps"][1]["command"]
    assert "remain offline" in value["network_scope"]


def test_seal_binds_destination_manifest(tmp_path: Path) -> None:
    lock = json.loads((ROOT / "setup/asset_sources.lock.json").read_text())
    for relative in ("commonpool_s/metadata/a.parquet", "commonpool_s/features/a.npz", "commonpool_s/full_shards/00000000.tar", "evaluation/a.bin"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    uid = tmp_path / "commonpool_s/manifests/available_uids.npy"
    uid.parent.mkdir(parents=True, exist_ok=True)
    uid.write_bytes(b"fixture")
    checkpoint = tmp_path / "verifier/checkpoints" / lock["released_control"]["filename"]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"fixture")
    manifest, contract = MODULE.seal(tmp_path, ROOT, lock)
    contract_value = json.loads(contract.read_text())
    assert contract_value["asset_manifest_sha256"] == MODULE.sha256_file(manifest)
    assert json.loads(manifest.read_text())["file_count"] == 4
    assert json.loads((tmp_path / "evaluation/manifest.json").read_text())["file_count"] == 1
    assert json.loads((tmp_path / "verifier/manifest.json").read_text())["file_count"] == 1
    first = manifest.read_bytes(), contract.read_bytes()
    manifest, contract = MODULE.seal(tmp_path, ROOT, lock)
    assert (manifest.read_bytes(), contract.read_bytes()) == first


def test_harbor_wrapper_uses_generated_read_only_mounts(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    (assets / "commonpool_s/manifests").mkdir(parents=True)
    (assets / "commonpool_s/manifests/asset_manifest.json").write_text("{}")
    (assets / "evaluation").mkdir()
    (assets / "contracts").mkdir()
    (assets / "contracts/baseline_contract.json").write_text("{}")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    harbor = fake_bin / "harbor"
    harbor.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
    harbor.chmod(0o755)
    result = subprocess.run(
        [str(ROOT / "run_harbor.sh"), "-a", "codex", "-m", "fixture"],
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}", "DATACOMP_ASSETS_ROOT": str(assets)},
        text=True,
        capture_output=True,
        check=True,
    )
    assert "/datasets/commonpool_s:ro" in result.stdout
    assert "/datasets/evaluation:ro" in result.stdout
    assert "--path" in result.stdout
