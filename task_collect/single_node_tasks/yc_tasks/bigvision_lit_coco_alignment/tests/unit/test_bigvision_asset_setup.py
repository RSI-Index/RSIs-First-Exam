import importlib.util
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("bigvision_prepare_assets", ROOT / "setup/prepare_assets.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_dry_run_plan_names_official_sources(tmp_path: Path) -> None:
    lock = json.loads((ROOT / "setup/asset_sources.lock.json").read_text())
    value = MODULE.plan(ROOT, tmp_path, None, lock)
    assert value["mode"] == "official_task_owned_bootstrap"
    assert value["steps"][1]["builder"] == "coco_captions:1.1.0"
    assert "account" in value["steps"][2]["prerequisite"]


def test_seal_binds_destination_manifest(tmp_path: Path) -> None:
    for relative in ("tfds/coco/file", "tfds/imagenet/file", "initializers/vit_b16_augreg.npz", "verifier/checkpoints/control.npz"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    manifest, contract = MODULE.seal(tmp_path, ROOT)
    contract_value = json.loads(contract.read_text())
    assert contract_value["asset_manifest_sha256"] == MODULE.sha256_file(manifest)
    assert json.loads(manifest.read_text())["file_count"] == 3
    assert json.loads((tmp_path / "verifier/manifest.json").read_text())["file_count"] == 1
    first = manifest.read_bytes(), contract.read_bytes()
    manifest, contract = MODULE.seal(tmp_path, ROOT)
    assert (manifest.read_bytes(), contract.read_bytes()) == first


def test_harbor_wrapper_uses_generated_read_only_mounts(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    (assets / "tfds").mkdir(parents=True)
    (assets / "initializers").mkdir()
    (assets / "initializers/harbor_manifest.json").write_text("{}")
    (assets / "contracts").mkdir()
    (assets / "contracts/baseline_contract.json").write_text("{}")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    harbor = fake_bin / "harbor"
    harbor.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
    harbor.chmod(0o755)
    result = subprocess.run(
        [str(ROOT / "run_harbor.sh"), "-a", "codex", "-m", "fixture"],
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}", "BIGVISION_ASSETS_ROOT": str(assets)},
        text=True,
        capture_output=True,
        check=True,
    )
    assert "/datasets/tfds:ro" in result.stdout
    assert "/datasets/initializers:ro" in result.stdout
    assert "--path" in result.stdout
