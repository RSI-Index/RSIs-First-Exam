#!/usr/bin/env python3
"""Acquire official LiT/COCO assets and seal a destination-cluster contract."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path


TASK_ID = "bigvision_lit_coco_alignment"


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def sha256_file(path: Path) -> str:
    return digest(path, "sha256")


def md5_base64(path: Path) -> str:
    return base64.b64encode(bytes.fromhex(digest(path, "md5"))).decode()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def download_verified(entry: dict, assets_root: Path) -> Path:
    destination = assets_root / entry["path"]
    if destination.is_file() and destination.stat().st_size == entry["size"] and md5_base64(destination) == entry["md5_base64"]:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.is_file() and partial.stat().st_size == entry["size"] and md5_base64(partial) == entry["md5_base64"]:
        os.replace(partial, destination)
        return destination
    offset = partial.stat().st_size if partial.is_file() and partial.stat().st_size < entry["size"] else 0
    headers = {"User-Agent": "scale-autoresearch-asset-bootstrap/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(entry["url"], headers=headers)
    with urllib.request.urlopen(request) as response:
        append = offset > 0 and getattr(response, "status", None) == 206
        with partial.open("ab" if append else "wb") as output:
            shutil.copyfileobj(response, output, 8 * 1024 * 1024)
    if partial.stat().st_size != entry["size"]:
        raise RuntimeError(f"download size mismatch: {entry['path']}")
    if md5_base64(partial) != entry["md5_base64"]:
        raise RuntimeError(f"download MD5 mismatch: {entry['path']}")
    os.replace(partial, destination)
    return destination


def require_imagenet_archives(manual_dir: Path, lock: dict) -> None:
    required = lock["tfds"]["imagenet2012"]["required_archives"]
    missing = [name for name in required if not (manual_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            "ImageNet is license-gated. Download the official ILSVRC2012 archives after accepting its terms, "
            f"place {missing} in {manual_dir}, and rerun setup. No mirror is used."
        )


def prepare_tfds(tfds_root: Path, manual_dir: Path, lock: dict) -> None:
    import tensorflow_datasets as tfds

    tfds_root.mkdir(parents=True, exist_ok=True)
    coco = tfds.builder(lock["tfds"]["coco_captions"]["builder"], data_dir=str(tfds_root))
    coco.download_and_prepare()
    imagenet = tfds.builder(lock["tfds"]["imagenet2012"]["builder"], data_dir=str(tfds_root))
    imagenet.download_and_prepare(download_config=tfds.download.DownloadConfig(manual_dir=str(manual_dir)))


def validate_tfds(tfds_root: Path, lock: dict) -> None:
    for name, source in lock["tfds"].items():
        builder, version = source["builder"].split(":", 1)
        root = tfds_root / builder / version
        if not (root / "dataset_info.json").is_file():
            raise RuntimeError(f"official TFDS preparation is missing {name} dataset_info.json")
        if not any(root.glob("*.tfrecord-*")):
            raise RuntimeError(f"official TFDS preparation has no data shards for {name}")


def validate_public_objects(assets_root: Path, lock: dict) -> None:
    for entry in lock["public_objects"]:
        path = assets_root / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["size"] or md5_base64(path) != entry["md5_base64"]:
            raise RuntimeError(f"official object differs from lock: {entry['path']}")


def inventory(assets_root: Path, roots: list[Path], *, exclude: set[Path] | None = None) -> list[dict]:
    excluded = {path.resolve() for path in (exclude or set())}
    records: list[dict] = []
    for root in roots:
        if not root.exists():
            raise RuntimeError(f"required asset path is missing: {root}")
        paths = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file() and not path.is_symlink())
        for path in paths:
            if path.resolve() in excluded:
                continue
            records.append({"path": path.relative_to(assets_root).as_posix(), "size": path.stat().st_size, "sha256": sha256_file(path)})
    records.sort(key=lambda row: row["path"])
    return records


def seal(assets_root: Path, task_root: Path) -> tuple[Path, Path]:
    manifest_path = assets_root / "initializers" / "harbor_manifest.json"
    records = inventory(assets_root, [assets_root / "tfds", assets_root / "initializers"], exclude={manifest_path})
    manifest = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "source_lock_sha256": sha256_file(task_root / "setup" / "asset_sources.lock.json"),
        "file_count": len(records),
        "total_bytes": sum(row["size"] for row in records),
        "files": records,
    }
    atomic_json(manifest_path, manifest)
    verifier_manifest = assets_root / "verifier" / "manifest.json"
    verifier_records = inventory(assets_root, [assets_root / "verifier"], exclude={verifier_manifest})
    atomic_json(verifier_manifest, {
        "schema_version": 1,
        "task_id": TASK_ID,
        "source_lock_sha256": manifest["source_lock_sha256"],
        "file_count": len(verifier_records),
        "total_bytes": sum(row["size"] for row in verifier_records),
        "files": verifier_records,
    })
    template = json.loads((task_root / "baseline_contract.json").read_text())
    template["asset_manifest_sha256"] = sha256_file(manifest_path)
    template["deployment_asset_manifest"] = str(manifest_path)
    template["calibration_provenance"]["asset_manifest_sha256"] = {
        "evidence_class": "destination_cluster_asset_lock",
        "source": str(manifest_path),
    }
    contract_path = assets_root / "contracts" / "baseline_contract.json"
    atomic_json(contract_path, template)
    return manifest_path, contract_path


def plan(task_root: Path, assets_root: Path, manual_dir: Path | None, lock: dict) -> dict:
    return {
        "task_id": TASK_ID,
        "mode": "official_task_owned_bootstrap",
        "network_scope": "setup only; candidate, trainer, and verifier remain offline",
        "assets_root": str(assets_root),
        "imagenet_manual_dir": str(manual_dir) if manual_dir else "required for non-dry-run",
        "steps": [
            {"name": "public_initializers", "objects": len(lock["public_objects"]) - 1, "provider": "official Google Cloud Storage"},
            {"name": "coco", "builder": lock["tfds"]["coco_captions"]["builder"]},
            {"name": "imagenet", "builder": lock["tfds"]["imagenet2012"]["builder"], "prerequisite": lock["tfds"]["imagenet2012"]["note"]},
            {"name": "released_control", "objects": 1, "role": "verifier-only"},
            {"name": "seal", "outputs": ["initializers/harbor_manifest.json", "contracts/baseline_contract.json"]},
        ],
        "task_root": str(task_root),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-root", type=Path, default=Path("/task"))
    parser.add_argument("--assets-root", type=Path, default=Path("/assets"))
    parser.add_argument("--imagenet-manual-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_root = args.task_root.resolve()
    assets_root = args.assets_root.resolve()
    manual_dir = args.imagenet_manual_dir.resolve() if args.imagenet_manual_dir else None
    lock = json.loads((task_root / "setup" / "asset_sources.lock.json").read_text())
    if args.dry_run:
        print(json.dumps(plan(task_root, assets_root, manual_dir, lock), indent=2, sort_keys=True))
        return
    if not args.verify_only:
        if manual_dir is None:
            raise RuntimeError("--imagenet-manual-dir is required; see setup/README.md")
        require_imagenet_archives(manual_dir, lock)
        assets_root.mkdir(parents=True, exist_ok=True)
        for entry in lock["public_objects"]:
            download_verified(entry, assets_root)
        prepare_tfds(assets_root / "tfds", manual_dir, lock)
    validate_public_objects(assets_root, lock)
    validate_tfds(assets_root / "tfds", lock)
    manifest, contract = seal(assets_root, task_root)
    print(json.dumps({"status": "ready", "asset_manifest": str(manifest), "deployment_contract": str(contract)}, sort_keys=True))


if __name__ == "__main__":
    main()
