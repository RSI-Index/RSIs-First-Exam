#!/usr/bin/env python3
"""Acquire the pinned official DataComp-S assets, then seal a local deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path


TASK_ID = "datacomp_s_filter_discovery"
UID_DTYPE = "u8,u8"


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def clone_pinned(lock: dict, destination: Path) -> None:
    commit = lock["source_code"]["commit"]
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--filter=blob:none", lock["source_code"]["repository"], str(destination)])
    run(["git", "fetch", "origin", commit, "--depth", "1"], cwd=destination)
    run(["git", "checkout", "--detach", commit], cwd=destination)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=destination, text=True).strip()
    if actual != commit:
        raise RuntimeError(f"source checkout mismatch: expected {commit}, found {actual}")


def download_verified(url: str, destination: Path, expected_size: int, expected_sha256: str) -> None:
    if destination.is_file() and destination.stat().st_size == expected_size and sha256_file(destination) == expected_sha256:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.is_file() and partial.stat().st_size == expected_size and sha256_file(partial) == expected_sha256:
        os.replace(partial, destination)
        return
    offset = partial.stat().st_size if partial.is_file() and partial.stat().st_size < expected_size else 0
    headers = {"User-Agent": "scale-autoresearch-asset-bootstrap/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:
        append = offset > 0 and getattr(response, "status", None) == 206
        with partial.open("ab" if append else "wb") as output:
            shutil.copyfileobj(response, output, 8 * 1024 * 1024)
    if partial.stat().st_size != expected_size:
        raise RuntimeError(f"download size mismatch for {destination.name}")
    if sha256_file(partial) != expected_sha256:
        raise RuntimeError(f"download SHA256 mismatch for {destination.name}")
    os.replace(partial, destination)


def download_metadata_pinned(lock: dict, commonpool: Path) -> None:
    from huggingface_hub import snapshot_download

    metadata = commonpool / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    cache = commonpool / ".hf-cache"
    source = lock["commonpool_s"]
    for pattern in ("*.parquet", "*.npz"):
        snapshot_download(
            repo_id=source["huggingface_dataset"],
            revision=source["revision"],
            repo_type="dataset",
            allow_patterns=pattern,
            local_dir=metadata,
            cache_dir=cache,
            local_dir_use_symlinks=False,
        )
    if len(list(metadata.glob("*.parquet"))) != source["expected_metadata_parquets"]:
        raise RuntimeError("pinned CommonPool-S metadata Parquet count differs from the lock")
    if not any(metadata.glob("*.npz")):
        raise RuntimeError("pinned CommonPool-S feature arrays are missing")
    shutil.rmtree(cache, ignore_errors=True)


def validate_commonpool(commonpool: Path, lock: dict) -> None:
    import pyarrow.parquet as pq

    source = lock["commonpool_s"]
    metadata = sorted((commonpool / "metadata").glob("*.parquet"))
    features = sorted((commonpool / "features").glob("*.npz"))
    shards = commonpool / "full_shards"
    if len(metadata) != source["expected_metadata_parquets"]:
        raise RuntimeError("CommonPool-S metadata Parquet count differs from the lock")
    if sum(pq.ParquetFile(path).metadata.num_rows for path in metadata) != source["expected_metadata_rows"]:
        raise RuntimeError("CommonPool-S metadata row count differs from the lock")
    if len(features) != source["expected_metadata_parquets"]:
        raise RuntimeError("CommonPool-S feature-file count differs from the lock")
    expected_shards = source["expected_img2dataset_shards"]
    for suffix in (".tar", ".parquet", "_stats.json"):
        count = len(list(shards.glob(f"*{suffix}")))
        if count != expected_shards:
            raise RuntimeError(f"CommonPool-S {suffix} shard count differs: expected {expected_shards}, found {count}")


def normalize_commonpool(commonpool: Path, lock: dict) -> None:
    metadata = commonpool / "metadata"
    upstream_shards = commonpool / "shards"
    full_shards = commonpool / "full_shards"
    features = commonpool / "features"
    if upstream_shards.exists() and not full_shards.exists():
        upstream_shards.rename(full_shards)
    features.mkdir(parents=True, exist_ok=True)
    for feature in sorted(metadata.glob("*.npz")):
        target = features / feature.name
        if target.exists():
            raise RuntimeError(f"refusing to overwrite normalized feature: {target}")
        feature.rename(target)
    if not metadata.is_dir() or not full_shards.is_dir() or not any(features.glob("*.npz")):
        raise RuntimeError("official CommonPool-S output is incomplete after normalization")
    validate_commonpool(commonpool, lock)


def validate_evaluation(evaluation: Path, source_root: Path, lock: dict) -> None:
    import yaml

    tasks = yaml.safe_load((source_root / lock["evaluation"]["task_list"]).read_text())
    if len(tasks) != lock["evaluation"]["expected_tasks"]:
        raise RuntimeError("pinned DataComp task list does not contain 40 tasks")
    missing: list[str] = []
    for original in tasks:
        task = original.split("/", 1)[1] if original.startswith(("retrieval/", "misc/", "fairness/")) else original
        if original.startswith(("retrieval/", "misc/")):
            if not list((evaluation / "hf_cache").glob(f"nlphuji___{task}*")):
                missing.append(original)
            continue
        root = evaluation / f"wds_{task.replace('/', '-')}_test"
        try:
            nshards = int((root / "test/nshards.txt").read_text())
        except (OSError, ValueError):
            missing.append(original)
            continue
        required = [root / "classnames.txt", root / "zeroshot_classification_templates.txt"]
        required.extend(root / f"test/{index}.tar" for index in range(nshards))
        if any(not path.is_file() for path in required):
            missing.append(original)
    if missing:
        raise RuntimeError(f"official evaluator download is incomplete for: {missing}")


def validate_released_control(assets_root: Path, lock: dict) -> None:
    checkpoint = lock["released_control"]
    path = assets_root / "verifier" / "checkpoints" / checkpoint["filename"]
    if not path.is_file() or path.stat().st_size != checkpoint["size"] or sha256_file(path) != checkpoint["sha256"]:
        raise RuntimeError("released-control checkpoint differs from the official lock")


def create_uid_universe(shard_root: Path, output: Path) -> int:
    import numpy as np
    import pyarrow.parquet as pq

    pairs: list[tuple[int, int]] = []
    parquets = sorted(shard_root.glob("[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].parquet"))
    if not parquets:
        raise RuntimeError("no img2dataset status Parquets found")
    for parquet in parquets:
        table = pq.read_table(parquet, columns=["uid", "status"])
        for uid, status in zip(table.column("uid").to_pylist(), table.column("status").to_pylist()):
            if status == "success":
                if not isinstance(uid, str) or len(uid) != 32:
                    raise RuntimeError(f"invalid successful UID in {parquet.name}: {uid!r}")
                pairs.append((int(uid[:16], 16), int(uid[16:], 16)))
    array = np.asarray(pairs, dtype=np.dtype(UID_DTYPE))
    array.sort(order=list(array.dtype.names or ()))
    if len(array) > 1 and np.any((array["f0"][1:] == array["f0"][:-1]) & (array["f1"][1:] == array["f1"][:-1])):
        raise RuntimeError("duplicate successful UID in CommonPool-S snapshot")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npy")
    np.save(temporary, array, allow_pickle=False)
    os.replace(temporary, output)
    return len(array)


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
            records.append({
                "path": path.relative_to(assets_root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    records.sort(key=lambda row: row["path"])
    return records


def seal(assets_root: Path, task_root: Path, lock: dict) -> tuple[Path, Path]:
    commonpool = assets_root / "commonpool_s"
    manifest_path = commonpool / "manifests" / "asset_manifest.json"
    roots = [
        commonpool / "metadata",
        commonpool / "features",
        commonpool / "full_shards",
        commonpool / "manifests" / "available_uids.npy",
    ]
    records = inventory(assets_root, roots, exclude={manifest_path})
    manifest = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "source_lock_sha256": sha256_file(task_root / "setup" / "asset_sources.lock.json"),
        "file_count": len(records),
        "total_bytes": sum(row["size"] for row in records),
        "files": records,
    }
    atomic_json(manifest_path, manifest)
    for manifest_root, output in (
        (assets_root / "evaluation", assets_root / "evaluation" / "manifest.json"),
        (assets_root / "verifier", assets_root / "verifier" / "manifest.json"),
    ):
        scoped = inventory(assets_root, [manifest_root], exclude={output})
        atomic_json(output, {
            "schema_version": 1,
            "task_id": TASK_ID,
            "source_lock_sha256": manifest["source_lock_sha256"],
            "file_count": len(scoped),
            "total_bytes": sum(row["size"] for row in scoped),
            "files": scoped,
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


def plan(task_root: Path, assets_root: Path, lock: dict) -> dict:
    return {
        "task_id": TASK_ID,
        "mode": "official_task_owned_bootstrap",
        "network_scope": "setup only; candidate, trainer, and verifier remain offline",
        "assets_root": str(assets_root),
        "steps": [
            {"name": "source", "repository": lock["source_code"]["repository"], "commit": lock["source_code"]["commit"]},
            {"name": "commonpool_s", "command": "python download_upstream.py --scale small --data_dir /assets/commonpool_s --download_npz"},
            {"name": "evaluation", "command": "python download_evalsets.py /assets/evaluation"},
            {"name": "released_control", "url": lock["released_control"]["url"], "role": lock["released_control"]["role"]},
            {"name": "seal", "outputs": ["commonpool_s/manifests/asset_manifest.json", "contracts/baseline_contract.json"]},
        ],
        "task_root": str(task_root),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-root", type=Path, default=Path("/task"))
    parser.add_argument("--assets-root", type=Path, default=Path("/assets"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_root = args.task_root.resolve()
    assets_root = args.assets_root.resolve()
    lock = json.loads((task_root / "setup" / "asset_sources.lock.json").read_text())
    if args.dry_run:
        print(json.dumps(plan(task_root, assets_root, lock), indent=2, sort_keys=True))
        return
    if not args.verify_only:
        assets_root.mkdir(parents=True, exist_ok=True)
        source = assets_root / ".bootstrap" / "datacomp"
        clone_pinned(lock, source)
        commonpool = assets_root / "commonpool_s"
        if not (commonpool / ".official_download_complete").exists():
            download_metadata_pinned(lock, commonpool)
            run(["python", str(source / "download_upstream.py"), "--scale", "small", "--data_dir", str(commonpool), "--download_npz"])
            normalize_commonpool(commonpool, lock)
            (commonpool / ".official_download_complete").write_text(lock["commonpool_s"]["revision"] + "\n")
        if not (assets_root / "evaluation" / ".official_download_complete").exists():
            run(["python", str(source / "download_evalsets.py"), str(assets_root / "evaluation")], cwd=source)
            (assets_root / "evaluation" / ".official_download_complete").write_text(lock["source_code"]["commit"] + "\n")
        uid_path = commonpool / "manifests" / "available_uids.npy"
        create_uid_universe(commonpool / "full_shards", uid_path)
        checkpoint = lock["released_control"]
        download_verified(checkpoint["url"], assets_root / "verifier" / "checkpoints" / checkpoint["filename"], checkpoint["size"], checkpoint["sha256"])
    validate_commonpool(assets_root / "commonpool_s", lock)
    validate_evaluation(assets_root / "evaluation", assets_root / ".bootstrap" / "datacomp", lock)
    validate_released_control(assets_root, lock)
    manifest, contract = seal(assets_root, task_root, lock)
    print(json.dumps({"status": "ready", "asset_manifest": str(manifest), "deployment_contract": str(contract)}, sort_keys=True))


if __name__ == "__main__":
    main()
