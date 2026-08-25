#!/usr/bin/env python3
"""Frozen evaluator. Verifier image only; the agent never sees this file.

Delegates to UPSTREAM's evaluate.py and UPSTREAM's aggregate_scores.get_aggregate_scores. The
headline mean is IMPORTED, not reimplemented, and that is a deliberate choice rather than laziness:
aggregate_scores.py:42 carries `assert len(df) == 38`, and that assert is the only thing standing
between us and a silently-39-dataset average. tasklist.yml holds forty tasks; two
(fairness/fairface, fairness/utkface) fall out implicitly because they have no `main_metric` key, so
upstream's dropna() removes them. Reimplementing the mean would discard the check along with the
code, and the resulting number would still be called average_38.

What this file adds around that:
  * the assets are staged, by COUNT not existence
  * the pool is the SAME pool as every other round (uid fingerprint) -- exit 2, not a zero
  * open_clip is pinned, because train.py:13 imports its internals directly
  * the submitted model is a real training output with upstream's own info.pkl

Usage:  python tests/evaluate.py            (inside the verifier image)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dir_hash import dir_manifest_sha256          # noqa: E402  ONE implementation, shared

PROJECT = Path("/opt/project")
CONTRACT = Path(os.environ.get("DATACOMP_CONTRACT", "/tests/contract.yaml"))
MANIFEST = Path(os.environ.get("DATACOMP_ASSET_MANIFEST", "/assets/manifest.yaml"))
EVAL_DATA = Path(os.environ.get("DATACOMP_EVAL_DATA", "/eval-data/datacomp_evalsets"))
SUBMISSION = Path("/app/output/train")
OUT = Path("/app/output/verifier-metrics.json")

# Contract constants; tests/contract_check.py asserts each against contract.yaml.
N_DATASETS = 38
HEADLINE_AGG_KEY = "Average"
GUARD_DATASET = "ImageNet 1k"
EVAL_BATCH_SIZE = 64


def read_state_naively(path: Path) -> str | None:
    """The manifest's state without a YAML parser; strips an inline comment.

    The comment strip is a bug fix, not defensive style: the VLM-R1 sibling shipped
    `state: NOT_STAGED  # ...` and three readers comparing the whole trailing string against
    "NOT_STAGED" all silently stopped being able to fire.
    """
    for line in path.read_text().splitlines():
        if line.startswith("state:"):
            return line.split(":", 1)[1].split("#", 1)[0].strip()
    return None


def load_yaml(path: Path):
    import yaml
    return yaml.safe_load(path.read_text())


def assert_ready() -> dict:
    problems: list[str] = []
    if not MANIFEST.is_file():
        raise SystemExit(f"asset manifest missing at {MANIFEST}")
    state = read_state_naively(MANIFEST)
    if state != "STAGED":
        problems.append(f"manifest state is {state!r}; this package declares its assets by URL and "
                        f"environment/fetch_assets.sh has not been run")

    contract = load_yaml(CONTRACT)
    if contract["data"].get("leaderboard_comparable") is not False:
        # Not a formality. The pool is crawled from the live web, so a `true` here would be a false
        # claim printed next to every number this file produces.
        problems.append("contract data.leaderboard_comparable is not false; the pool is crawled and "
                        "site-local, so no number here is comparable to the published benchmark")
    if str(contract["data"].get("open_clip_version", "PIN_REQUIRED")) == "PIN_REQUIRED":
        problems.append("contract data.open_clip_version is PIN_REQUIRED. train.py:13 does "
                        "`from training.main import main` -- it imports open_clip's internal "
                        "training entry point, so an unpinned open_clip is an unpinned recipe")

    if not EVAL_DATA.is_dir() or len(list(EVAL_DATA.iterdir())) < 10:
        problems.append(f"{EVAL_DATA} holds fewer than 10 entries; the 38 evaluation sets are not "
                        f"staged (the Dockerfile mkdir -p's this path, so -d is true when empty)")

    # ---- the pool must be the SAME pool as every other round in this chain -------------------
    # This is the check this package exists around. Link rot is monotone and re-crawling gives a
    # different pool, so without it a chain measures the internet.
    lock = Path("/assets/pool.lock.yaml")
    if not lock.is_file():
        problems.append("/assets/pool.lock.yaml missing; there is no record of which pool this "
                        "chain's rounds share")
    else:
        pool = load_yaml(lock)
        for key in ("realized_uids", "uid_set_sha256", "download_rate", "crawled_at_utc"):
            if key not in pool or str(pool[key]).startswith("PENDING"):
                problems.append(f"pool.lock.yaml has no {key}")
        want = contract["data"].get("uid_set_sha256")
        if want and not str(want).startswith("PENDING") and pool.get("uid_set_sha256") != want:
            problems.append(f"pool fingerprint mismatch: lockfile {pool.get('uid_set_sha256')} != "
                            f"contract {want}. The pool changed under the chain.")
        floor = 0.7 * 12_800_000
        if isinstance(pool.get("realized_uids"), int) and pool["realized_uids"] < floor:
            problems.append(f"realized_uids {pool['realized_uids']} is below the preregistered "
                            f"launch gate of {int(floor)} (70% of 12.8M); a crawl that lost more "
                            f"than 30% of the pool is a different benchmark")

    if not SUBMISSION.is_dir():
        problems.append(f"no training output at {SUBMISSION}")
    elif not (SUBMISSION / "info.pkl").is_file():
        # Upstream's evaluate.py opens this unconditionally (evaluate.py:319). Its absence means the
        # submission is not a training output, and letting upstream raise on it would surface as an
        # opaque pickle error rather than as the contract violation it is.
        problems.append(f"{SUBMISSION}/info.pkl missing; upstream evaluate.py requires it and its "
                        f"absence means this is not a train.py output directory")

    if problems:
        raise SystemExit("REFUSING TO EVALUATE:\n" + "\n".join(f"  - {p}" for p in problems))
    return {
        "manifest_state": state,
        "open_clip_version": contract["data"]["open_clip_version"],
        "pool_uid_set_sha256": load_yaml(Path("/assets/pool.lock.yaml"))["uid_set_sha256"],
        "eval_data_digest": dir_manifest_sha256(EVAL_DATA) if EVAL_DATA.is_dir() else None,
    }


def run_upstream_eval(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(PROJECT / "evaluate.py"),
           "--track", "filtering",
           "--train_output_dir", str(SUBMISSION),
           "--output_dir", str(out_dir),
           "--data_dir", str(EVAL_DATA),
           "--batch_size", str(EVAL_BATCH_SIZE)]
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(PROJECT))
    results = out_dir / "eval_results.jsonl"
    if not results.is_file():
        raise SystemExit(f"upstream evaluate.py wrote no {results}")
    return results


def main() -> int:
    ready = assert_ready()
    results = run_upstream_eval(Path("/app/output/eval"))

    sys.path.insert(0, str(PROJECT))
    from aggregate_scores import get_aggregate_scores    # UPSTREAM's, with its len(df) == 38 assert

    agg = get_aggregate_scores(str(results))
    rows = [json.loads(ln) for ln in results.read_text().splitlines() if ln.strip()]
    scored = [r for r in rows if (r.get("metrics") or {}).get("main_metric") is not None]
    if len(scored) != N_DATASETS:
        # Upstream's assert already fires on this. Repeating it with the names is what makes the
        # failure diagnosable: the two datasets that drop out do so implicitly, via a dropna().
        raise SystemExit(
            f"{len(scored)} datasets carry a main_metric, expected {N_DATASETS}. Dropped: "
            f"{sorted(r['dataset'] for r in rows if r not in scored)}. Expected exactly "
            f"FairFace and UTKFace -- neither has a main_metric key in tasklist.yml, so "
            f"evaluate.py falls back to acc1, which the fairness evaluators do not produce."
        )

    metrics = {
        "average_38": float(agg[HEADLINE_AGG_KEY]),
        "imagenet_top1": float(agg["ImageNet"]),
        "imagenet_shifts": float(agg["ImageNet dist. shifts"]),
        "vtab": float(agg["VTAB"]),
        "retrieval": float(agg["Retrieval"]),
        "n_datasets_scored": len(scored),
        "datasets_dropped": sorted(r["dataset"] for r in rows if r not in scored),
        "aggregate_via": "upstream aggregate_scores.get_aggregate_scores",
        "leaderboard_comparable": False,
        "assets": ready,
        "results_file": str(results),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k: v for k, v in metrics.items() if k != "assets"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
