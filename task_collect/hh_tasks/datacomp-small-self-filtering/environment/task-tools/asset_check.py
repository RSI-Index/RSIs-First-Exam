#!/usr/bin/env python3
"""Round-start preflight, run by the agent inside its own container. STDLIB ONLY.

The check that matters here is the POOL FINGERPRINT. Everything else in this file is the usual
staged/not-staged hygiene; the fingerprint is what makes six rounds of a chain comparable at all,
because the pool was crawled from the live web and re-crawling is guaranteed to give a different one.

--root exists so the checks are testable against a fixture tree with no containers.

    python /task-tools/asset_check.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

MIN_POOL_FRACTION = 0.7          # contract.yaml launch_gates.pool_is_large_enough
NOMINAL_POOL = 12_800_000        # README.md:48, "small: 12.8M pool size"


def read_scalar(path: Path, key: str) -> str | None:
    """One top-level scalar from a YAML file, without a YAML parser.

    The comment strip is a bug fix, not defensive style: the VLM-R1 sibling shipped
    `state: NOT_STAGED  # ...` and three readers comparing the whole trailing string against
    "NOT_STAGED" all silently stopped being able to fire.
    """
    if not path.is_file():
        return None
    for line in path.read_text().splitlines():
        if line.startswith(f"{key}:"):
            return line.split(":", 1)[1].split("#", 1)[0].strip().strip('"')
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/"))
    ap.add_argument("--verifier", action="store_true",
                    help="also require the verifier-only eval mount")
    args = ap.parse_args()
    r = args.root
    problems: list[str] = []
    notes: list[str] = []

    manifest = r / "assets/manifest.yaml"
    state = read_scalar(manifest, "state")
    if state is None:
        problems.append(f"no asset manifest at {manifest}")
    elif state != "STAGED":
        problems.append(f"manifest state is {state!r}: this package declares its assets by URL and "
                        f"environment/fetch_assets.sh has not been run")

    # ---- the pool fingerprint ------------------------------------------------------------------
    lock = r / "assets/pool.lock.yaml"
    if not lock.is_file():
        problems.append(f"{lock} missing. It records the realized uid set of the crawl; without it "
                        f"there is nothing holding this round's pool to the pool earlier rounds "
                        f"trained on, and a re-crawl would be invisible.")
    else:
        realized = read_scalar(lock, "realized_uids")
        fingerprint = read_scalar(lock, "uid_set_sha256")
        rate = read_scalar(lock, "download_rate")
        if not fingerprint or fingerprint.startswith("PENDING"):
            problems.append("pool.lock.yaml has no uid_set_sha256")
        if not realized or not realized.isdigit():
            problems.append("pool.lock.yaml has no numeric realized_uids")
        else:
            n = int(realized)
            floor = int(MIN_POOL_FRACTION * NOMINAL_POOL)
            if n < floor:
                problems.append(
                    f"realized_uids {n} is below the preregistered launch gate of {floor} "
                    f"({MIN_POOL_FRACTION:.0%} of {NOMINAL_POOL}). A crawl that lost more than "
                    f"{1 - MIN_POOL_FRACTION:.0%} of the pool is a different benchmark, and the "
                    f"filters being compared were designed against a full pool."
                )
            else:
                notes.append(f"pool: {n} uids ({n / NOMINAL_POOL:.1%} of nominal), "
                             f"fingerprint {fingerprint[:12]}, crawl rate {rate}")

    uids = r / "assets/pool_uids.npy"
    if not uids.is_file():
        problems.append(f"{uids} missing. The verifier checks every submitted uid against it, and "
                        f"without it RH-RSI-003 is unenforceable -- so a round would not be scored "
                        f"on trust, it would not be scored at all.")

    for d, what in ((r / "datasets/commonpool/metadata", "parquet metadata"),
                    (r / "datasets/commonpool/shards", "the crawled webdataset tars")):
        if not d.is_dir() or not any(d.iterdir()):
            problems.append(f"{d} is missing or empty ({what})")

    if args.verifier:
        ev = r / "eval-data/datacomp_evalsets"
        if not ev.is_dir() or len(list(ev.iterdir())) < 10:
            problems.append(f"{ev} holds fewer than 10 entries; the 38 evaluation sets are not staged")
    else:
        # In the agent container this path must NOT exist. It is the containment boundary for
        # RH-RSI-003 and it is cheap to assert from this side too.
        ev = r / "eval-data"
        if ev.exists():
            problems.append(f"{ev} exists in the agent container; the evaluation sets are "
                            f"verifier-only and their ABSENCE is what makes RH-RSI-003 structural "
                            f"rather than a promise")

    # Blocklist sweep. Substrings, not URLs: the realistic failure is a local directory named after
    # the source.
    for needle, why in (("datacomp_1b", "upstream's published best subset -- intersecting our pool "
                                        "with it would make the benchmark measure a join"),
                        ("datacomp_evalsets", "the evaluation sets are the metric"),
                        ("clip-vit", "an untracked scorer; the permitted route to that signal is the "
                                     "similarities that ship in the pool metadata")):
        for base in (r / "datasets", r / "assets"):
            if not base.is_dir():
                continue
            for path in base.rglob("*"):
                if needle in path.name.lower():
                    problems.append(f"blocklisted asset present: {path} ({why})")

    for line in notes:
        print(f"  {line}")
    if problems:
        print("ASSET PREFLIGHT FAILED", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print("asset preflight OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
