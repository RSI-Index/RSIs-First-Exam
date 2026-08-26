#!/usr/bin/env python3
"""The pool must be the SAME pool in every round of a chain. This is that check.

WHY THIS FILE EXISTS AND ITS SIBLINGS HAVE NO EQUIVALENT. The other two packages point at hosted
artifacts: a HF branch might move, which is a hazard. Here the pool is not distributed as data at
all -- download_upstream.py fetches parquets of URLs and img2dataset crawls the live web. Link rot
is monotone, so re-crawling is GUARANTEED to produce a different pool. If round 4 ran against a
re-crawl, the chain's slope would partly measure the internet, and nothing about the round would
look wrong.

So the realized uid set is fingerprinted once, at staging, and re-verified at every round start.

SEVERITY: a mismatch is exit 2, an INFRASTRUCTURE FAILURE, never a reward of zero. The agent did not
do anything wrong when the pool changed underneath it, and feeding a 0.0 into chain_score.py as
though it were a measurement is how a chain gets reported as saturated.

    python tests/pool_check.py --lock /assets/pool.lock.yaml --shards /datasets/commonpool/shards
    python tests/pool_check.py --lock ... --shards ... --recompute      # full re-fingerprint, slow
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import yaml

EXIT_OK, EXIT_MISMATCH = 0, 2


def uid_fingerprint(shards: Path) -> tuple[int, str]:
    """sha256 over the sorted uid list, and the count.

    Reads uids from the webdataset shards' own file listing rather than opening the tars: every
    member is named <uid>.<ext>, so the .json members give the uid set without decoding images. If
    the layout ever changes this raises instead of returning a plausible wrong digest -- a
    fingerprint that silently changes meaning is worse than no fingerprint.
    """
    import tarfile
    uids: set[str] = set()
    tars = sorted(shards.glob("*.tar"))
    if not tars:
        raise RuntimeError(f"no .tar shards under {shards}")
    for tar_path in tars:
        with tarfile.open(tar_path) as tf:
            names = tf.getnames()
        if not names:
            raise RuntimeError(f"{tar_path} is empty")
        for name in names:
            stem = name.split("/")[-1]
            if "." not in stem:
                raise RuntimeError(
                    f"{tar_path} member {name!r} has no extension; the uid is the part before the "
                    f"first dot and this layout is not what the fingerprint assumes. Refusing to "
                    f"guess rather than return a digest that means something else."
                )
            uids.add(stem.split(".", 1)[0])
    digest = hashlib.sha256()
    for uid in sorted(uids):
        digest.update(uid.encode())
        digest.update(b"\n")
    return len(uids), digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", type=Path, default=Path("/assets/pool.lock.yaml"))
    ap.add_argument("--shards", type=Path, default=Path("/datasets/commonpool/shards"))
    ap.add_argument("--recompute", action="store_true",
                    help="re-read every shard and compare digests. Slow (it opens ~12.8M members' "
                         "worth of tar indexes) but it is the only check that actually detects a "
                         "re-crawl; without it this only verifies the shard COUNT and total size.")
    ap.add_argument("--round", type=int, default=None)
    args = ap.parse_args()

    problems: list[str] = []
    if not args.lock.is_file():
        print(f"POOL CHECK FAILED: {args.lock} missing -- there is no record of which pool this "
              f"chain's rounds share", file=sys.stderr)
        return EXIT_MISMATCH
    lock = yaml.safe_load(args.lock.read_text()) or {}

    for key in ("realized_uids", "uid_set_sha256", "n_shards", "shards_bytes", "crawled_at_utc"):
        if key not in lock or str(lock[key]).startswith("PENDING"):
            problems.append(f"pool.lock.yaml has no usable {key}")

    if not args.shards.is_dir():
        problems.append(f"{args.shards} does not exist")
    else:
        tars = sorted(args.shards.glob("*.tar"))
        if lock.get("n_shards") not in (None, len(tars)):
            problems.append(f"{len(tars)} shard tars present, lockfile records {lock['n_shards']}")
        total = sum(p.stat().st_size for p in tars)
        if isinstance(lock.get("shards_bytes"), int) and total != lock["shards_bytes"]:
            problems.append(f"shard bytes {total} != lockfile {lock['shards_bytes']}")
        if args.recompute:
            n, digest = uid_fingerprint(args.shards)
            if n != lock.get("realized_uids"):
                problems.append(f"realized uid count {n} != lockfile {lock.get('realized_uids')}")
            if digest != lock.get("uid_set_sha256"):
                problems.append(
                    f"UID SET FINGERPRINT MISMATCH: {digest} != {lock.get('uid_set_sha256')}. The "
                    f"pool is not the pool this chain's earlier rounds trained on. Every "
                    f"cross-round comparison in this chain is void."
                )
        else:
            print("  note: shard count and total bytes only. Pass --recompute for the uid "
                  "fingerprint, which is the check that detects a re-crawl.")

    if problems:
        print("POOL CHECK FAILED", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print("  This is an INFRASTRUCTURE FAILURE (exit 2), not a zero: a pool that changed "
              "underneath the chain is not a round the agent earned nothing on.", file=sys.stderr)
        return EXIT_MISMATCH
    print(f"pool OK: {lock.get('realized_uids')} uids, {lock.get('n_shards')} shards, "
          f"fingerprint {str(lock.get('uid_set_sha256'))[:12]}"
          f"{' (verified)' if args.recompute else ' (declared; not recomputed)'}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
