#!/usr/bin/env python3
"""The identity filter: `no_filter`. Serves BOTH the reference anchor and the null chain.

ONE FILE, TWO ROLES, ON PURPOSE:
  * reference_improver anchor -- reward 1.0 means "matched training on the unfiltered pool at the
    same 12.8M-sample budget". Not a tuned baseline of ours, so the reward's denominator has an
    unambiguous meaning.
  * null chain -- six rounds of this improver, seed varying, everything else identical. Its slope is
    the noise structure of the metric under the chain protocol; an agent's positive slope is
    attributable only against it.

One file for both means the null chain's flatness cannot come from a difference between two
implementations. It also forces byte-determinism: no timestamp, no round, no seed in the output.
reference/assert_null_chain.py checks that by sha256 across rounds.

This is `no_filter` from baselines.py:9-17, reimplemented in four lines rather than invoked, for one
reason: baselines.py reads the parquet metadata to enumerate uids, and the REALIZED pool is the
crawled shards, which is a subset of what the metadata lists. The anchor has to be "every uid we
actually have", not "every uid the metadata mentions" -- otherwise the reference arm's subset would
include uids the resharder cannot find, and the arm would silently differ from every agent arm.

    python passthrough_improver.py --pool-uids /assets/pool_uids.npy --out /app/output/subset.npy
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-uids", type=Path, default=Path("/assets/pool_uids.npy"),
                    help="the REALIZED uid set, written by fetch_assets.sh from the crawled shards")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sha256-out", type=Path, default=None)
    ap.add_argument("--manifest-out", type=Path, default=None)
    # Accepted and IGNORED, so this is a drop-in for the CLI the harness calls every improver with.
    # Accepting them and using none of them is what makes it the identity.
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--prev-policy", default=None)
    ap.add_argument("--round", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    import numpy as np

    if not args.pool_uids.is_file():
        raise SystemExit(
            f"{args.pool_uids} missing. This is the realized uid set and the identity filter is "
            f"DEFINED as all of it; falling back to the metadata's uid list would make the reference "
            f"arm include uids the crawl never fetched, and the anchor every reward is divided by "
            f"would then describe a pool that does not exist."
        )
    uids = np.load(args.pool_uids, allow_pickle=False)
    uids = np.unique(uids)          # unique() also sorts; baselines.py:86 writes sorted uids

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, uids, allow_pickle=False)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()

    if args.sha256_out:
        args.sha256_out.write_text(digest + "\n")
    if args.manifest_out:
        # NOTE what is absent: no round, no seed, no timestamp. Anything varying per round would make
        # the null chain's artifacts differ and its byte-identity assertion would fail for a reason
        # that has nothing to do with filtering.
        args.manifest_out.write_text(json.dumps({
            "improver": "reference/identity (no_filter over the REALIZED pool)",
            "n_uids": int(uids.size),
            "prev_policy_used": False,
            "scoring_route": "none",
            "tool_gpu_seconds": 0.0,
            "subset_sha256": digest,
        }, indent=2, sort_keys=True) + "\n")
    print(f"identity filter: {uids.size} uids sha256={digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
