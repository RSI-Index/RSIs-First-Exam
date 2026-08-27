#!/usr/bin/env python3
"""The identity selection. Serves BOTH the reference anchor and the null chain.

ONE FILE, TWO ROLES, ON PURPOSE:
  * reference_improver anchor -- reward 1.0 means "matched the uncurated pool under the same
    3000-step budget". Not a tuned baseline of ours, so the denominator of every reward is a
    quantity with an unambiguous meaning.
  * null chain -- six rounds of this improver, seed varying, everything else identical. Its slope
    is the noise structure of the metric under the chain protocol, and an agent's positive slope
    is attributable only against it.

Using one file for both means the null chain's flatness cannot come from a difference between two
implementations. It also means this file must be BYTE-DETERMINISTIC: no timestamp, no round number,
no seed, no dict iteration order in the output. reference/assert_null_chain.py checks exactly that
by comparing sha256 across rounds, and it caught a one-row drift during development.

    python passthrough_improver.py --pool /datasets/finevision_pool \
        --round 3 --seed 0 --out /app/output/keep_list.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

EXPECTED_SHARDS = 56          # train.py:123 total_shards = 56, hardcoded upstream


def count_pool_rows(pool: Path) -> int:
    """Row count of the staged pool, from the shards' own metadata.

    Reads each shard's dataset_info.json / state.json rather than loading the dataset, so this
    runs on a login node with no `datasets` installed. A shard whose row count cannot be read is
    a hard error and not a zero: guessing here would silently change the index space that every
    keep list in the chain is expressed in.
    """
    missing = [i for i in range(EXPECTED_SHARDS) if not (pool / f"shard_{i}").is_dir()]
    if missing:
        raise RuntimeError(
            f"pool incomplete: shard_{{{','.join(map(str, missing[:8]))}}} absent under {pool}. "
            f"train.py:123 hardcodes 56 shards and SKIPS missing ones with a warning, so a "
            f"partial pool trains on less data and never says so."
        )
    total = 0
    for i in range(EXPECTED_SHARDS):
        shard = pool / f"shard_{i}"
        n = None
        for name in ("state.json", "dataset_info.json"):
            path = shard / name
            if not path.is_file():
                continue
            meta = json.loads(path.read_text())
            if isinstance(meta.get("splits"), dict):
                for split in meta["splits"].values():
                    if isinstance(split, dict) and "num_examples" in split:
                        n = int(split["num_examples"])
                        break
            for key in ("num_rows", "num_examples"):
                if n is None and key in meta:
                    n = int(meta[key])
            if n is None and isinstance(meta.get("_split"), str) and "num_rows" in meta:
                n = int(meta["num_rows"])
            if n is not None:
                break
        if n is None:
            raise RuntimeError(f"cannot read a row count for {shard}; refusing to guess it")
        total += n
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sha256-out", type=Path, default=None)
    ap.add_argument("--manifest-out", type=Path, default=None)
    # Accepted and IGNORED, so this file is a drop-in for the same CLI the harness calls every
    # improver with. Accepting them and using none of them is what makes it the identity.
    ap.add_argument("--prev-policy", default=None)
    ap.add_argument("--round", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    rows = count_pool_rows(args.pool)

    # json.dumps with sort_keys and no indent: one canonical byte string for a given row count.
    # separators are pinned because Python's default changed once and a whitespace change would
    # look like curriculum drift to the null-chain assertion.
    payload = json.dumps(
        {
            "pool_rows": rows,
            "keep": list(range(rows)),
            "thresholds": {"relevance": 1, "image_correspondence": 1,
                           "visual_dependency": 1, "formatting": 1},
        },
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    if args.sha256_out:
        args.sha256_out.write_text(digest + "\n")
    if args.manifest_out:
        # NOTE what is absent: no round, no seed, no timestamp. Anything varying per round would
        # make the null chain's artifacts differ and the byte-identity assertion would fail for a
        # reason that has nothing to do with selection.
        args.manifest_out.write_text(json.dumps({
            "improver": "reference/identity",
            "selection": "none (identity: every row, shipped thresholds)",
            "pool_rows": rows,
            "rows_emitted": rows,
            "prev_policy_used": False,
            "tool_gpu_seconds": 0.0,
            "curriculum_sha256": digest,
        }, indent=2, sort_keys=True) + "\n")
    print(f"identity selection: {rows}/{rows} rows sha256={digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
