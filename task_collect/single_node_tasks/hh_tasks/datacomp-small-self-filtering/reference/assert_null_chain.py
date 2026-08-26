#!/usr/bin/env python3
"""The null chain is only usable if BOTH of these hold. Run it before any slope is quoted.

  1. the submitted subset is byte-identical in every round  -- else the "null" chain varied its input
  2. the round scores are NOT all identical                 -- else measured variance is 0, sigma is 0,
                                                               min_detectable_gain is 0, and every
                                                               noise gate downstream cannot fail

Condition 2 is the one that gets forgotten. Six identical numbers look like a perfect null result and
are in fact an instrument reporting nothing: 2*sigma == 0 makes any gain whatsoever read as
significant. A gate that cannot fail is a disqualifying defect, and a zero-variance null chain
manufactures a whole family of them at once.

Here there is a THIRD thing worth checking, and it is specific to this package: the pool fingerprint
must be the same in every round. A null chain run across a re-crawl would show variation that is the
internet rather than the seed, and the sigma taken from it would be too large -- which would hide
real effects rather than invent them, but is just as wrong.

    python assert_null_chain.py --chain-dir chains/null_seed0 --rounds 6
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

HEADLINE = "average_38"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain-dir", type=Path, required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--artifact", default="subset.npy")
    args = ap.parse_args()

    problems: list[str] = []
    digests: dict[int, str] = {}
    scores: dict[int, float] = {}
    pools: dict[int, str] = {}

    for k in range(1, args.rounds + 1):
        # <chain>/round_k, which is what harness/run_chain.py actually writes. This said
        # <chain>/rounds/round_k until a fake chain was run against it: every round reported its
        # artifact missing, so the assertion would have "failed" a perfectly good null chain for a
        # path typo -- and the plausible reaction to that is to stop trusting the assertion.
        rd = args.chain_dir / f"round_{k}"
        art = rd / args.artifact
        if not art.is_file():
            problems.append(f"round {k}: {args.artifact} missing -- an incomplete null chain is not "
                            f"a null chain")
        else:
            digests[k] = hashlib.sha256(art.read_bytes()).hexdigest()
        rj = rd / "reward.json"
        if not rj.is_file():
            problems.append(f"round {k}: reward.json missing")
        else:
            payload = json.loads(rj.read_text())
            if HEADLINE not in payload:
                problems.append(f"round {k}: reward.json has no {HEADLINE!r}")
            else:
                scores[k] = float(payload[HEADLINE])
            if payload.get("pool_uid_set_sha256"):
                pools[k] = payload["pool_uid_set_sha256"]

    # ---- condition 1 -------------------------------------------------------------------------
    distinct = sorted(set(digests.values()))
    if len(distinct) > 1:
        by: dict[str, list[int]] = {}
        for k, d in sorted(digests.items()):
            by.setdefault(d, []).append(k)
        detail = "; ".join(f"{d[:12]} in rounds {rs}" for d, rs in by.items())
        problems.append(f"the subset is NOT identical across rounds ({len(distinct)} distinct "
                        f"digests: {detail}). The null chain's whole purpose is that its input did "
                        f"not change, so its slope measures the metric and not the filter.")

    # ---- condition 2 -------------------------------------------------------------------------
    if len(scores) >= 2:
        values = [scores[k] for k in sorted(scores)]
        if len(set(values)) == 1:
            problems.append(
                f"every round scored exactly {values[0]!r}. Measured variance is 0, so sigma is 0, "
                f"min_detectable_gain = 2*sigma = 0, and every noise gate downstream passes "
                f"automatically. Either the evaluator is returning a cached result or the rounds were "
                f"not re-run -- both are defects, and neither is a clean null result."
            )
        else:
            sd = statistics.stdev(values)
            print(f"round scores: {values}")
            print(f"  spread: sd={sd:.6g} range={max(values)-min(values):.6g}")
            # Upstream's own observation, for orientation only -- README.md:194: "differences in
            # accuracy typically at the range of 0.2 percentage points on ImageNet and up to 0.006 on
            # average". Not a threshold: it is their hardware and their pool, and the word is
            # "typically". Printed so a wildly different spread prompts a look rather than a shrug.
            if sd > 0.02:
                print(f"  NOTE sd={sd:.4g} is well above the 0.006 upstream reports as typical for "
                      f"seed variation on the average (README.md:194). Not a failure -- our pool is "
                      f"a different crawl -- but worth understanding before sigma is used.")

    # ---- condition 3, specific to this package -----------------------------------------------
    if len(set(pools.values())) > 1:
        problems.append(f"the pool fingerprint differs across rounds ({sorted(set(pools.values()))}). "
                        f"The null chain ran across more than one crawl, so its spread is partly the "
                        f"internet and the sigma taken from it does not describe seed noise.")
    elif not pools:
        print("  note: no pool fingerprint in the reward files; cannot confirm one crawl across rounds")

    if problems:
        print("NULL CHAIN NOT USABLE", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print(f"null chain OK: {args.rounds} rounds, 1 distinct subset digest ({distinct[0][:12]}), "
          f"scores vary, one pool")
    return 0


if __name__ == "__main__":
    sys.exit(main())
