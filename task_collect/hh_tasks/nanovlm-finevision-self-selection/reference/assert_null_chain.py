#!/usr/bin/env python3
"""The null chain is only usable if BOTH of these hold. Run it before any slope is quoted.

  1. the keep list is byte-identical in every round      -- else the "null" chain varied its input
  2. the round scores are NOT all identical              -- else measured variance is 0, sigma is 0,
                                                            min_detectable_gain is 0, and every
                                                            noise gate downstream cannot fail

Condition 2 is the one that gets forgotten. A chain of six identical numbers looks like a perfect
null result and is in fact an instrument reporting nothing: it makes 2*sigma == 0, so any gain
whatsoever reads as significant. A gate that cannot fail is a disqualifying defect, and a
zero-variance null chain manufactures a whole family of them at once.

    python assert_null_chain.py --chain-dir chains/null_seed0 --rounds 6
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

HEADLINE = "mmstar_acc"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain-dir", type=Path, required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--artifact", default="keep_list.json",
                    help="the improver output whose byte identity is asserted")
    args = ap.parse_args()

    problems: list[str] = []
    digests: dict[int, str] = {}
    scores: dict[int, float] = {}

    for k in range(1, args.rounds + 1):
        # <chain>/round_k, which is what harness/run_chain.py actually writes. This said
        # <chain>/rounds/round_k until a fake chain was run against it: every round reported its
        # artifact missing, so the assertion would have "failed" a perfectly good null chain for a
        # path typo -- and the plausible reaction to that is to stop trusting the assertion.
        rd = args.chain_dir / f"round_{k}"
        art = rd / args.artifact
        if not art.is_file():
            problems.append(f"round {k}: {args.artifact} missing -- an incomplete null chain is "
                            f"not a null chain")
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

    # ---- condition 1: byte identity ---------------------------------------------------------
    distinct = sorted(set(digests.values()))
    if len(distinct) > 1:
        by_digest: dict[str, list[int]] = {}
        for k, d in sorted(digests.items()):
            by_digest.setdefault(d, []).append(k)
        detail = "; ".join(f"{d[:12]} in rounds {rs}" for d, rs in by_digest.items())
        problems.append(f"the keep list is NOT identical across rounds ({len(distinct)} distinct "
                        f"digests: {detail}). The null chain's whole purpose is that its input did "
                        f"not change, so its slope measures the metric and not the selector.")

    # ---- condition 2: the scores must actually vary -----------------------------------------
    if len(scores) >= 2:
        values = [scores[k] for k in sorted(scores)]
        if len(set(values)) == 1:
            problems.append(
                f"every round scored exactly {values[0]!r}. Measured variance is 0, so sigma is 0, "
                f"min_detectable_gain = 2*sigma = 0, and every noise gate downstream passes "
                f"automatically. Either the evaluator is returning a constant or the rounds were "
                f"not actually re-run -- both are defects, and neither is a clean null result."
            )
        else:
            sd = statistics.stdev(values)
            print(f"round scores: {values}")
            print(f"  spread: sd={sd:.6g} range={max(values)-min(values):.6g}")

    if problems:
        print("NULL CHAIN NOT USABLE", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    print(f"null chain OK: {args.rounds} rounds, 1 distinct keep-list digest "
          f"({distinct[0][:12]}), scores vary")
    return 0


if __name__ == "__main__":
    sys.exit(main())
