#!/usr/bin/env python3
"""Prove the null chain was actually null. Run it BEFORE `null_chain.slope` is written anywhere.

`null_chain.slope` is the null distribution that every positive slope in this benchmark is
attributed against. If the "improver that does nothing" in fact did something -- a set iteration
order, a dict ordering, a timestamp, `hash()` on a string, a stray round number -- then its slope
is not a null distribution, and the attribution built on it is void in a way no downstream check
can detect. The failure is silent by construction: a drifting curriculum looks exactly like a
working chain.

So: the emitted curriculum must be byte-identical in every round. Assert it.

Two things this checks that are easy to conflate:
  1. the CURRICULUM is identical across rounds  -- the improver did nothing (this is the point)
  2. the SCORES are not identical across rounds -- the seed did vary (if the scores are also
     byte-identical, the null chain measured one run six times, its variance is 0 by
     construction, and `min_detectable_gain` computed from it would be 0: a gate that cannot fail)

    python assert_null_chain.py --chain-dir /path/chain-null-seed0 --rounds 6
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            d.update(chunk)
    return d.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain-dir", type=Path, required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--headline", default="lisa_test_acc_iou50")
    args = ap.parse_args()

    problems: list[str] = []
    digests: dict[int, str] = {}
    scores: dict[int, float] = {}

    for k in range(1, args.rounds + 1):
        r = args.chain_dir / f"round_{k}"
        # The digest the improver recorded, if present; otherwise hash the curriculum ourselves.
        # Preferring our own hash over a recorded one where both exist -- a recorded digest is a
        # claim, and this file exists to check claims.
        curriculum = None
        for candidate in (r / "curriculum.jsonl", r / "carry_snapshot" / "curriculum.jsonl"):
            if candidate.is_file():
                curriculum = candidate
                break
        recorded = None
        for candidate in (r / "curriculum.sha256", r / "improver_manifest.json"):
            if candidate.is_file():
                if candidate.suffix == ".sha256":
                    recorded = candidate.read_text().strip()
                else:
                    recorded = json.loads(candidate.read_text()).get("curriculum_sha256")
                break
        if curriculum is not None:
            digests[k] = sha256_file(curriculum)
            if recorded and recorded != digests[k]:
                problems.append(f"round {k}: recorded curriculum_sha256 {recorded[:12]}... does "
                                f"not match the file's actual {digests[k][:12]}...")
        elif recorded:
            digests[k] = recorded
            problems.append(f"round {k}: no curriculum.jsonl found; falling back to the RECORDED "
                            f"digest, which is a claim rather than a measurement")
        else:
            problems.append(f"round {k}: neither a curriculum.jsonl nor a recorded digest; "
                            f"identity across rounds cannot be established")

        reward = r / "reward.json"
        if reward.is_file():
            payload = json.loads(reward.read_text())
            if args.headline in payload:
                scores[k] = float(payload[args.headline])

    # 1. the improver did nothing
    unique = sorted(set(digests.values()))
    if len(unique) > 1:
        groups: dict[str, list[int]] = {}
        for k, d in sorted(digests.items()):
            groups.setdefault(d, []).append(k)
        problems.append(
            "NOT A NULL CHAIN: the curriculum changed between rounds. "
            + "; ".join(f"{d[:12]}... rounds {ks}" for d, ks in groups.items())
            + ". A null chain whose input drifts has a recursion in it, so its slope is not a "
              "null distribution and cannot be used to attribute any other chain's slope."
        )

    # 2. the seed did vary
    if len(scores) >= 2:
        distinct = sorted(set(round(v, 6) for v in scores.values()))
        if len(distinct) == 1:
            problems.append(
                f"every round scored exactly {distinct[0]}. Either the seed did not vary or the "
                f"evaluation is deterministic given the artifact -- either way this chain's "
                f"variance is 0 by construction, so a sigma or min_detectable_gain derived from "
                f"it would be 0 and every downstream noise gate CANNOT FAIL."
            )
    else:
        problems.append(f"only {len(scores)} round(s) reported {args.headline}; cannot check that "
                        f"the seed varied")

    print(f"rounds checked: {args.rounds}")
    for k in sorted(digests):
        mark = "=" if len(unique) == 1 else " "
        print(f"  round {k}: curriculum {digests[k][:16]}... {mark}  "
              f"{args.headline}={scores.get(k, 'absent')}")

    if problems:
        print("\nNULL CHAIN INVALID")
        for p in problems:
            print(f"  - {p}")
        return 1
    spread = max(scores.values()) - min(scores.values())
    print(f"\nNULL CHAIN VALID: one curriculum across all {args.rounds} rounds; "
          f"{args.headline} spread {spread:.4f} over {len(scores)} rounds")
    print("  This spread IS the seed noise. contract.yaml sigma_lisa should be consistent with "
          "it; if the two disagree materially, find out why before scoring anything.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
