#!/usr/bin/env python3
"""The uncurated-pool improver. Serves TWO anchors: `reference_improver` and `null_chain`.

reference_improver = one round of this at the contract's step budget. It defines reward 1.0, so
whatever this file emits IS the bar the benchmark is scored against. It therefore has exactly one
job and must do nothing clever: hand the trainer the pool as upstream ships it, in upstream file
order, and let the trainer's own RepeatRandomSampler and max_steps decide what is consumed.

null_chain = six rounds of this, seed varying in the TRAINER only. For that to be a null
distribution the emitted curriculum must be BYTE-IDENTICAL in every round. Everything below that
looks like paranoia is there to make that true:

  * rows are emitted in the order they appear in the source files, files in the order the
    contract lists them -- no set, no dict-ordering dependence, no sort on a mutable key
  * no timestamp, no hostname, no round number, no seed reaches the output
  * `--sha256-out` writes the digest so reference/assert_null_chain.py can prove identity across
    rounds instead of assuming it

Determinism here is not a style preference. `hash()` is per-process randomised in Python, and a
corpus built with it once differed on 23,432 of 24,496 records while being called deterministic.
A null chain with a drifting curriculum has a recursion in it, and every attribution that leans
on its slope is void.

    python passthrough_improver.py --pool /datasets/vlmr1/rec_jsons_train \
        --out /app/output/curriculum.jsonl --sha256-out /app/output/curriculum.sha256
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Order is fixed by the contract (contract.yaml data.train_pool.jsonl), not discovered from the
# filesystem. `Path.glob` order is filesystem-dependent, which is precisely the kind of thing
# that makes a "deterministic" corpus differ between two machines.
POOL_FILES = ("refcoco_train.jsonl", "refcocop_train.jsonl", "refcocog_train.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sha256-out", type=Path, default=None)
    ap.add_argument("--manifest-out", type=Path, default=None)
    # Accepted and IGNORED. The chain harness passes the same argv to every improver; a
    # passthrough that errored on --prev-policy would force the harness to special-case the
    # anchor arms, and a harness that treats the anchor differently from the agent arms is not
    # measuring the same thing in both.
    ap.add_argument("--prev-policy", default=None, help="accepted and deliberately unused")
    ap.add_argument("--images", default=None, help="accepted and deliberately unused")
    ap.add_argument("--round", type=int, default=None, help="accepted and deliberately unused")
    ap.add_argument("--seed", type=int, default=None, help="accepted and deliberately unused")
    args = ap.parse_args()

    missing = [name for name in POOL_FILES if not (args.pool / name).is_file()]
    if missing:
        print(f"pool incomplete: {missing} not under {args.pool}. Refusing to emit a partial "
              f"uncurated baseline -- it would silently become a curated one, and it is the "
              f"anchor that every reward in this benchmark is divided by.", file=sys.stderr)
        return 2

    rows: list[str] = []
    per_file: dict[str, int] = {}
    for name in POOL_FILES:
        n = 0
        with (args.pool / name).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                # Validate but do not normalise. Re-serialising through json.dumps would make the
                # output depend on this interpreter's dict ordering and float repr.
                try:
                    json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"{name}: unparseable row ({exc}); refusing to emit", file=sys.stderr)
                    return 2
                rows.append(line)
                n += 1
        per_file[name] = n

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = ("\n".join(rows) + "\n").encode("utf-8")
    args.out.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    if args.sha256_out:
        args.sha256_out.write_text(digest + "\n")
    if args.manifest_out:
        args.manifest_out.write_text(json.dumps({
            "improver": "passthrough",
            "selection": "none -- the uncurated pool in upstream file order",
            "rows_total": len(rows),
            "rows_per_file": per_file,
            "curriculum_sha256": digest,
            "prev_policy_used": False,
            "tool_gpu_seconds": 0.0,
            "determinism_note": ("byte-identical across rounds by construction: fixed file order, "
                                 "source line order, no re-serialisation, nothing round- or "
                                 "seed-dependent in the output"),
        }, indent=2) + "\n")

    print(f"rows={len(rows)} {per_file} sha256={digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
