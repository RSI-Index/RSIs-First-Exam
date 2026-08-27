#!/usr/bin/env python3
"""Publish this round's model into /carry, prune to the slot cap, append the ledger row.

The agent calls this. The chain harness ALSO prunes, independently, because an agent that simply
never calls this tool would otherwise hand round k+1 an unbounded ensemble -- RH-RSI-006 has to hold
whether or not the agent cooperates.

ORDERING MATTERS: publish, THEN prune. Snapshotting /carry before publishing is how the sibling
harness ended up passing an empty frozen_models/ to every round -- the recursion channel carried
nothing and nothing errored.

    python /task-tools/carry_tool.py publish --round 3 --checkpoint /app/output/train
    python /task-tools/carry_tool.py show
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

WEIGHT_SUFFIXES = {".pt", ".pth", ".bin", ".safetensors"}


def sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            d.update(chunk)
    return d.hexdigest()


def dir_digest(root: Path) -> str:
    """Order-stable digest over relative path + size + content; same convention as tests/dir_hash.py."""
    d = hashlib.sha256()
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        d.update(p.relative_to(root).as_posix().encode())
        d.update(str(p.stat().st_size).encode())
        d.update(bytes.fromhex(sha256_file(p)))
    return d.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["publish", "show"])
    ap.add_argument("--round", type=int)
    ap.add_argument("--checkpoint", type=Path, help="the train.py output dir for this round")
    ap.add_argument("--carry", type=Path, default=Path("/carry"))
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--subset", type=Path, default=Path("/app/output/subset.npy"))
    ap.add_argument("--manifest", type=Path, default=Path("/app/output/improver_manifest.json"))
    args = ap.parse_args()

    models = args.carry / "frozen_models"
    models.mkdir(parents=True, exist_ok=True)

    if args.action == "show":
        for p in sorted(models.glob("round_*")):
            print(f"  {p.name}  {dir_digest(p)[:12]}")
        ledger = args.carry / "ledger.jsonl"
        if ledger.is_file():
            print(f"  ledger: {len(ledger.read_text().splitlines())} row(s)")
        return 0

    if args.round is None or args.checkpoint is None:
        print("publish needs --round and --checkpoint", file=sys.stderr)
        return 2
    if not args.checkpoint.is_dir():
        print(f"no training output at {args.checkpoint}", file=sys.stderr)
        return 1
    if not (args.checkpoint / "info.pkl").is_file():
        # Upstream's evaluate.py:319 opens this unconditionally, and carry_check.py reads the recipe
        # back out of it. Refusing here beats letting the verifier fail on an opaque pickle error.
        print(f"refusing to publish: {args.checkpoint}/info.pkl missing, so this is not a train.py "
              f"output directory and the recipe that ran cannot be read back", file=sys.stderr)
        return 1

    # 1. PUBLISH -- weights only. The training output dir also holds logs and, depending on flags,
    # intermediate epoch checkpoints; copying all of it into /carry would grow the carry channel
    # without bound and hand round k+1 an ensemble by accident.
    dest = models / f"round_{args.round}"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    copied = 0
    for p in sorted(args.checkpoint.rglob("*")):
        if p.is_file() and (p.suffix.lower() in WEIGHT_SUFFIXES or p.name == "info.pkl"):
            target = dest / p.relative_to(args.checkpoint)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
            copied += 1
    if not copied:
        print(f"refusing to publish: no weight files under {args.checkpoint}", file=sys.stderr)
        shutil.rmtree(dest)
        return 1
    print(f"published {dest} ({copied} file(s))")

    # 2. PRUNE, keeping the most recent `slots` by round number
    keep = sorted(models.glob("round_*"), key=lambda p: int(p.name.split("_")[1]))[-args.slots:]
    for p in sorted(models.glob("round_*")):
        if p not in keep:
            shutil.rmtree(p)
            print(f"pruned {p.name}")
    print(f"carry now holds: {[p.name for p in sorted(keep, key=lambda x: int(x.name.split('_')[1]))]}")

    # 3. APPEND the ledger row. Append only, one row, never a rewrite.
    row = {"round": args.round, "checkpoint_digest": dir_digest(dest)}
    if args.subset.is_file():
        row["subset_sha256"] = hashlib.sha256(args.subset.read_bytes()).hexdigest()
        try:
            import numpy as np
            row["n_uids"] = int(np.load(args.subset, allow_pickle=False).size)
        except Exception:
            row["n_uids"] = None
    if args.manifest.is_file():
        m = json.loads(args.manifest.read_text())
        row["tool_gpu_seconds"] = m.get("tool_gpu_seconds")
        row["prev_policy_used"] = m.get("prev_policy_used")
        row["scoring_route"] = m.get("scoring_route")
    improver = args.carry / "improver" / "improver.py"
    if improver.is_file():
        row["improver_sha256"] = sha256_file(improver)
    with (args.carry / "ledger.jsonl").open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"ledger += round {args.round}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
