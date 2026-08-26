#!/usr/bin/env python3
"""Publish this round's checkpoint into /carry, prune to the slot cap, append the ledger row.

The agent calls this. The chain harness ALSO prunes, independently, because an agent that simply
never calls this tool would otherwise hand round k+1 an unbounded ensemble -- and RH-RSI-006 has to
hold whether or not the agent cooperates.

ORDERING MATTERS and it is the one thing to get right here: publish, THEN prune. Publishing after
the prune would let the cap evict the checkpoint that was just produced when the cap is 1, and
snapshotting /carry before publishing is how the sibling harness ended up passing an empty
frozen_models/ to every round -- the recursion channel carried nothing and nothing errored.

    python /task-tools/carry_tool.py publish --round 3 --checkpoint /app/output/checkpoint
    python /task-tools/carry_tool.py show
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            d.update(chunk)
    return d.hexdigest()


def dir_digest(root: Path) -> str:
    """Order-stable digest over relative path + size + content. Same convention as tests/dir_hash.py."""
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
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--carry", type=Path, default=Path("/carry"))
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--manifest", type=Path, default=Path("/app/output/improver_manifest.json"))
    ap.add_argument("--keep-list", type=Path, default=Path("/app/output/keep_list.json"))
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
        print(f"no checkpoint at {args.checkpoint}", file=sys.stderr)
        return 1
    stepdirs = [p.name for p in args.checkpoint.rglob("step_*") if p.is_dir()]
    if stepdirs:
        # Refuse here rather than let the verifier zero the round for it. RH-RSI-006: several
        # checkpoints means the best of them can be chosen on validation.
        print(f"refusing to publish: intermediate checkpoints present ({stepdirs[:4]}). The frozen "
              f"recipe sets eval_in_epochs=False so exactly one checkpoint exists.", file=sys.stderr)
        return 1

    # 1. PUBLISH
    dest = models / f"round_{args.round}"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(args.checkpoint, dest)
    print(f"published {dest}")

    # 2. PRUNE, keeping the most recent `slots` by round number
    keep = sorted(models.glob("round_*"), key=lambda p: int(p.name.split("_")[1]))[-args.slots:]
    for p in sorted(models.glob("round_*")):
        if p not in keep:
            shutil.rmtree(p)
            print(f"pruned {p.name}")
    print(f"carry now holds: {[p.name for p in sorted(keep, key=lambda x: int(x.name.split('_')[1]))]}")

    # 3. APPEND the ledger row. Append only, one row, never a rewrite.
    row = {"round": args.round, "checkpoint_digest": dir_digest(dest)}
    if args.keep_list.is_file():
        spec = json.loads(args.keep_list.read_text())
        row["keep_rows"] = len(spec.get("keep", []))
        row["pool_rows"] = spec.get("pool_rows")
        row["thresholds"] = spec.get("thresholds")
        row["keep_list_sha256"] = hashlib.sha256(args.keep_list.read_bytes()).hexdigest()
    if args.manifest.is_file():
        m = json.loads(args.manifest.read_text())
        row["tool_gpu_seconds"] = m.get("tool_gpu_seconds")
        row["prev_policy_used"] = m.get("prev_policy_used")
    improver = args.carry / "improver" / "improver.py"
    if improver.is_file():
        row["improver_sha256"] = sha256_file(improver)
    with (args.carry / "ledger.jsonl").open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"ledger += round {args.round}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
