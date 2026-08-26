#!/usr/bin/env python3
"""carry_tool.py -- the tool the reference template referenced but never shipped.

instruction.md in samples/gated_deltanet called /task-tools/gdn_async_run.py and
/task-tools/gdn_task_tool.py at five sites; neither program was in bundle.zip and neither
Dockerfile installed them. The async loop was the best property of that template and it was
not in the package. This file and its two siblings are here so that does not repeat.

Subcommands
  init      create an empty carry tree (round 1 only)
  snapshot  seal this round's carry for the next round; enforces carry_model_slots
  verify    local self-check an agent can run before handing off
  meter-start / meter-stop   declare improver (tool) GPU-seconds; RH-RSI-004
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

CARRY = Path("/carry")
METER = CARRY / ".meter.json"
MODEL_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest(root: Path) -> dict:
    entries = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        entries[path.relative_to(root).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_bytes(path.read_bytes()) if path.stat().st_size < (1 << 26) else "large",
        }
    return entries


def gpu_seconds_now() -> float:
    """Wall-clock * visible GPUs. Crude on purpose: it cannot be gamed by nvidia-smi parsing,
    and over-counting a tool pass is the safe direction for a cap."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
        gpus = max(1, len([l for l in out.stdout.splitlines() if l.strip()]))
    except Exception:
        gpus = 1
    return time.monotonic() * gpus


def cmd_init(_: argparse.Namespace) -> int:
    for sub in ("improver", "notes", "frozen_models"):
        (CARRY / sub).mkdir(parents=True, exist_ok=True)
    ledger = CARRY / "ledger.jsonl"
    ledger.touch(exist_ok=True)
    print(json.dumps({"initialised": str(CARRY), "ledger_rows": 0}))
    return 0


def cmd_meter_start(args: argparse.Namespace) -> int:
    state = json.loads(METER.read_text()) if METER.is_file() else {"total_gpu_seconds": 0.0, "spans": []}
    if state.get("open") is not None:
        print("a meter span is already open; call meter-stop first")
        return 2
    state["open"] = {"label": args.label, "start": gpu_seconds_now()}
    METER.write_text(json.dumps(state, indent=2))
    print(json.dumps({"metering": args.label}))
    return 0


def cmd_meter_stop(_: argparse.Namespace) -> int:
    if not METER.is_file():
        print("no meter state; nothing to stop")
        return 2
    state = json.loads(METER.read_text())
    span = state.pop("open", None)
    if span is None:
        print("no open meter span")
        return 2
    spent = max(0.0, gpu_seconds_now() - float(span["start"]))
    state["total_gpu_seconds"] = float(state.get("total_gpu_seconds", 0.0)) + spent
    state.setdefault("spans", []).append({"label": span["label"], "gpu_seconds": round(spent, 1)})
    METER.write_text(json.dumps(state, indent=2))
    print(json.dumps({"span_gpu_seconds": round(spent, 1),
                      "total_gpu_seconds": round(state["total_gpu_seconds"], 1)}))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    problems = []
    if not (CARRY / "improver").is_dir() or not any((CARRY / "improver").rglob("*")):
        problems.append("carry/improver is empty -- the improver is the deliverable of a round")
    ledger = CARRY / "ledger.jsonl"
    rows = [l for l in ledger.read_text().splitlines() if l.strip()] if ledger.is_file() else []
    seen = set()
    for line in rows:
        try:
            seen.add(int(json.loads(line)["round_index"]))
        except Exception:
            problems.append("unparseable ledger row")
    missing = [k for k in range(1, args.round + 1) if k not in seen]
    if missing:
        problems.append(f"ledger missing rounds {missing}")
    if not METER.is_file():
        problems.append("no tool GPU-seconds declared (RH-RSI-004 requires the declaration)")
    stray = [p.as_posix() for p in CARRY.rglob("*")
             if p.is_file() and p.suffix.lower() in MODEL_SUFFIXES
             and "frozen_models" not in p.parts]
    if stray:
        problems.append(f"model weights outside frozen_models/: {stray}")
    print(json.dumps({"ok": not problems, "problems": problems}, indent=2))
    return 0 if not problems else 3


def cmd_snapshot(args: argparse.Namespace) -> int:
    if cmd_verify(args) != 0:
        print("refusing to snapshot: fix the problems above first")
        return 3
    slots = args.slots
    submission = Path(args.submission)
    dest = CARRY / "frozen_models" / f"round_{args.round}"
    if submission.is_dir():
        dest.mkdir(parents=True, exist_ok=True)
        for path in submission.iterdir():
            if path.is_file():
                shutil.copy2(path, dest / path.name)
    # RH-RSI-006: keep only the most recent `slots` rounds. The chain must not accumulate
    # an ensemble; that would turn recursion into model averaging.
    kept = sorted((CARRY / "frozen_models").glob("round_*"),
                  key=lambda p: int(p.name.split("_")[1]), reverse=True)
    for stale in kept[slots:]:
        shutil.rmtree(stale, ignore_errors=True)

    meter = json.loads(METER.read_text()) if METER.is_file() else {}
    snap = {
        "round": args.round,
        "carry_manifest_sha256": sha256_bytes(json.dumps(manifest(CARRY), sort_keys=True).encode()),
        "ledger_rows": len([l for l in (CARRY / "ledger.jsonl").read_text().splitlines() if l.strip()]),
        "tool_gpu_seconds": round(float(meter.get("total_gpu_seconds", 0.0)), 1),
        "frozen_model_rounds": sorted(p.name for p in (CARRY / "frozen_models").glob("round_*")),
    }
    (CARRY / "snapshot.json").write_text(json.dumps(snap, indent=2) + "\n")
    print(json.dumps(snap, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init").set_defaults(fn=cmd_init)
    m1 = sub.add_parser("meter-start"); m1.add_argument("--label", required=True); m1.set_defaults(fn=cmd_meter_start)
    sub.add_parser("meter-stop").set_defaults(fn=cmd_meter_stop)
    v = sub.add_parser("verify"); v.add_argument("--round", type=int, required=True); v.set_defaults(fn=cmd_verify)
    s = sub.add_parser("snapshot")
    s.add_argument("--round", type=int, required=True)
    s.add_argument("--slots", type=int, default=2)
    s.add_argument("--submission", default="/app/output/submission")
    s.set_defaults(fn=cmd_snapshot)
    args = parser.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
