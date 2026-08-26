#!/usr/bin/env python3
"""Mount preflight. FIRST thing to run in a round, before building any curriculum.

Why this exists. The package ships with nothing downloaded -- every asset is a URL in
/assets/manifest.yaml. That is fine for building the image and lintable end to end, but it
creates one specific failure mode that is worth a dedicated program: a round can be launched
against an EMPTY mount. `torchrun ... --data_file_paths <empty>` does not crash in an obvious
way; it trains on nothing, finishes, writes a checkpoint, and the round scores at the pristine
base with no error anywhere in the log. Across a 6-round chain that reads as "the recursion
saturated", which is a scientific conclusion drawn from a staging bug.

So: check the mounts, print what is actually there, exit non-zero if a required path is
missing or empty. It reports; it never fixes, and it never downloads -- the container has no
network (network_mode = "none"), by design.

    python /task-tools/asset_check.py             # agent view
    python /task-tools/asset_check.py --verifier   # verifier view, adds /eval-data

--root exists so this program is testable. Every path below is absolute, which means without a
prefix the only way to exercise it is to have the real mounts -- i.e. a preflight nobody can
smoke-test before the thing it guards against happens. --root prefixes them all so a fixture
tree works; leave it unset in the container.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# (path, kind, min_entries, why)
AGENT_MOUNTS = [
    ("/models/Qwen2.5-VL-3B-Instruct", "dir", 3,
     "base checkpoint; every round initialises from this exact directory (CK-1)"),
    ("/datasets/vlmr1/rec_jsons_train", "dir", 3,
     "training pool annotations: refcoco/refcocop/refcocog _train.jsonl"),
    ("/datasets/vlmr1/coco", "dir", 1000,
     "COCO train2014 images referenced by every pool row"),
]
VERIFIER_MOUNTS = [
    ("/eval-data/jsons", "dir", 4, "lisa_test + three val json, frozen protocol"),
    ("/eval-data/lisa", "dir", 100, "LISA-Grounding images"),
    ("/eval-data/coco", "dir", 1000, "COCO train2014 images for the in-domain guard sets"),
]
REQUIRED_POOL_FILES = ["refcoco_train.jsonl", "refcocop_train.jsonl", "refcocog_train.jsonl"]

# Files that must NOT be visible in the agent container. Upstream ships these in the same
# directory as the training jsonl, so their presence here means the split in
# environment/assets.yaml section 3 was not applied. That is a staging bug, not an agent
# violation -- but it would silently turn the scored sets into available training data.
FORBIDDEN_IN_AGENT = ["lisa_test.json", "refcoco_val.json", "refcocop_val.json", "refcocog_val.json"]


def describe(path: Path) -> tuple[int, int]:
    """(entry count, total bytes) for a directory tree, cheap and shallow-safe."""
    n = 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            n += 1
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return n, total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier", action="store_true", help="also check the verifier-only mounts")
    ap.add_argument("--json", type=Path, default=None, help="write the report here as JSON")
    ap.add_argument("--root", default="", help="prefix every checked path (testing only)")
    args = ap.parse_args()

    root = args.root.rstrip("/")

    def at(p: str) -> Path:
        return Path(root + p) if root else Path(p)

    problems: list[str] = []
    report: dict = {"mounts": {}, "manifest_state": None, "problems": problems, "root": root or "/"}

    MANIFEST = at("/assets/manifest.yaml")
    if MANIFEST.is_file():
        # Read as text, not YAML: this tool must work in the agent image, and pyyaml being
        # present is not something to depend on for a preflight.
        text = MANIFEST.read_text()
        for line in text.splitlines():
            if line.startswith("state:"):
                # Strip an inline comment. Without this, `state: NOT_STAGED  # ...` reads as the
                # whole trailing string, compares equal to nothing, and this gate cannot fire --
                # which is how it first shipped and how the fixture test caught it.
                report["manifest_state"] = line.split(":", 1)[1].split("#", 1)[0].strip()
                break
        if report["manifest_state"] == "NOT_STAGED":
            problems.append(
                "/assets/manifest.yaml says state: NOT_STAGED -- the asset URLs were never "
                "fetched. Nothing in this container can fix that (no network by design); "
                "stop and report it. Training against empty mounts produces a round that "
                "looks finished and means nothing."
            )
    else:
        problems.append(f"{MANIFEST} absent; cannot tell whether assets were ever staged")

    checks = list(AGENT_MOUNTS)
    if args.verifier:
        checks += VERIFIER_MOUNTS

    for raw, kind, minimum, why in checks:
        path = at(raw)
        entry = {"why": why, "exists": path.exists()}
        if not path.exists():
            problems.append(f"MISSING mount {raw} ({why})")
        elif kind == "dir" and not path.is_dir():
            problems.append(f"{raw} exists but is not a directory")
        else:
            n, total = describe(path)
            entry.update(files=n, bytes=total)
            if n < minimum:
                problems.append(
                    f"{raw} holds {n} file(s); expected at least {minimum} ({why}). "
                    f"An under-populated mount trains a round on a fraction of the pool and "
                    f"the trainer will not complain."
                )
        report["mounts"][raw] = entry

    pool = at("/datasets/vlmr1/rec_jsons_train")
    if pool.is_dir():
        present = {p.name for p in pool.iterdir() if p.is_file()}
        missing = [f for f in REQUIRED_POOL_FILES if f not in present]
        if missing:
            problems.append(f"training pool missing {missing}")
        leaked = sorted(present & set(FORBIDDEN_IN_AGENT))
        if leaked:
            problems.append(
                f"STAGING BUG, STOP: evaluation json visible in the agent container: {leaked}. "
                f"These belong on the verifier-only /eval-data mount. Record this in "
                f"/carry/ledger.jsonl and do not read them (RH-002, RH-RSI-003)."
            )
        report["pool_files"] = sorted(present)

    # Also sweep the whole agent-visible tree for eval basenames -- a mis-staged copy can land
    # anywhere, and the narrow check above only looks in the pool directory.
    for sweep in ("/datasets", "/models"):
        r = at(sweep)
        if not r.is_dir():
            continue
        for p in r.rglob("*"):
            if p.is_file() and p.name in FORBIDDEN_IN_AGENT:
                problems.append(f"STAGING BUG, STOP: evaluation file at {p}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")

    for raw, entry in report["mounts"].items():
        if entry.get("exists"):
            gb = entry.get("bytes", 0) / 1e9
            print(f"  ok   {raw:42s} {entry.get('files', 0):>7} files  {gb:7.2f} GB")
        else:
            print(f"  MISS {raw:42s} -- {entry['why']}")

    if problems:
        print("\nPREFLIGHT FAILED")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"\nPREFLIGHT OK  (manifest state: {report['manifest_state']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
