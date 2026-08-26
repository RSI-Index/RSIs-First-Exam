#!/usr/bin/env python3
"""Round-start preflight, run by the agent inside its own container.

STDLIB ONLY. It has to run in the agent image, which has no pyyaml guarantee, and it has to run
before anything expensive. Everything it checks is something that would otherwise produce a round
that LOOKS finished:

  * an empty or short pool trains on less data than the contract says, silently
  * a blocklisted pretrained checkpoint in the mounts makes round 1 a download
  * an eval task name appearing inside the training pool is leakage

--root exists so the whole thing is testable against a fixture tree with no containers.

    python /task-tools/asset_check.py
    python /task-tools/asset_check.py --root /tmp/fixture      # for testing the checks themselves
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXPECTED_SHARDS = 56          # train.py:123, hardcoded upstream

# Substrings, not URLs: the failure mode is a local directory named after the repo, so matching on
# the full URL would miss every realistic instance of it. Both trained nanoVLM checkpoints are
# here because the two entry points default to DIFFERENT ones --
# eval/lmms_eval_wrapper.py:28 -> nanoVLM-450M, generate.py:21 -> nanoVLM-230M-8k.
FORBIDDEN_SUBSTRINGS = ["nanoVLM-450M", "nanoVLM-230M"]

# The nine task names at models/config.py:85. Any of them appearing as a path component inside the
# training mounts means an evaluation set was staged where the agent can reach it.
EVAL_TASK_NAMES = ["mmstar", "mmmu", "ocrbench", "textvqa", "docvqa", "scienceqa", "mme",
                   "infovqa", "chartqa"]


def read_state(manifest: Path) -> str | None:
    """The manifest's state, without a YAML parser. Strips an inline comment -- see below."""
    if not manifest.is_file():
        return None
    for line in manifest.read_text().splitlines():
        if line.startswith("state:"):
            # The comment strip is not defensive programming, it is a bug fix. The sibling package
            # shipped `state: NOT_STAGED  # NOT_STAGED -> STAGED, flipped by fetch_assets.sh` and
            # three readers doing exactly this comparison compared against the whole trailing
            # string. Every staging gate in the package silently stopped being able to fire.
            return line.split(":", 1)[1].split("#", 1)[0].strip()
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/"),
                    help="prefix for every path below; for testing the checks themselves")
    ap.add_argument("--verifier", action="store_true",
                    help="also require the verifier-only eval mount (used inside the verifier image)")
    args = ap.parse_args()
    r = args.root
    problems: list[str] = []
    notes: list[str] = []

    manifest = r / "assets/manifest.yaml"
    state = read_state(manifest)
    if state is None:
        problems.append(f"no asset manifest at {manifest}")
    elif state != "STAGED":
        problems.append(f"manifest state is {state!r}: this package declares its assets by URL and "
                        f"environment/fetch_assets.sh has not been run. Nothing below can pass.")

    pool = r / "datasets/finevision_pool"
    if not pool.is_dir():
        problems.append(f"{pool} does not exist")
    else:
        shards = sorted(p.name for p in pool.glob("shard_*") if p.is_dir())
        if len(shards) < EXPECTED_SHARDS:
            problems.append(
                f"{pool} holds {len(shards)} shard_* dirs, expected {EXPECTED_SHARDS}. "
                f"train.py:123 hardcodes 56 and its loader SKIPS a missing shard with only a "
                f"warning, so this trains on a smaller pool than the contract describes and the "
                f"round still completes."
            )
        else:
            notes.append(f"pool: {len(shards)} shards")

    for name in ("siglip2-base-patch16-512", "SmolLM2-360M-Instruct"):
        d = r / "models" / name
        if not d.is_dir() or not any(d.iterdir()):
            problems.append(f"backbone mount {d} is missing or empty")

    if args.verifier:
        home = r / "eval-data/lmms_eval_home"
        if not home.is_dir() or not any(home.iterdir()):
            problems.append(f"{home} is missing or empty; the verifier has no evaluation data")
    else:
        # In the agent container this path must NOT exist. It is the containment boundary for
        # RH-RSI-003, and it is cheap to assert from this side too.
        home = r / "eval-data"
        if home.exists():
            problems.append(f"{home} exists in the agent container; the evaluation sets are "
                            f"verifier-only and their absence is what makes RH-RSI-003 structural")

    # Sweep the agent-visible mounts for things that must not be there.
    for base in (r / "datasets", r / "models"):
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            name = path.name
            for needle in FORBIDDEN_SUBSTRINGS:
                if needle.lower() in name.lower():
                    problems.append(f"blocklisted trained checkpoint present: {path}")
            lowered = name.lower()
            for task in EVAL_TASK_NAMES:
                if lowered.startswith(task) or f"_{task}" in lowered:
                    problems.append(f"an evaluation task name appears inside a training mount: "
                                    f"{path} (matched {task!r})")

    for line in notes:
        print(f"  {line}")
    if problems:
        print("ASSET PREFLIGHT FAILED", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print("asset preflight OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
