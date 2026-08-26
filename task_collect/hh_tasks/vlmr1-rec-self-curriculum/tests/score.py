#!/usr/bin/env python3
"""Per-round reward. ANCHORED, not absolute -- and the anchors come from contract.yaml.

Why anchored: the reference package used reward = 1/(1+ln P), an absolute map. Measured at
its own baseline (P=14.8076 -> 0.270626), a 10% metric improvement earned +0.0079 reward.
Round-to-round increments in a chain would land in the third decimal, under seed noise. An
anchored map puts the whole [0,1] range across the interval that matters.

Two invariants this file enforces, both of which the reference package got wrong:
  D-2  the metric key scored here IS the key named in task.toml [metadata.optimization].
       There, score.py rewarded val_ppl@1x while task.toml declared the geomean, and the
       recorded baseline reward was the geomean value -- so the shipped scorer could not
       reproduce the shipped baseline.
  D-3  feeding the recorded baseline metrics through this file must reproduce the recorded
       baseline reward. tests/contract_check.py asserts it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

HEADLINE_KEY = "lisa_test_acc_iou50"
GUARD_KEY = "indomain_avg_acc_iou50"


def anchored(score: float, pristine: float, reference: float, sigma: float) -> float:
    """Anchored at pristine=0 and reference=1, floored at 0, and DELIBERATELY NOT capped at 1.

    An earlier version of this file clipped to [0, 1] the way AutoLab-1.1 does. That is right
    for a one-shot task where the reference anchor is the target, and wrong here: a chain is
    supposed to EXCEED the reference improver, and clipping made every at-or-above-reference
    round score exactly 1.0 -- so auc pinned at 1.0 and the slope went to 0 while the raw
    metric was still climbing. Caught by running tests/chain_score.py on a synthetic chain,
    which is the only reason it was caught. The upper cap is removed; 1.0 now means "matched
    the uncurated-pool recipe" rather than "won".
    """
    span = reference - pristine
    if span == 0:
        raise RuntimeError("degenerate anchors: reference == pristine, reward cannot resolve")
    if abs(span) < 2.0 * sigma:
        # Added while building the cheap-tier sibling, before any anchor here was measured, and
        # it is a STRICTER refusal rather than a relaxation. `reference != pristine` is satisfied
        # by a 0.3-sigma gap, and dividing by a 0.3-sigma denominator makes the reward amplified
        # noise: a round that moved the metric by seed luck alone would score 3.0. The instrument
        # cannot resolve the interval it is being asked to measure against, so it refuses.
        raise RuntimeError(
            f"anchors are inside the noise band: |reference - pristine| = {abs(span):.6g} "
            f"< 2*sigma = {2.0 * sigma:.6g}. The reward denominator would be noise."
        )
    return max(0.0, (score - pristine) / span)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=Path("/tests/contract.yaml"))
    parser.add_argument("--metrics", type=Path, default=Path("/logs/verifier/metrics.json"))
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--out", type=Path, default=Path("/logs/verifier/reward.json"))
    args = parser.parse_args()

    contract = yaml.safe_load(args.contract.read_text())
    metrics = json.loads(args.metrics.read_text())

    declared = contract["metric"]["headline"]["name"]
    if declared != HEADLINE_KEY:
        raise RuntimeError(f"D-2 violated: contract headline is {declared}, this scorer rewards {HEADLINE_KEY}")

    anchors = contract["anchors"]
    pristine = anchors["pristine_base"][HEADLINE_KEY]
    reference = anchors["reference_improver"][HEADLINE_KEY]
    sigma = anchors["pristine_base"]["sigma_lisa"]
    if "MEASURE" in (pristine, reference, sigma):
        raise RuntimeError(
            "anchors are unmeasured. Run the pristine-base and reference-improver arms at "
            "3 seeds and fill contract.yaml before any agent round is scored. Scoring a "
            "chain against MEASURE would be inventing the bar."
        )

    score = float(metrics[HEADLINE_KEY])
    guard = float(metrics[GUARD_KEY])
    guard_pristine = float(anchors["pristine_base"][GUARD_KEY])
    max_drop = float(contract["metric"]["guard"]["max_relative_drop"])

    reward = anchored(score, float(pristine), float(reference), float(sigma))
    guard_ok = guard >= guard_pristine * (1.0 - max_drop)
    if not guard_ok:
        # A round that wins out-of-domain by destroying in-domain accuracy swapped the
        # distribution; it did not improve grounding. Zero, and say why.
        reward = 0.0

    # Noise-normalised view. A chain claim is a claim about round-to-round DIFFERENCES, so the
    # natural unit is the measured noise floor, not accuracy points.
    z = (score - float(reference)) / float(sigma) if float(sigma) else None

    result = {
        "round": args.round,
        "reward": round(reward, 6),
        "reward_scale_note": "1.0 == matched the reference improver; >1.0 == beat it",
        "z_vs_reference": round(z, 3) if z is not None else None,
        "policy_gate": 1.0,
        "metric": HEADLINE_KEY,
        HEADLINE_KEY: score,
        GUARD_KEY: guard,
        "guard_ok": guard_ok,
        "guard_floor": guard_pristine * (1.0 - max_drop),
        "anchor_pristine": pristine,
        "anchor_reference": reference,
        "sigma": sigma,
        "min_detectable_gain": 2.0 * float(sigma),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    (args.out.parent / "reward.txt").write_text(f"{reward:.6f}\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
