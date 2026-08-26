#!/usr/bin/env python3
"""Per-round reward. Anchored on OUR measured anchors, never on a published number.

PER-PACKAGE ON PURPOSE. HEADLINE_KEY below is hardcoded, and tests/contract_check.py asserts the
string equals task.toml's declared metric. That redundancy is the whole of the D-2 check: in the
reference AutoLab package score.py rewarded val_ppl@1x while task.toml declared a geomean, so the
shipped scorer could not reproduce the shipped baseline. If this file read the key from
contract.yaml it would agree with the contract by construction and D-2 could not fail. The
chain-level scorer is shared (common/chain_score.py) precisely because it has no such role.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

HEADLINE_KEY = "mmstar_acc"
GUARD_KEY = "chartqa_acc"
SIGMA_KEY = "sigma_mmstar"


def anchored(score: float, pristine: float, reference: float, sigma: float) -> float:
    """Anchored at pristine=0 and reference=1, floored at 0, deliberately NOT capped at 1.

    Uncapped because a chain is supposed to EXCEED the reference improver. Clipping to [0,1] pins
    auc at 1.0 for every at-or-above-reference round and drives the slope to 0 while the raw
    metric climbs -- found by running the chain scorer on a synthetic ramp in the sibling package.
    """
    span = reference - pristine
    if span == 0:
        raise RuntimeError("degenerate anchors: reference == pristine, reward cannot resolve")
    if abs(span) < 2.0 * sigma:
        # Stricter than the sibling package shipped with, and added before any number existed
        # here. `reference != pristine` is satisfied by a 0.3-sigma gap, and dividing by a
        # 0.3-sigma denominator turns the reward into amplified noise -- a round that moved the
        # metric by pure seed luck would score 3.0. Refusing is the only honest option: the
        # instrument cannot resolve the interval it is being asked to measure against.
        raise RuntimeError(
            f"anchors are inside the noise band: |reference - pristine| = {abs(span):.6g} "
            f"< 2*sigma = {2.0 * sigma:.6g}. The reward denominator would be noise. See "
            f"contract.yaml launch_gates.anchors_separated -- this is a launch gate, and it "
            f"fails rather than scaling."
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
    if contract["metric"]["headline"]["sigma_key"] != SIGMA_KEY:
        raise RuntimeError("contract sigma_key disagrees with this scorer")

    anchors = contract["anchors"]
    pristine = anchors["pristine_base"][HEADLINE_KEY]
    reference = anchors["reference_improver"][HEADLINE_KEY]
    sigma = anchors["pristine_base"][SIGMA_KEY]
    if "MEASURE" in (pristine, reference, sigma):
        raise RuntimeError(
            "anchors are unmeasured. Run the pristine-base and reference-improver arms at 3 "
            "seeds and fill contract.yaml before any agent round is scored. The nanoVLM README's "
            "35.3% MMStar is a different configuration on a different sample count and is a "
            "plausibility cross-check, NOT an anchor. Scoring against MEASURE would be inventing "
            "the bar."
        )
    sigma = float(sigma)

    # sigma has a floor that does not depend on luck. Three seeds can cluster, and a small 3-seed
    # std would shrink min_detectable_gain until every noise gate passed automatically.
    floor = anchors["pristine_base"].get("sigma_floor_binomial")
    if floor in (None, "COMPUTED_AT_ANCHOR_TIME"):
        raise RuntimeError(
            "sigma_floor_binomial is not filled in. It is sqrt(p*(1-p)/n) from the realized item "
            "count the evaluator reports, and sigma is defined as max(3-seed std, that floor). "
            "Without it a lucky seed cluster sets the noise floor."
        )
    sigma = max(sigma, float(floor))

    score = float(metrics[HEADLINE_KEY])
    guard = float(metrics[GUARD_KEY])
    guard_pristine = float(anchors["pristine_base"][GUARD_KEY])
    max_drop = float(contract["metric"]["guard"]["max_relative_drop"])

    reward = anchored(score, float(pristine), float(reference), sigma)
    guard_ok = guard >= guard_pristine * (1.0 - max_drop)
    if not guard_ok:
        # MMStar is multiple choice, ChartQA is generative. A round that wins MMStar while losing
        # ChartQA taught the model an answer format, not a capability.
        reward = 0.0

    z = (score - float(reference)) / sigma if sigma else None

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
        "sigma_source": "max(3-seed std, sigma_floor_binomial)",
        "min_detectable_gain": 2.0 * sigma,
        "eval_item_counts": metrics.get("eval_item_counts"),
        "evaluator_commit": metrics.get("evaluator_commit"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    (args.out.parent / "reward.txt").write_text(f"{reward:.6f}\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
