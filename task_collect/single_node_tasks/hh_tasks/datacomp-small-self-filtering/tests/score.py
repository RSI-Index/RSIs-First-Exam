#!/usr/bin/env python3
"""Per-round reward. Anchored on OUR measured anchors -- there are no published ones to misuse here.

PER-PACKAGE ON PURPOSE. HEADLINE_KEY is hardcoded and tests/contract_check.py asserts it equals
task.toml's declared metric. That redundancy is the whole of the D-2 check: in the reference AutoLab
package score.py rewarded one metric while task.toml declared another, so the shipped scorer could
not reproduce the shipped baseline. If this file read the key from contract.yaml it would agree by
construction and D-2 could not fail. chain_score.py is shared precisely because it has no such role.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

HEADLINE_KEY = "average_38"
GUARD_KEY = "imagenet_top1"
SIGMA_KEY = "sigma_average"


def anchored(score: float, pristine: float, reference: float, sigma: float) -> float:
    """Anchored at pristine=0 and reference=1, floored at 0, deliberately NOT capped at 1."""
    span = reference - pristine
    if span == 0:
        raise RuntimeError("degenerate anchors: reference == pristine, reward cannot resolve")
    if abs(span) < 2.0 * sigma:
        # `reference != pristine` is satisfied by a 0.3-sigma gap, and dividing by a 0.3-sigma
        # denominator makes the reward amplified noise -- a round that moved the metric by seed luck
        # alone would score 3.0. Refusing is the only honest option.
        raise RuntimeError(
            f"anchors are inside the noise band: |reference - pristine| = {abs(span):.6g} "
            f"< 2*sigma = {2.0 * sigma:.6g}. See contract.yaml launch_gates.anchors_separated."
        )
    return max(0.0, (score - pristine) / span)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", type=Path, default=Path("/tests/contract.yaml"))
    ap.add_argument("--metrics", type=Path, default=Path("/logs/verifier/metrics.json"))
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--out", type=Path, default=Path("/logs/verifier/reward.json"))
    args = ap.parse_args()

    contract = yaml.safe_load(args.contract.read_text())
    metrics = json.loads(args.metrics.read_text())

    declared = contract["metric"]["headline"]["name"]
    if declared != HEADLINE_KEY:
        raise RuntimeError(f"D-2 violated: contract headline is {declared}, this scorer rewards "
                           f"{HEADLINE_KEY}")
    if contract["metric"]["headline"]["sigma_key"] != SIGMA_KEY:
        raise RuntimeError("contract sigma_key disagrees with this scorer")
    if metrics.get("n_datasets_scored") != contract["metric"]["protocol"]["n_datasets_asserted"]:
        raise RuntimeError(
            f"the evaluator scored {metrics.get('n_datasets_scored')} datasets, the contract asserts "
            f"{contract['metric']['protocol']['n_datasets_asserted']}. A mean over a different "
            f"number of datasets is a different metric wearing the same name."
        )

    anchors = contract["anchors"]
    pristine = anchors["pristine_base"][HEADLINE_KEY]
    reference = anchors["reference_improver"][HEADLINE_KEY]
    sigma = anchors["pristine_base"][SIGMA_KEY]
    if "MEASURE" in (pristine, reference, sigma):
        raise RuntimeError(
            "anchors are unmeasured. The datacomp repository states NO accuracy numbers at all -- "
            "README.md:117 refers every baseline to the paper's Table 3 -- and our pool is a "
            "site-local crawl, so a published number would be the wrong denominator even if we "
            "were allowed to use one. Run the pristine and no_filter arms at 3 seeds first."
        )
    sigma = float(sigma)

    score = float(metrics[HEADLINE_KEY])
    guard = float(metrics[GUARD_KEY])
    guard_pristine = float(anchors["pristine_base"][GUARD_KEY])
    max_drop = float(contract["metric"]["guard"]["max_relative_drop"])

    reward = anchored(score, float(pristine), float(reference), sigma)
    guard_ok = guard >= guard_pristine * (1.0 - max_drop)
    if not guard_ok:
        # aggregate_scores.py reports ImageNet separately from the Average because the two can move
        # in opposite directions. A filter that lifts the mean while wrecking ImageNet swapped the
        # distribution; it did not improve the data.
        reward = 0.0

    z = (score - float(reference)) / sigma if sigma else None
    result = {
        "round": args.round,
        "reward": round(reward, 6),
        "reward_scale_note": "1.0 == matched no_filter at the same 12.8M-sample budget; >1.0 == beat it",
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
        "min_detectable_gain": 2.0 * sigma,
        "n_datasets_scored": metrics.get("n_datasets_scored"),
        "pool_uid_set_sha256": (metrics.get("assets") or {}).get("pool_uid_set_sha256"),
        "leaderboard_comparable": False,
        "subscores": {k: metrics.get(k) for k in ("imagenet_shifts", "vtab", "retrieval")},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    (args.out.parent / "reward.txt").write_text(f"{reward:.6f}\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
