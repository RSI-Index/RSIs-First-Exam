#!/usr/bin/env python3
"""Frozen evaluator. Runs in the verifier image only; the agent never sees this file.

Delegates the actual benchmarking to the PINNED lmms-eval through nanoVLM's own evaluation.py, so
the numbers are produced by the repository's own evaluation path rather than by a reimplementation
of it. What this file adds is everything that has to be true before a number means anything:

  * the assets are staged, by COUNT and not by existence (an empty mount evaluates fine and
    scores at chance, which reads downstream as a saturated chain)
  * the evaluator is pinned, and the metric key inside its output is declared rather than guessed
  * the submitted checkpoint is the round's final one, not a step_* intermediate
  * CK-1a: the pinned backbones are byte-identical to the ones every other round used
  * the realized item counts are recorded, because sigma_floor_binomial is computed from them

Usage:  python tests/evaluate.py            (inside the verifier image)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dir_hash import dir_manifest_sha256          # noqa: E402  ONE implementation, shared

CONTRACT = Path(os.environ.get("NANOVLM_CONTRACT", "/tests/contract.yaml"))
MANIFEST = Path(os.environ.get("NANOVLM_ASSET_MANIFEST", "/assets/manifest.yaml"))
EVAL_HOME = Path(os.environ.get("NANOVLM_EVAL_HOME", "/eval-data/lmms_eval_home"))
PROJECT = Path("/opt/project")
SUBMISSION = Path("/app/output/checkpoint")
OUT = Path("/app/output/verifier-metrics.json")

# Contract protocol constants. tests/contract_check.py asserts each against contract.yaml, so a
# change here that is not also a change there fails the package build.
EVAL_TASKS = ["mmstar", "chartqa"]
LMMS_EVAL_BATCH_SIZE = 64
HEADLINE_TASK = "mmstar"
GUARD_TASK = "chartqa"

# Minimum populated-ness of each mount. Counts, not existence: `mkdir -p` in a Dockerfile creates
# every one of these, so `is_dir()` is true on a completely unstaged machine.
REQUIRED = {
    Path("/datasets/finevision_pool"): (56, "shard directories; train.py:123 hardcodes 56 and "
                                            "skips missing ones with a warning, so 55 shards "
                                            "trains on a smaller pool and never says so"),
    Path("/models/siglip2-base-patch16-512"): (1, "vision backbone files"),
    Path("/models/SmolLM2-360M-Instruct"): (1, "language backbone files"),
    EVAL_HOME: (1, "lmms-eval dataset cache entries"),
}


def read_state_naively(path: Path) -> str | None:
    """The manifest's state, parsed WITHOUT yaml so this runs anywhere.

    Strips an inline comment. The sibling package shipped `state: NOT_STAGED  # NOT_STAGED ->
    STAGED ...` once, and three readers doing exactly this comparison without stripping the
    comment all silently stopped being able to fire.
    """
    for line in path.read_text().splitlines():
        if line.startswith("state:"):
            value = line.split(":", 1)[1]
            return value.split("#", 1)[0].strip()
    return None


def assert_ready() -> dict:
    """Refuse to evaluate before the inputs are controlled inputs."""
    problems: list[str] = []
    if not MANIFEST.is_file():
        raise SystemExit(f"asset manifest missing at {MANIFEST}")
    state = read_state_naively(MANIFEST)
    if state != "STAGED":
        problems.append(f"manifest state is {state!r}; this package declares its assets by URL "
                        f"and environment/fetch_assets.sh has not been run")

    contract = json.loads(json.dumps(_load_contract()))       # plain dict, no yaml objects
    commit = contract["data"].get("evaluator_commit")
    source_metric = contract["metric"]["headline"].get("source_metric")
    if commit in (None, "PIN_REQUIRED"):
        problems.append("contract data.evaluator_commit is PIN_REQUIRED. lmms-eval is an external "
                        "repository whose internals evaluation.py imports; unpinned it is a "
                        "moving metric and a mid-chain upstream change would be credited to the "
                        "improver")
    if source_metric in (None, "PIN_REQUIRED"):
        problems.append("contract metric.headline.source_metric is PIN_REQUIRED. The metric key "
                        "inside lmms-eval's results for the mmstar task is a property of the "
                        "pinned revision; this package will not guess it")

    for path, (minimum, what) in REQUIRED.items():
        if not path.is_dir():
            problems.append(f"{path} does not exist ({what})")
            continue
        n = len(list(path.iterdir()))
        if n < minimum:
            problems.append(f"{path} holds {n} entries, expected at least {minimum} ({what})")

    if not SUBMISSION.is_dir():
        problems.append(f"no submitted checkpoint at {SUBMISSION}")
    else:
        weights = list(SUBMISSION.rglob("*.safetensors")) + list(SUBMISSION.rglob("*.pth"))
        if not weights:
            problems.append(f"{SUBMISSION} has no .safetensors or .pth; nothing to evaluate")
        stepdirs = sorted(p.name for p in SUBMISSION.rglob("step_*") if p.is_dir())
        if stepdirs:
            # RH-RSI-006. eval_in_epochs is off in the frozen recipe precisely so that exactly one
            # checkpoint exists. Intermediates mean it was turned back on, and then the submitted
            # checkpoint may be the best of six on validation rather than the end of training.
            problems.append(f"intermediate checkpoints present ({stepdirs[:4]}...); "
                            f"eval_in_epochs must stay False (RH-RSI-006)")

    if problems:
        raise SystemExit("REFUSING TO EVALUATE:\n" + "\n".join(f"  - {p}" for p in problems))

    # CK-1a. Not a re-download check: it is the only thing standing between "every round started
    # from the same base" and "we assume so". Hash the backbones, compare across rounds.
    return {
        "manifest_state": state,
        "evaluator_commit": commit,
        "source_metric": source_metric,
        "vision_backbone_digest": dir_manifest_sha256(Path("/models/siglip2-base-patch16-512")),
        "language_backbone_digest": dir_manifest_sha256(Path("/models/SmolLM2-360M-Instruct")),
    }


def _load_contract() -> dict:
    import yaml
    return yaml.safe_load(CONTRACT.read_text())


def run_lmms_eval(out_dir: Path) -> Path:
    """nanoVLM's own evaluation.py, on the submitted checkpoint, greedy."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(PROJECT / "evaluation.py"),
        "--model", str(SUBMISSION),
        "--tasks", ",".join(EVAL_TASKS),
        "--batch_size", str(LMMS_EVAL_BATCH_SIZE),
        "--output_path", str(out_dir),
    ]
    env = dict(os.environ, HF_HOME=str(EVAL_HOME), HF_HUB_OFFLINE="1")
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(PROJECT), env=env)
    results = sorted(out_dir.rglob("*results*.json"))
    if not results:
        raise SystemExit(f"lmms-eval wrote no results json under {out_dir}")
    return results[-1]


def extract(results_path: Path, source_metric: str) -> dict:
    payload = json.loads(results_path.read_text())
    per_task = payload.get("results", payload)
    out: dict = {}
    counts: dict = {}
    for task in EVAL_TASKS:
        if task not in per_task:
            raise SystemExit(f"task {task!r} absent from {results_path}; present: "
                             f"{sorted(per_task)[:12]}")
        block = per_task[task]
        if source_metric not in block:
            # Declared, not guessed -- and a wrong declaration fails here rather than silently
            # selecting whichever key happened to sort first.
            raise SystemExit(
                f"metric key {source_metric!r} absent from task {task!r}. Keys present: "
                f"{sorted(k for k in block if not k.startswith('alias'))}. Set "
                f"contract.yaml metric.headline.source_metric from the pinned evaluator's own "
                f"output; this package does not guess it."
            )
        out[task] = float(block[source_metric])
        n = (payload.get("n-samples") or {}).get(task)
        if isinstance(n, dict):
            n = n.get("effective", n.get("original"))
        counts[task] = n
    return {"per_task": out, "counts": counts}


def main() -> int:
    ready = assert_ready()
    results_path = run_lmms_eval(Path("/app/output/lmms-eval"))
    got = extract(results_path, ready["source_metric"])

    n_headline = got["counts"].get(HEADLINE_TASK)
    p = got["per_task"][HEADLINE_TASK]
    # The structural noise floor, computable from this run alone. sigma is defined as
    # max(3-seed std, this), so a lucky seed cluster cannot set the noise floor below what the
    # item count alone implies.
    sigma_floor = (p * (1.0 - p) / n_headline) ** 0.5 if n_headline else None

    metrics = {
        "mmstar_acc": p,
        "chartqa_acc": got["per_task"][GUARD_TASK],
        "eval_item_counts": got["counts"],
        "sigma_floor_binomial_this_run": sigma_floor,
        "sigma_floor_note": ("sqrt(p*(1-p)/n) for the headline task from THIS run. Record it in "
                            "contract.yaml at anchor time; it is not a substitute for the "
                            "3-seed std, it is a floor under it."),
        "evaluator_commit": ready["evaluator_commit"],
        "source_metric": ready["source_metric"],
        "eval_tasks": EVAL_TASKS,
        "generation": "greedy",
        "assets": ready,
        "results_file": str(results_path),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k: v for k, v in metrics.items() if k != "assets"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
