#!/usr/bin/env python3
"""Drift gate (spec D-1/D-2/D-3). Runs at PACKAGE BUILD time, not at scoring time.

The contract lives in five places -- contract.yaml, task.toml, instruction.md, policy.yaml and the
evaluator's constants. In the reference AutoLab package those copies had drifted until the described
task was unscoreable. Run this in CI; a non-zero exit fails the build.

Usage:  python tests/contract_check.py --package .
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import tomllib
from pathlib import Path

import yaml

RESOURCE_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".zip", ".tar", ".gz",
                     ".jpg", ".jpeg", ".png", ".npy", ".npz", ".pyc", ".arrow", ".parquet"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", type=Path, default=Path("."))
    args = ap.parse_args()
    pkg = args.package
    problems: list[str] = []

    def fail(msg: str) -> None:
        problems.append(msg)

    contract = yaml.safe_load((pkg / "contract.yaml").read_text())
    task_text = (pkg / "task.toml").read_text()
    task = tomllib.loads(task_text)
    policy = yaml.safe_load((pkg / "policy.yaml").read_text())
    instruction = (pkg / "instruction.md").read_text()
    readme = (pkg / "README.md").read_text()
    evaluate_src = (pkg / "tests" / "evaluate.py").read_text()
    score_src = (pkg / "tests" / "score.py").read_text()
    testsh = (pkg / "tests" / "test.sh").read_text()

    if "harbor-canary GUID" not in task_text.splitlines()[0]:
        fail("task.toml line 1 lacks the harbor-canary marker (all 36 AutoLab tasks carry one)")

    # ---- D-2: the scored metric IS the declared metric ---------------------------------------
    declared = task["metadata"]["optimization"]["metric"]
    if declared != contract["metric"]["headline"]["name"]:
        fail(f"task.toml metric {declared!r} != contract headline "
             f"{contract['metric']['headline']['name']!r}")
    if f'HEADLINE_KEY = "{declared}"' not in score_src:
        fail(f"tests/score.py does not reward the declared metric {declared!r}")
    if f'SIGMA_KEY = "{contract["metric"]["headline"]["sigma_key"]}"' not in score_src:
        fail("tests/score.py SIGMA_KEY != contract metric.headline.sigma_key")

    # ---- the frozen recipe must agree everywhere ----------------------------------------------
    recipe = contract["frozen_recipe"]
    tr = task["rsi"]["frozen_recipe"]
    for key in ("scale", "batch_size", "learning_rate", "train_num_samples", "warmup", "model", "seed"):
        if recipe[key] != tr[key]:
            fail(f"frozen_recipe.{key}: contract {recipe[key]!r} != task.toml {tr[key]!r}")
    if int(contract["per_round_budget"]["samples_seen"]) != int(recipe["train_num_samples"]):
        fail("per_round_budget.samples_seen != frozen_recipe.train_num_samples; the budget IS the "
             "scale config's sample count")
    if int(task["rsi"]["per_round_budget"]["samples_seen"]) != int(recipe["train_num_samples"]):
        fail("task.toml per_round_budget.samples_seen != frozen_recipe.train_num_samples")
    for key in ("tool_gpu_seconds_cap",):
        if int(contract["per_round_budget"][key]) != int(task["rsi"]["per_round_budget"][key]):
            fail(f"per_round_budget.{key} differs between contract.yaml and task.toml")

    # ---- the headline must be UPSTREAM's mean, not a reimplementation --------------------------
    # aggregate_scores.py carries `assert len(df) == 38`. Reimplementing the mean would discard that
    # check along with the code, and the resulting number would still be called average_38.
    if "from aggregate_scores import get_aggregate_scores" not in evaluate_src:
        fail("tests/evaluate.py does not import upstream's get_aggregate_scores; a local mean would "
             "throw away its `assert len(df) == 38`, which is the only thing standing between us and "
             "a silently-39-dataset average")
    if re.search(r"\bmean\(\s*\[?\s*r\[", evaluate_src) or "np.mean(" in evaluate_src:
        fail("tests/evaluate.py appears to compute its own mean over the datasets")
    n_declared = int(contract["metric"]["protocol"]["n_datasets_asserted"])
    if f"N_DATASETS = {n_declared}" not in evaluate_src:
        fail(f"tests/evaluate.py N_DATASETS != contract protocol.n_datasets_asserted ({n_declared})")
    if task["verifier"]["environment"]["env"]["DATACOMP_N_DATASETS"] != str(n_declared):
        fail("task.toml DATACOMP_N_DATASETS disagrees with contract protocol.n_datasets_asserted")
    if f'GUARD_DATASET = "{contract["metric"]["guard"]["dataset"]}"' not in evaluate_src:
        fail("tests/evaluate.py GUARD_DATASET != contract metric.guard.dataset")
    # The 38-vs-40 fact must stay written down: it is implicit in a dropna() and nobody rederives it.
    for name, text in (("contract.yaml", (pkg / "contract.yaml").read_text()),):
        for needle in ("fairface", "utkface"):
            if needle not in text.lower():
                fail(f"{name} no longer names {needle} as one of the two datasets that fall out of "
                     f"the 40; which two drop out is implicit in upstream's dropna() and this is the "
                     f"only place it is recorded")

    # ---- the pool is crawled: this claim may never become true ---------------------------------
    if contract["data"].get("leaderboard_comparable") is not False:
        fail("contract data.leaderboard_comparable must be false. The pool is crawled from the live "
             "web, so it is site-local and time-local and no number here is comparable to the "
             "published benchmark. This is not a caveat that can be revisited by editing it.")
    if not (pkg / "tests" / "pool_check.py").is_file():
        fail("tests/pool_check.py missing; nothing would verify that a chain's rounds share one pool")
    if "pool_check.py" not in testsh:
        fail("tests/test.sh does not run pool_check.py; a re-crawl mid-chain would be undetected")
    if "infra_fail" not in testsh or "exit 2" not in testsh:
        fail("tests/test.sh has no infra_fail path; a changed pool or unstaged asset would be "
             "recorded as a reward of 0.0, which is a measurement the agent did not earn")
    for text, name in ((instruction, "instruction.md"), (readme, "README.md")):
        if "crawl" not in text.lower():
            fail(f"{name} does not tell the reader the pool is crawled and therefore not the "
                 f"published pool")

    # ---- instruction.md must state what the verifier enforces ---------------------------------
    if str(recipe["train_num_samples"]) not in instruction and "12.8M" not in instruction:
        fail("instruction.md never states the samples-seen budget")
    if str(int(contract["per_round_budget"]["tool_gpu_seconds_cap"])) not in instruction:
        fail("instruction.md does not state the tool GPU-second cap")
    if "should not modify any hyper-parameters" not in instruction:
        fail("instruction.md does not quote upstream's own hyper-parameter prohibition "
             "(README.md:190); the agent must know the recipe is not a knob")

    # ---- policy.yaml alignment ---------------------------------------------------------------
    if str(policy.get("contract_version")) != str(contract["contract_version"]):
        fail("policy.yaml contract_version != contract.yaml contract_version")
    if int(policy["constraints"]["samples_seen_per_round"]) != int(recipe["train_num_samples"]):
        fail("policy.yaml samples_seen_per_round != frozen_recipe.train_num_samples")
    if policy["constraints"]["headline_metric"] != declared:
        fail("policy.yaml headline_metric != declared metric")
    if int(policy["constraints"]["tool_gpu_seconds_cap"]) != int(contract["per_round_budget"]["tool_gpu_seconds_cap"]):
        fail("policy.yaml tool_gpu_seconds_cap != contract per_round_budget")
    # tasklist.yml MUST be protected: editing it changes which 38 datasets are averaged, and the
    # change would present itself as a filtering result.
    for path in ("tasklist.yml", "aggregate_scores.py", "scale_configs.py", "evaluate.py"):
        if path not in policy["scope"]["protected_paths"]:
            fail(f"policy.yaml protected_paths does not include {path}")

    allowed = {"id", "severity", "title", "description"}
    for rule in policy.get("rules", []):
        stray = set(rule) - allowed
        if stray:
            fail(f"policy rule {rule.get('id')} has stray keys {sorted(stray)} -- the description "
                 f"was truncated by YAML flow-scalar parsing")
        if len(rule.get("description") or "") < 40:
            fail(f"policy rule {rule.get('id')} description is suspiciously short; check for comma "
                 f"truncation")
        if rule.get("severity") not in {"hard_zero", "warn"}:
            fail(f"policy rule {rule.get('id')} has severity {rule.get('severity')!r}")
    ids = [r.get("id") for r in policy.get("rules", [])]
    if len(ids) != len(set(ids)):
        fail("duplicate rule ids in policy.yaml")
    # The register must be non-empty, or the comparison is against nothing. In the VLM-R1 sibling this
    # same check shipped against a contract.yaml with no `rules` key at all: it compared the empty
    # set, reported healthy, and a hard_zero rule could still be deleted from policy.yaml unnoticed.
    contract_rules = contract.get("rules") or []
    if len(contract_rules) < 7:
        fail(f"contract.yaml lists {len(contract_rules)} rules; the register must enumerate every "
             f"rule the judge enforces (>= 7 here), or the id comparison compares against nothing")
    missing = sorted({r["id"] for r in contract_rules} - set(ids))
    if missing:
        fail(f"contract rules {missing} are absent from policy.yaml, so the judge never sees them")
    # Severity must agree too: demoting a hard_zero to warn keeps the id present while removing the
    # consequence.
    policy_sev = {r.get("id"): r.get("severity") for r in policy.get("rules", [])}
    for rule in contract_rules:
        if rule["id"] in policy_sev and policy_sev[rule["id"]] != rule.get("severity"):
            fail(f"rule {rule['id']} is {rule.get('severity')} in contract.yaml but "
                 f"{policy_sev[rule['id']]} in policy.yaml")

    tests_policy = pkg / "tests" / "policy.yaml"
    if tests_policy.exists():
        if hashlib.sha256((pkg / "policy.yaml").read_bytes()).hexdigest() != \
           hashlib.sha256(tests_policy.read_bytes()).hexdigest():
            fail("tests/policy.yaml differs from policy.yaml -- the exact defect that made the "
                 "reference package unscoreable")

    if int(task["agent"].get("timeout_sec", 0)) <= 0:
        fail("[agent] timeout_sec unset; rounds without an equal wall-clock cap are incomparable")

    for tool in re.findall(r"/task-tools/([A-Za-z0-9_]+\.py)", instruction):
        if not (pkg / "environment" / "task-tools" / tool).is_file():
            fail(f"instruction.md tells the agent to run /task-tools/{tool}, which does not exist")

    # ---- the reward map ----------------------------------------------------------------------
    if "max(0.0," not in score_src:
        fail("tests/score.py no longer floors the reward at 0")
    if "min(1.0" in score_src or "min(1," in score_src:
        fail("tests/score.py caps the reward at 1.0; a chain must be able to exceed the reference or "
             "the slope goes to 0 while the metric climbs")
    if contract["reward"]["per_round"] != "anchored_linear_uncapped":
        fail("contract reward.per_round is not anchored_linear_uncapped")
    if "2.0 * sigma" not in score_src:
        fail("tests/score.py no longer refuses anchors inside the noise band; a 0.3-sigma denominator "
             "turns the reward into amplified noise")
    for name, text in (("instruction.md", instruction), ("task.toml", task_text)):
        if re.search(r"clip\(", text) or "0, 1 )" in text:
            fail(f"{name} describes the reward as clipped; score.py is uncapped. Fix the prose.")

    # ---- shared files must match tests/common.lock -------------------------------------------
    lock = pkg / "tests" / "common.lock"
    if not lock.is_file():
        fail("tests/common.lock missing; run common/sync_common.sh <package>")
    else:
        entries: dict[str, str] = {}
        for line in lock.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                digest, _, name = line.partition("  ")
                entries[name.strip()] = digest.strip()
        if not entries:
            fail("tests/common.lock lists no files; a lock with no entries cannot fail")
        for name, want in entries.items():
            copy = pkg / "tests" / name
            if not copy.is_file():
                fail(f"tests/common.lock names {name}, which is not in tests/")
            elif hashlib.sha256(copy.read_bytes()).hexdigest() != want:
                fail(f"tests/{name} does not match tests/common.lock; it was edited in place instead "
                     f"of in common/, or the package was not re-synced")
        shared_dir = pkg.resolve().parent.parent / "common"
        if shared_dir.is_dir():
            for name, want in entries.items():
                src = shared_dir / name
                if not src.is_file():
                    fail(f"common/{name} is gone but tests/common.lock still names it")
                elif hashlib.sha256(src.read_bytes()).hexdigest() != want:
                    fail(f"tests/{name} is STALE: common/{name} changed since the last sync")

    # ---- upstream claim checker present and wired into BOTH images ---------------------------
    if not (pkg / "tests" / "upstream_check.py").is_file():
        fail("tests/upstream_check.py missing; nothing verifies contract.yaml's citations, and above "
             "all nothing verifies that SCALE_CONFIGS still holds the frozen recipe")
    for df in ("environment/Dockerfile", "tests/Dockerfile"):
        text = (pkg / df).read_text()
        if "upstream_check.py" not in text:
            fail(f"{df} does not run tests/upstream_check.py at build time")
        if not re.search(r"^ARG\s+OPEN_CLIP_VERSION\s*$", text, re.M):
            fail(f"{df} must declare `ARG OPEN_CLIP_VERSION` with NO default: train.py:13 imports "
                 f"open_clip's internal training entry point, so an unpinned open_clip is an "
                 f"unpinned recipe and the build must fail rather than resolve to the current release")
    if "open_clip_version" not in testsh:
        fail("tests/test.sh gate 0 does not refuse a PIN_REQUIRED open_clip")
    if not (pkg / "tests" / "dir_hash.py").is_file():
        fail("tests/dir_hash.py missing; evaluate.py imports it")
    if "from dir_hash import dir_manifest_sha256" not in evaluate_src:
        fail("tests/evaluate.py defines its own directory hash instead of importing dir_hash.py")

    # ---- assets: declared by URL, nothing downloaded ------------------------------------------
    flags: list[str] = []
    assets_path = pkg / "environment" / "assets.yaml"
    if not assets_path.is_file():
        fail("environment/assets.yaml missing; the package must declare every dataset by URL since "
             "it ships none of them")
    else:
        assets_text = assets_path.read_text()
        assets = yaml.safe_load(assets_text)
        state = assets.get("state")
        if state not in {"NOT_STAGED", "STAGED"}:
            fail(f"assets.yaml state is {state!r}; expected NOT_STAGED or STAGED")
        if state == "NOT_STAGED":
            flags.append("ASSETS NOT STAGED")

        # Three programs read this line WITHOUT a YAML parser: task-tools/asset_check.py, which must
        # run with no pyyaml; tests/evaluate.py; and tests/test.sh via grep. In the VLM-R1 sibling an
        # inline comment on it made all three staging gates unable to fire.
        state_lines = [ln for ln in assets_text.splitlines() if ln.startswith("state:")]
        if len(state_lines) != 1:
            fail(f"assets.yaml has {len(state_lines)} top-level `state:` lines; the naive readers "
                 f"take the first")
        elif "#" in state_lines[0]:
            fail(f"inline comment on assets.yaml's state line ({state_lines[0]!r}); the three "
                 f"non-YAML readers cannot fire through it")
        elif state_lines[0].split(":", 1)[1].strip() != state:
            fail("assets.yaml state line does not text-parse to what pyyaml reads")

        items = []
        for section in ("train", "eval", "code"):
            items += list(assets.get(section) or [])
        urls = {i["url"] for i in items if "url" in i}
        toml_urls = {i["url"] for i in task["environment"]["assets"]["item"] if "url" in i}
        if urls != toml_urls:
            fail(f"asset URL sets differ: only in assets.yaml {sorted(urls - toml_urls)}; only in "
                 f"task.toml {sorted(toml_urls - urls)}")
        if task["environment"]["assets"].get("state") != str(state).lower():
            fail("task.toml [environment.assets].state disagrees with assets.yaml state")

        by_id = {i.get("id"): i for i in items}
        env = task["environment"]["env"]
        for asset_id, mount in (("commonpool_metadata", env["DATACOMP_METADATA"]),
                                ("commonpool_shards", env["DATACOMP_SHARDS"])):
            if str(by_id.get(asset_id, {}).get("mount", "")) != mount:
                fail(f"asset {asset_id} mounts at {by_id.get(asset_id, {}).get('mount')!r} but the "
                     f"container env points at {mount!r}")
        eval_root = task["verifier"]["environment"]["env"]["DATACOMP_EVAL_DATA"]
        if str(by_id.get("evalsets", {}).get("mount", "")) != eval_root:
            fail(f"evalsets mounts at {by_id.get('evalsets', {}).get('mount')!r}, not the verifier's "
                 f"{eval_root!r} -- the train/eval split is the containment boundary for RH-RSI-003")
        for asset_id in ("commonpool_metadata", "commonpool_shards"):
            if str(by_id.get(asset_id, {}).get("mount", "")).startswith("/eval-data"):
                fail(f"agent mount for {asset_id} sits under /eval-data")

        blocked = " ".join(str(e.get("url", "") or e.get("url_pattern", "") or
                               e.get("path_pattern", "")) for e in assets.get("must_not_stage") or [])
        for needle, why in (("datacomp_evalsets", "the evaluation sets are the metric"),
                            ("datacomp_1b", "upstream's own published best subset of the xlarge pool")):
            if needle not in blocked:
                fail(f"assets.yaml must_not_stage does not cover {needle} ({why})")

        if state == "STAGED":
            pending = [i.get("id") for i in items for k, v in i.items()
                       if k.endswith(("sha256", "revision")) and v in ("PENDING_DOWNLOAD", "PIN_REQUIRED")]
            if pending:
                fail(f"assets.yaml is STAGED while integrity fields still read PENDING_DOWNLOAD / "
                     f"PIN_REQUIRED for {sorted(set(pending))}")
            for name in ("staged.lock.yaml", "pool.lock.yaml"):
                if not (pkg / "environment" / name).is_file():
                    fail(f"assets.yaml is STAGED but environment/{name} does not exist")
        if not (pkg / "environment" / "fetch_assets.sh").is_file():
            fail("environment/fetch_assets.sh missing; the URLs have no fetcher")

    for key in ("realized_uids", "uid_set_sha256", "download_rate"):
        if str(contract["data"].get(key)) == "PENDING_DOWNLOAD":
            if "POOL NOT FINGERPRINTED" not in flags:
                flags.append("POOL NOT FINGERPRINTED")
    if str(contract["data"].get("open_clip_version")) == "PIN_REQUIRED":
        flags.append("OPEN_CLIP NOT PINNED")

    # ---- reference/ must be structurally unable to reach an image -----------------------------
    ref = pkg / "reference"
    if ref.is_dir():
        di = pkg / ".dockerignore"
        if not di.is_file():
            fail(".dockerignore absent while reference/ exists; both images build from the package "
                 "root, so the anchor improver is one COPY away from the agent")
        elif not any(ln.strip().rstrip("/") == "reference" for ln in di.read_text().splitlines()):
            fail(".dockerignore does not exclude reference/")
        for df in ("environment/Dockerfile", "tests/Dockerfile"):
            for line in (pkg / df).read_text().splitlines():
                s = line.strip()
                if not s.upper().startswith(("COPY", "ADD")):
                    continue
                if "reference/" in s:
                    fail(f"{df} references reference/ in: {s!r}")
                parts = s.split()
                if len(parts) >= 3 and parts[1] in (".", "./"):
                    fail(f"{df} does a whole-context copy ({s!r}); name paths explicitly")
        if not (ref / "improvers" / "passthrough_improver.py").is_file():
            fail("reference/improvers/passthrough_improver.py missing; it defines the reward-1.0 "
                 "anchor AND the null chain")
        if not (ref / "assert_null_chain.py").is_file():
            fail("reference/assert_null_chain.py missing; nothing would check that the null chain's "
                 "subset was constant, that its scores varied, or that it ran on one pool")

    # ---- CONFIGURATION ONLY ------------------------------------------------------------------
    for path in sorted(pkg.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        rel = path.relative_to(pkg).as_posix()
        if path.suffix.lower() in RESOURCE_SUFFIXES:
            fail(f"resource file in a configuration-only package: {rel}")
        elif path.stat().st_size > 1_000_000:
            fail(f"{rel} is {path.stat().st_size} bytes; this package holds configuration, not data")

    unmeasured = [k for k, v in contract["anchors"]["pristine_base"].items() if v == "MEASURE"]
    unmeasured += [k for k, v in contract["anchors"]["reference_improver"].items() if v == "MEASURE"]
    if unmeasured:
        flags.append("ANCHORS UNMEASURED")

    if problems:
        print("CONTRACT DRIFT DETECTED")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("PACKAGE VALID" + (", " + ", ".join(flags) if flags else ""))
    if unmeasured:
        print(f"  anchors still MEASURE: {sorted(set(unmeasured))}")
    if "ASSETS NOT STAGED" in flags:
        print("  assets are declared by URL only; environment/fetch_assets.sh has not been run")
    if "POOL NOT FINGERPRINTED" in flags:
        print("  the pool has not been crawled, so no uid fingerprint exists to hold a chain to")
    if "OPEN_CLIP NOT PINNED" in flags:
        print("  open_clip is PIN_REQUIRED; train.py imports its internals, so the recipe is unpinned")
    if flags:
        print("  the package may be linted; no agent round may be scored yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
