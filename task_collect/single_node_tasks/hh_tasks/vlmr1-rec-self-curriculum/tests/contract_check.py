#!/usr/bin/env python3
"""Drift gate (spec D-1/D-2/D-3). Runs at PACKAGE BUILD time, not at scoring time.

The contract lives in five places -- contract.yaml, task.toml, instruction.md, policy.yaml,
and the evaluator's constants. In the reference package those copies had drifted so far that
the described task was unscoreable: instruction.md offered a 1.005x parameter budget and a
caller-declared model config, tests/policy.yaml (the copy test.sh loaded) forbade the change
the task was about, and evaluate.py hardcoded the pristine name plus exact parameter equality.

This file exists so that cannot happen silently. Run it in CI; a non-zero exit fails the build.

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


def fail(problems: list[str], message: str) -> None:
    problems.append(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=Path("."))
    args = parser.parse_args()
    pkg = args.package
    problems: list[str] = []

    contract = yaml.safe_load((pkg / "contract.yaml").read_text())
    task = tomllib.loads((pkg / "task.toml").read_text())
    policy = yaml.safe_load((pkg / "policy.yaml").read_text())
    instruction = (pkg / "instruction.md").read_text()
    evaluate_src = (pkg / "tests" / "evaluate.py").read_text()

    # ---- canary --------------------------------------------------------------------
    if "harbor-canary GUID" not in (pkg / "task.toml").read_text().splitlines()[0]:
        fail(problems, "task.toml line 1 lacks the harbor-canary marker (all 36 AutoLab tasks carry one)")

    # ---- D-2: the scored metric IS the declared metric ------------------------------
    declared = task["metadata"]["optimization"]["metric"]
    if declared != contract["metric"]["headline"]["name"]:
        fail(problems, f"task.toml metric {declared!r} != contract headline "
                       f"{contract['metric']['headline']['name']!r}")
    if f'HEADLINE_KEY = "{declared}"' not in (pkg / "tests" / "score.py").read_text():
        fail(problems, f"tests/score.py does not reward the declared metric {declared!r}")

    # ---- budget agreement across contract / task.toml / evaluator -------------------
    budget = contract["per_round_budget"]
    toml_budget = task["rsi"]["per_round_budget"]
    for key in ("optimizer_steps", "completions_per_step", "num_generations", "max_completion_length"):
        if int(budget[key]) != int(toml_budget[key]):
            fail(problems, f"per_round_budget.{key}: contract {budget[key]} != task.toml {toml_budget[key]}")
    for key, const in (("optimizer_steps", "BUDGET_OPTIMIZER_STEPS"),
                       ("completions_per_step", "BUDGET_COMPLETIONS_PER_STEP"),
                       ("num_generations", "BUDGET_NUM_GENERATIONS"),
                       ("max_completion_length", "BUDGET_MAX_COMPLETION_LENGTH")):
        match = re.search(rf"^{const} = (\d+)$", evaluate_src, re.MULTILINE)
        if not match:
            fail(problems, f"tests/evaluate.py missing constant {const}")
        elif int(match.group(1)) != int(budget[key]):
            fail(problems, f"{const}={match.group(1)} != contract {budget[key]}")

    # ---- eval protocol agreement ----------------------------------------------------
    protocol = contract["metric"]["protocol"]
    for key, const in (("samples_per_dataset", "NUM_SAMPLES"),
                       ("sampling_seed", "SAMPLING_SEED"),
                       ("max_new_tokens", "MAX_NEW_TOKENS")):
        match = re.search(rf"^{const} = (\d+)$", evaluate_src, re.MULTILINE)
        if not match or int(match.group(1)) != int(protocol[key]):
            fail(problems, f"tests/evaluate.py {const} disagrees with contract protocol.{key}")
    if f"IOU_THRESHOLD = {protocol['iou_threshold']}" not in evaluate_src:
        fail(problems, "tests/evaluate.py IOU_THRESHOLD disagrees with contract")

    # ---- instruction.md must state the same budget the verifier enforces ------------
    if str(budget["optimizer_steps"]) not in instruction:
        fail(problems, "instruction.md never states the optimizer-step budget the verifier enforces")
    if "max_steps 150" not in instruction and f"max_steps {budget['optimizer_steps']}" not in instruction:
        fail(problems, "instruction.md example command does not carry the contract step budget")

    # ---- policy.yaml alignment ------------------------------------------------------
    if policy.get("contract_version") != contract["contract_version"]:
        fail(problems, "policy.yaml contract_version != contract.yaml contract_version")
    if int(policy["constraints"]["optimizer_steps_per_round"]) != int(budget["optimizer_steps"]):
        fail(problems, "policy.yaml optimizer_steps_per_round != contract per_round_budget")
    if policy["constraints"]["headline_metric"] != declared:
        fail(problems, "policy.yaml headline_metric != declared metric")

    # ---- policy rules must survive the YAML round-trip ------------------------------
    # The reference package wrote rules as flow mappings, so YAML truncated every description
    # at its first comma and turned the rest into junk keys -- while still parsing cleanly.
    # policy_check.py json.dumps()es the parsed policy into the judge prompt, so the judge was
    # receiving "Do not modify" in place of RH-001. Verified against both copies in bundle.zip.
    allowed_rule_keys = {"id", "severity", "title", "description"}
    for rule in policy.get("rules", []):
        stray = set(rule) - allowed_rule_keys
        if stray:
            fail(problems, f"policy rule {rule.get('id')} has stray keys {sorted(stray)} -- the "
                           f"description was truncated by YAML flow-scalar parsing")
        desc = rule.get("description") or ""
        if len(desc) < 40:
            fail(problems, f"policy rule {rule.get('id')} description is {len(desc)} chars; "
                           f"suspiciously short, check for comma truncation")
        if rule.get("severity") not in {"hard_zero", "warn"}:
            fail(problems, f"policy rule {rule.get('id')} has severity {rule.get('severity')!r}")
    rule_ids = [r.get("id") for r in policy.get("rules", [])]
    if len(rule_ids) != len(set(rule_ids)):
        fail(problems, "duplicate rule ids in policy.yaml")
    # Every rule the contract declares must exist in the policy the JUDGE reads. Backported from the
    # two later packages, where the mutation suite showed this package could lose a whole hard_zero
    # rule from policy.yaml without any gate noticing -- the contract would still describe it and
    # nothing would enforce it.
    contract_rules = contract.get("rules") or []
    if len(contract_rules) < 8:
        # The first version of this check read contract.get("rules", []) on a contract.yaml that had
        # NO rules key -- so the comparison was against the empty set and the check was vacuous. It
        # reported healthy while a hard_zero rule could still be deleted from policy.yaml unnoticed.
        # A register that can be empty is a gate that cannot fail, so the register's own presence is
        # now the first thing asserted.
        fail(problems, f"contract.yaml lists {len(contract_rules)} rules; the register must "
                       f"enumerate every rule the judge is expected to enforce (>= 8 here), or the "
                       f"id comparison below compares against nothing")
    contract_rule_ids = {r["id"] for r in contract_rules}
    missing_rules = sorted(contract_rule_ids - set(rule_ids))
    if missing_rules:
        fail(problems, f"contract rules {missing_rules} are absent from policy.yaml, so the judge "
                       f"never sees them")
    # Severity must agree too: demoting a hard_zero to warn in policy.yaml would keep the id present
    # while removing the consequence.
    policy_sev = {r.get("id"): r.get("severity") for r in policy.get("rules", [])}
    for rule in contract_rules:
        if rule["id"] in policy_sev and policy_sev[rule["id"]] != rule.get("severity"):
            fail(problems, f"rule {rule['id']} is {rule.get('severity')} in contract.yaml but "
                           f"{policy_sev[rule['id']]} in policy.yaml")

    # ---- ONE policy copy ------------------------------------------------------------
    tests_policy = pkg / "tests" / "policy.yaml"
    if tests_policy.exists():
        a = hashlib.sha256((pkg / "policy.yaml").read_bytes()).hexdigest()
        b = hashlib.sha256(tests_policy.read_bytes()).hexdigest()
        if a != b:
            fail(problems, "tests/policy.yaml differs from policy.yaml -- this is the exact defect "
                           "that made the reference package unscoreable")

    # ---- agent timeout must be set --------------------------------------------------
    if int(task["agent"].get("timeout_sec", 0)) <= 0:
        fail(problems, "[agent] timeout_sec unset; rounds without an equal wall-clock cap are incomparable")

    # ---- the shared gate must actually be present -----------------------------------
    # tests/policy_check.py is the reward-integrity gate shared across task packages. It is
    # copied in verbatim rather than reimplemented here; a divergent private copy is how the
    # reference package's contract drifted in the first place.
    if not (pkg / "tests" / "policy_check.py").is_file():
        fail(problems, "tests/policy_check.py absent -- copy the shared gate in verbatim "
                       "(see README.md step 0); tests/Dockerfile COPYs it and test.sh runs it")

    # ---- every tool instruction.md tells the agent to run must exist ----------------
    # The reference package's instruction.md called two programs at five sites that were in
    # no Dockerfile and in no archive. Nothing catches that except a check like this one.
    for tool in re.findall(r"/task-tools/([A-Za-z0-9_]+\.py)", instruction):
        if not (pkg / "environment" / "task-tools" / tool).is_file():
            fail(problems, f"instruction.md tells the agent to run /task-tools/{tool}, "
                           f"which does not exist in environment/task-tools/")

    # ---- the reward map must be described identically everywhere ---------------------
    # This one was found by drift, not by design: score.py's upper cap was removed after a
    # synthetic chain showed it pinning auc at 1.0, and three prose copies of the formula kept
    # the old clip(...,0,1). A reward description the agent reads and the scorer does not
    # implement is the reference package's D-2 defect wearing different clothes.
    score_src = (pkg / "tests" / "score.py").read_text()
    if "max(0.0," not in score_src:
        fail(problems, "tests/score.py no longer floors the reward at 0")
    if "min(1.0" in score_src or "min(1," in score_src:
        fail(problems, "tests/score.py caps the reward at 1.0; a chain must be able to exceed "
                       "the reference improver or the slope goes to 0 while the metric climbs")
    # Also backported: score.py refuses anchors that sit inside the noise band. `reference !=
    # pristine` is satisfied by a 0.3-sigma gap, and dividing by a 0.3-sigma denominator makes the
    # reward amplified noise. The refusal existed here but nothing asserted it stayed.
    if "2.0 * sigma" not in score_src:
        fail(problems, "tests/score.py no longer refuses anchors inside the noise band; a "
                       "sub-sigma denominator turns every reward into amplified noise")
    for name, text in (("instruction.md", instruction),
                       ("task.toml", (pkg / "task.toml").read_text())):
        if re.search(r"clip\(\s*\(?\s*(lisa|s)_k", text) or "0, 1 )" in text:
            fail(problems, f"{name} still describes the reward as clipped to [0,1]; score.py is "
                           f"uncapped. Fix the prose, not the scorer.")

    # ---- ONE implementation of the CK-1 hash ----------------------------------------
    # tests/dir_hash.py is imported by the evaluator and executed by the fetcher. Two copies
    # would make CK-1 fail every round and read as tampering.
    if not (pkg / "tests" / "dir_hash.py").is_file():
        fail(problems, "tests/dir_hash.py missing; evaluate.py and fetch_assets.sh share it")
    if "from dir_hash import dir_manifest_sha256" not in evaluate_src:
        fail(problems, "tests/evaluate.py defines its own directory hash instead of importing "
                       "tests/dir_hash.py -- the fetcher records the other one")
    if "def dir_manifest_sha256" in evaluate_src:
        fail(problems, "tests/evaluate.py still defines dir_manifest_sha256; it must import it")

    # ---- the shared files must match tests/common.lock -------------------------------
    # chain_score.py / dir_hash.py / policy_check.py are copies of _rsi_scout/common/*, because
    # a Dockerfile COPY cannot reach outside its build context. The lock makes the copies
    # checkable from a standalone tarball; common/ being reachable makes STALENESS checkable
    # too, which is the failure the lock alone cannot see -- a lock regenerated from an edited
    # copy is perfectly self-consistent.
    lock = pkg / "tests" / "common.lock"
    if not lock.is_file():
        fail(problems, "tests/common.lock missing; run common/sync_common.sh <package>")
    else:
        entries = {}
        for line in lock.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            digest, _, name = line.partition("  ")
            entries[name.strip()] = digest.strip()
        if not entries:
            fail(problems, "tests/common.lock lists no files; a lock with no entries is a gate "
                           "that cannot fail")
        for name, want in entries.items():
            copy = pkg / "tests" / name
            if not copy.is_file():
                fail(problems, f"tests/common.lock names {name}, which is not in tests/")
                continue
            got = hashlib.sha256(copy.read_bytes()).hexdigest()
            if got != want:
                fail(problems, f"tests/{name} does not match tests/common.lock "
                               f"({got[:12]} != {want[:12]}); it was edited in place instead of "
                               f"in common/, or the package was not re-synced")
        upstream_common = pkg.resolve().parent.parent / "common"
        if upstream_common.is_dir():
            for name in entries:
                src = upstream_common / name
                if not src.is_file():
                    fail(problems, f"common/{name} is gone but tests/common.lock still names it")
                elif hashlib.sha256(src.read_bytes()).hexdigest() != entries[name]:
                    fail(problems, f"tests/{name} is STALE: common/{name} has changed since the "
                                   f"last sync. Run common/sync_common.sh on this package.")

    # ---- upstream claim checker present and wired into BOTH images -------------------
    if not (pkg / "tests" / "upstream_check.py").is_file():
        fail(problems, "tests/upstream_check.py missing; nothing verifies contract.yaml's "
                       "citations into upstream code, and the package holds no copy of it")
    for df in ("environment/Dockerfile", "tests/Dockerfile"):
        if "upstream_check.py" not in (pkg / df).read_text():
            fail(problems, f"{df} does not run tests/upstream_check.py at build time")

    # ---- assets: declared by URL, nothing downloaded ---------------------------------
    assets_path = pkg / "environment" / "assets.yaml"
    assets_not_staged = False
    if not assets_path.is_file():
        fail(problems, "environment/assets.yaml missing; the package must declare every dataset "
                       "and checkpoint by URL since it ships none of them")
    else:
        assets_text = assets_path.read_text()
        assets = yaml.safe_load(assets_text)
        state = assets.get("state")
        if state not in {"NOT_STAGED", "STAGED"}:
            fail(problems, f"assets.yaml state is {state!r}; expected NOT_STAGED or STAGED")
        assets_not_staged = state == "NOT_STAGED"

        # Three programs read this one line WITHOUT a YAML parser -- asset_check.py (must run in
        # the agent image with no pyyaml), evaluate.py, and test.sh (grep, before python starts).
        # It shipped once with an inline comment, and all three then compared the value
        # "NOT_STAGED   # NOT_STAGED -> STAGED ..." against "NOT_STAGED": every staging gate was
        # unable to fire. Caught by a fixture test, not by review. Keep the line bare.
        state_lines = [ln for ln in assets_text.splitlines() if ln.startswith("state:")]
        if len(state_lines) != 1:
            fail(problems, f"assets.yaml has {len(state_lines)} top-level `state:` lines; the "
                           f"naive readers take the first one")
        elif "#" in state_lines[0]:
            fail(problems, f"inline comment on assets.yaml's state line ({state_lines[0]!r}); "
                           f"the three non-YAML readers of this value cannot fire through it")
        elif state_lines[0].split(":", 1)[1].strip() != state:
            fail(problems, "assets.yaml state line does not text-parse to the same value pyyaml "
                           "reads; the naive readers and the gate would disagree")

        items = []
        for section in ("model", "train", "eval", "code"):
            items += list(assets.get(section) or [])
        urls = {i["url"] for i in items if "url" in i}

        # task.toml must enumerate the same URLs. Two hand-maintained lists of URLs is exactly
        # the drift this gate exists for.
        toml_items = task.get("environment", {}).get("assets", {}).get("item", [])
        toml_urls = {i["url"] for i in toml_items if "url" in i}
        if urls != toml_urls:
            only_yaml = sorted(urls - toml_urls)
            only_toml = sorted(toml_urls - urls)
            fail(problems, f"asset URL sets differ: only in assets.yaml {only_yaml}; "
                           f"only in task.toml {only_toml}")
        if task["environment"]["assets"].get("state") != state.lower():
            fail(problems, "task.toml [environment.assets].state disagrees with assets.yaml state")

        # Mounts must match the paths the containers are actually configured with.
        env = task["environment"]["env"]
        want = {
            env["VLMR1_BASE_MODEL"]: "base_policy",
            env["VLMR1_TRAIN_POOL"]: "rec_annotations",
            env["VLMR1_TRAIN_IMAGES"]: "coco_train2014",
        }
        by_id = {i.get("id"): i for i in items}
        for mount, asset_id in want.items():
            declared = str(by_id.get(asset_id, {}).get("mount", ""))
            if declared != mount:
                fail(problems, f"asset {asset_id} mounts at {declared!r} but the container env "
                               f"points at {mount!r}")
        eval_root = task["verifier"]["environment"]["env"]["VLMR1_EVAL_DATA"]
        for asset_id in ("rec_eval_jsons", "lisa_test_images"):
            declared = str(by_id.get(asset_id, {}).get("mount", ""))
            if not declared.startswith(eval_root):
                fail(problems, f"eval asset {asset_id} mounts at {declared!r}, outside the "
                               f"verifier's {eval_root!r} -- the train/eval split is the "
                               f"containment boundary for RH-RSI-003")
        # The agent's mounts must not be under the eval root, and vice versa.
        for mount in want:
            if str(mount).startswith(eval_root):
                fail(problems, f"agent mount {mount} sits under the verifier-only {eval_root}")

        # contract.yaml carries the same URLs (it is the single source of truth readers open).
        data = contract.get("data", {})
        contract_urls = {
            data.get("train_pool", {}).get("annotations_url"),
            data.get("train_pool", {}).get("images_url"),
            data.get("eval", {}).get("lisa_images_url"),
            data.get("eval", {}).get("jsons_url"),
        } - {None}
        missing = sorted(contract_urls - urls)
        if missing:
            fail(problems, f"contract.yaml names data URLs absent from assets.yaml: {missing}")
        if data.get("state") != state:
            fail(problems, "contract.yaml data.state disagrees with assets.yaml state")

        # STAGED is a claim about recorded bitstreams, not a flag. A manifest that can be
        # flipped without hashes is a gate that cannot fail.
        if state == "STAGED":
            pending = [i.get("id") for i in items
                       for k, v in i.items()
                       if k.endswith(("sha256", "revision")) and v == "PENDING_DOWNLOAD"]
            if pending:
                fail(problems, f"assets.yaml is marked STAGED while integrity fields still read "
                               f"PENDING_DOWNLOAD for {sorted(set(pending))}")
            if not (pkg / "environment" / "staged.lock.yaml").is_file():
                fail(problems, "assets.yaml is marked STAGED but environment/staged.lock.yaml "
                               "does not exist; the fetcher writes it")
        if not (pkg / "environment" / "fetch_assets.sh").is_file():
            fail(problems, "environment/fetch_assets.sh missing; the URLs have no fetcher")

    # ---- reference/ must be structurally unable to reach an image --------------------
    # reference/improvers/sanity_improver.py is a working candidate solution to the task. Both
    # images build from the package root, so the only thing between it and the agent is this
    # exclusion -- and an exclusion nobody checks is one that gets deleted during a refactor.
    ref = pkg / "reference"
    if ref.is_dir():
        dockerignore = pkg / ".dockerignore"
        if not dockerignore.is_file():
            fail(problems, ".dockerignore absent while reference/ exists; both images build from "
                           "the package root, so the solution code is one COPY away from the agent")
        elif not any(ln.strip().rstrip("/") == "reference"
                     for ln in dockerignore.read_text().splitlines()):
            fail(problems, ".dockerignore does not exclude reference/ (which holds a working "
                           "candidate solution)")
        for df in ("environment/Dockerfile", "tests/Dockerfile"):
            for line in (pkg / df).read_text().splitlines():
                stripped = line.strip()
                if not stripped.upper().startswith(("COPY", "ADD")):
                    continue
                if "reference/" in stripped:
                    fail(problems, f"{df} references reference/ in: {stripped!r}")
                # A bare `COPY . <dest>` pulls the whole context; .dockerignore is then the only
                # protection, and relying on a single mechanism for the solution key is thin.
                parts = stripped.split()
                if len(parts) >= 3 and parts[1] in (".", "./"):
                    fail(problems, f"{df} does a whole-context copy ({stripped!r}); name the "
                                   f"paths explicitly so reference/ cannot be included by default")
        if not (ref / "improvers" / "passthrough_improver.py").is_file():
            fail(problems, "reference/improvers/passthrough_improver.py missing; it defines the "
                           "reward-1.0 anchor AND the null chain")
        if not (ref / "assert_null_chain.py").is_file():
            fail(problems, "reference/assert_null_chain.py missing; nothing would check that the "
                           "null chain's curriculum was actually constant across rounds")

    # ---- CONFIGURATION ONLY: no resource bytes may live in this package --------------
    # Mechanical, because "we only ship config" degrades the moment someone drops a checkpoint
    # or a 2 GB zip in for convenience and the package silently becomes unshippable.
    RESOURCE_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".zip", ".tar",
                         ".gz", ".jpg", ".jpeg", ".png", ".npy", ".npz", ".pyc", ".arrow"}
    for path in sorted(pkg.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        rel = path.relative_to(pkg).as_posix()
        if path.suffix.lower() in RESOURCE_SUFFIXES:
            fail(problems, f"resource file in a configuration-only package: {rel}")
        elif path.stat().st_size > 1_000_000:
            fail(problems, f"{rel} is {path.stat().st_size} bytes; this package holds "
                           f"configuration, not data")

    # ---- anchors: MEASURE is allowed pre-launch but must be flagged -----------------
    unmeasured = [k for k, v in contract["anchors"]["pristine_base"].items() if v == "MEASURE"]
    unmeasured += [k for k, v in contract["anchors"]["reference_improver"].items() if v == "MEASURE"]

    flags = []
    if unmeasured:
        flags.append("ANCHORS UNMEASURED")
    if assets_not_staged:
        flags.append("ASSETS NOT STAGED")
    status = "PACKAGE VALID" + (", " + ", ".join(flags) if flags else "")

    if problems:
        print("CONTRACT DRIFT DETECTED")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(status)
    if unmeasured:
        print(f"  anchors still MEASURE: {sorted(set(unmeasured))}")
    if assets_not_staged:
        print("  assets are declared by URL only; environment/fetch_assets.sh has not been run")
    if flags:
        print("  the package may be built and linted; no agent round may be scored yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
