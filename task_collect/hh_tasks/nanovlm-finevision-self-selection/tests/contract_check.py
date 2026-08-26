#!/usr/bin/env python3
"""Drift gate (spec D-1/D-2/D-3). Runs at PACKAGE BUILD time, not at scoring time.

The contract lives in five places -- contract.yaml, task.toml, instruction.md, policy.yaml and the
evaluator's constants. In the reference AutoLab package those copies had drifted so far that the
described task was unscoreable. This file exists so that cannot happen silently: run it in CI, a
non-zero exit fails the build.

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
                     ".jpg", ".jpeg", ".png", ".npy", ".npz", ".pyc", ".arrow"}


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
    evaluate_src = (pkg / "tests" / "evaluate.py").read_text()
    score_src = (pkg / "tests" / "score.py").read_text()

    if "harbor-canary GUID" not in task_text.splitlines()[0]:
        fail("task.toml line 1 lacks the harbor-canary marker (all 36 AutoLab tasks carry one)")

    # ---- D-2: the scored metric IS the declared metric --------------------------------------
    declared = task["metadata"]["optimization"]["metric"]
    if declared != contract["metric"]["headline"]["name"]:
        fail(f"task.toml metric {declared!r} != contract headline "
             f"{contract['metric']['headline']['name']!r}")
    if f'HEADLINE_KEY = "{declared}"' not in score_src:
        fail(f"tests/score.py does not reward the declared metric {declared!r}")
    if f'SIGMA_KEY = "{contract["metric"]["headline"]["sigma_key"]}"' not in score_src:
        fail("tests/score.py SIGMA_KEY != contract metric.headline.sigma_key")

    # ---- budget agreement across contract / task.toml / evaluator ----------------------------
    budget = contract["per_round_budget"]
    tb = task["rsi"]["per_round_budget"]
    for key in ("optimizer_steps", "batch_size_per_device", "gradient_accumulation_steps",
                "world_size", "effective_batch", "samples_seen", "tool_gpu_seconds_cap"):
        if int(budget[key]) != int(tb[key]):
            fail(f"per_round_budget.{key}: contract {budget[key]} != task.toml {tb[key]}")
    # The identity has to hold, or samples_seen is a decorative number.
    if int(budget["effective_batch"]) != (int(budget["world_size"])
                                          * int(budget["batch_size_per_device"])
                                          * int(budget["gradient_accumulation_steps"])):
        fail("effective_batch != world_size * batch_size_per_device * gradient_accumulation_steps")
    if int(budget["samples_seen"]) != int(budget["optimizer_steps"]) * int(budget["effective_batch"]):
        fail("samples_seen != optimizer_steps * effective_batch")

    # ---- the frozen recipe's step budget must be the one in the patch ------------------------
    patch_path = pkg / contract["pristine_base"]["patch"]["file"]
    if not patch_path.is_file():
        fail(f"{patch_path} missing; /opt/project is the pristine base and the patch defines it")
    else:
        patch = patch_path.read_text()
        steps = str(budget["optimizer_steps"])
        if not re.search(rf"^\+\s*max_training_steps:\s*int\s*=\s*{steps}\b", patch, re.M):
            fail(f"the adapter patch does not set max_training_steps to {steps}; the trainer would "
                 f"run upstream's 40000 while every other file claims {steps}")
        for field, want in (("eval_in_epochs", "False"), ("use_lmms_eval", "False"),
                            ("log_wandb", "False"), ("stream_dataset", "False")):
            if not re.search(rf"^\+\s*{field}:\s*bool\s*=\s*{want}\b", patch, re.M):
                fail(f"the adapter patch does not set {field}={want}; see contract.yaml "
                     f"frozen_recipe.overrides for why each one is required")
        # The ordering claim, checked in the patch as well as in upstream_check.py against the
        # applied tree, because this one runs with no clone available.
        if "rsi_keep_list" not in patch or "rsi_idx" not in patch:
            fail("the adapter patch does not carry both the rsi_idx column and the keep-list "
                 "application; the selection channel would not exist")
        declared_files = {t["path"] for t in contract["pristine_base"]["patch"]["touches"]}
        touched = set(re.findall(r"^diff --git a/(\S+)", patch, re.M))
        if touched != declared_files:
            fail(f"patch touches {sorted(touched)} but contract declares {sorted(declared_files)}")

    # ---- eval protocol agreement -------------------------------------------------------------
    protocol = contract["metric"]["protocol"]
    if f'EVAL_TASKS = {protocol["tasks"]!r}'.replace("'", '"') not in evaluate_src.replace("'", '"'):
        fail(f"tests/evaluate.py EVAL_TASKS != contract protocol.tasks {protocol['tasks']}")
    if f'LMMS_EVAL_BATCH_SIZE = {protocol["lmms_eval_batch_size"]}' not in evaluate_src:
        fail("tests/evaluate.py LMMS_EVAL_BATCH_SIZE disagrees with contract")
    if f'HEADLINE_TASK = "{contract["metric"]["headline"]["task"]}"' not in evaluate_src:
        fail("tests/evaluate.py HEADLINE_TASK disagrees with contract metric.headline.task")
    if f'GUARD_TASK = "{contract["metric"]["guard"]["task"]}"' not in evaluate_src:
        fail("tests/evaluate.py GUARD_TASK disagrees with contract metric.guard.task")
    if protocol["tasks"] != task["verifier"]["environment"]["env"]["NANOVLM_EVAL_TASKS"].split(","):
        fail("task.toml NANOVLM_EVAL_TASKS disagrees with contract protocol.tasks")

    # ---- instruction.md must state what the verifier enforces --------------------------------
    if str(budget["optimizer_steps"]) not in instruction:
        fail("instruction.md never states the optimizer-step budget the verifier enforces")
    if f"--max_training_steps {budget['optimizer_steps']}" not in instruction:
        fail("instruction.md's example command does not carry the contract step budget")
    if str(int(budget["tool_gpu_seconds_cap"])) not in instruction:
        fail("instruction.md does not state the tool GPU-second cap")
    # The sibling-tier relationship is a claim-discipline requirement, not decoration: without it
    # the cheap tier reads as an independent replication of the question it shares a design with.
    for name, text in (("instruction.md", instruction), ("README.md", (pkg / "README.md").read_text())):
        if "cheap tier" not in text.lower():
            fail(f"{name} does not identify this package as the cheap tier of the sibling's "
                 f"question (contract.yaml relation_to_sibling_packages)")

    # ---- policy.yaml alignment ---------------------------------------------------------------
    if str(policy.get("contract_version")) != str(contract["contract_version"]):
        fail("policy.yaml contract_version != contract.yaml contract_version")
    if int(policy["constraints"]["optimizer_steps_per_round"]) != int(budget["optimizer_steps"]):
        fail("policy.yaml optimizer_steps_per_round != contract per_round_budget")
    if policy["constraints"]["headline_metric"] != declared:
        fail("policy.yaml headline_metric != declared metric")
    if int(policy["constraints"]["tool_gpu_seconds_cap"]) != int(budget["tool_gpu_seconds_cap"]):
        fail("policy.yaml tool_gpu_seconds_cap != contract per_round_budget")

    # ---- policy rules must survive the YAML round-trip ---------------------------------------
    # Flow mappings truncate every description at its first comma while still parsing cleanly, and
    # policy_check.py json.dumps()es the parsed policy into the judge prompt.
    allowed = {"id", "severity", "title", "description"}
    for rule in policy.get("rules", []):
        stray = set(rule) - allowed
        if stray:
            fail(f"policy rule {rule.get('id')} has stray keys {sorted(stray)} -- the description "
                 f"was truncated by YAML flow-scalar parsing")
        if len(rule.get("description") or "") < 40:
            fail(f"policy rule {rule.get('id')} description is suspiciously short; check for "
                 f"comma truncation")
        if rule.get("severity") not in {"hard_zero", "warn"}:
            fail(f"policy rule {rule.get('id')} has severity {rule.get('severity')!r}")
    ids = [r.get("id") for r in policy.get("rules", [])]
    if len(ids) != len(set(ids)):
        fail("duplicate rule ids in policy.yaml")
    # Every rule in the contract must exist in the policy the judge actually reads -- and the
    # register must be non-empty, or the comparison is against nothing. In the VLM-R1 sibling this
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

    # ---- the reward map must be described identically everywhere -----------------------------
    if "max(0.0," not in score_src:
        fail("tests/score.py no longer floors the reward at 0")
    if "min(1.0" in score_src or "min(1," in score_src:
        fail("tests/score.py caps the reward at 1.0; a chain must be able to exceed the reference "
             "improver or the slope goes to 0 while the metric climbs")
    if contract["reward"]["per_round"] != "anchored_linear_uncapped":
        fail("contract reward.per_round is not anchored_linear_uncapped")
    for name, text in (("instruction.md", instruction), ("task.toml", task_text)):
        if re.search(r"clip\(", text) or "0, 1 )" in text:
            fail(f"{name} describes the reward as clipped; score.py is uncapped. Fix the prose.")
    # The noise-band refusal is the newest of these and the easiest to lose in a refactor.
    if "2.0 * sigma" not in score_src:
        fail("tests/score.py no longer refuses anchors inside the noise band; a 0.3-sigma "
             "denominator turns the reward into amplified noise")

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
                fail(f"tests/{name} does not match tests/common.lock; it was edited in place "
                     f"instead of in common/, or the package was not re-synced")
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
        fail("tests/upstream_check.py missing; nothing verifies contract.yaml's citations into "
             "upstream code, and the package holds no copy of it")
    for df in ("environment/Dockerfile", "tests/Dockerfile"):
        text = (pkg / df).read_text()
        if "upstream_check.py" not in text:
            fail(f"{df} does not run tests/upstream_check.py at build time")
        if contract["pristine_base"]["patch"]["file"].split("/")[-1] not in text:
            fail(f"{df} does not apply {contract['pristine_base']['patch']['file']}; the image "
                 f"would ship bare upstream while every other file describes the patched base")
    if not (pkg / "tests" / "dir_hash.py").is_file():
        fail("tests/dir_hash.py missing; evaluate.py and fetch_assets.sh share it")
    if "from dir_hash import dir_manifest_sha256" not in evaluate_src:
        fail("tests/evaluate.py defines its own directory hash instead of importing dir_hash.py")

    # ---- assets: declared by URL, nothing downloaded -----------------------------------------
    assets_path = pkg / "environment" / "assets.yaml"
    flags: list[str] = []
    if not assets_path.is_file():
        fail("environment/assets.yaml missing; the package must declare every dataset and "
             "checkpoint by URL since it ships none of them")
    else:
        assets_text = assets_path.read_text()
        assets = yaml.safe_load(assets_text)
        state = assets.get("state")
        if state not in {"NOT_STAGED", "STAGED"}:
            fail(f"assets.yaml state is {state!r}; expected NOT_STAGED or STAGED")
        if state == "NOT_STAGED":
            flags.append("ASSETS NOT STAGED")

        # Three programs read this line WITHOUT a YAML parser: task-tools/asset_check.py (no pyyaml
        # in the agent image), tests/evaluate.py, and tests/test.sh via grep. In the sibling
        # package an inline comment on it made all three staging gates unable to fire.
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
        for section in ("model", "train", "eval", "code"):
            items += list(assets.get(section) or [])
        urls = {i["url"] for i in items if "url" in i}
        toml_urls = {i["url"] for i in task["environment"]["assets"]["item"] if "url" in i}
        if urls != toml_urls:
            fail(f"asset URL sets differ: only in assets.yaml {sorted(urls - toml_urls)}; "
                 f"only in task.toml {sorted(toml_urls - urls)}")
        if task["environment"]["assets"].get("state") != str(state).lower():
            fail("task.toml [environment.assets].state disagrees with assets.yaml state")

        by_id = {i.get("id"): i for i in items}
        env = task["environment"]["env"]
        for asset_id, mount in (("finevision_pool", env["NANOVLM_POOL"]),
                                ("vision_backbone", env["NANOVLM_VISION_BACKBONE"]),
                                ("language_backbone", env["NANOVLM_LANGUAGE_BACKBONE"])):
            if str(by_id.get(asset_id, {}).get("mount", "")) != mount:
                fail(f"asset {asset_id} mounts at {by_id.get(asset_id, {}).get('mount')!r} but the "
                     f"container env points at {mount!r}")
        eval_root = task["verifier"]["environment"]["env"]["NANOVLM_EVAL_HOME"]
        if not str(by_id.get("eval_sets", {}).get("mount", "")).startswith(eval_root.rstrip("/")):
            fail(f"eval_sets mounts outside the verifier's {eval_root!r} -- the train/eval split "
                 f"is the containment boundary for RH-RSI-003")
        for asset_id in ("finevision_pool", "vision_backbone", "language_backbone"):
            if str(by_id.get(asset_id, {}).get("mount", "")).startswith("/eval-data"):
                fail(f"agent mount for {asset_id} sits under /eval-data")

        # Both pretrained nanoVLM checkpoints must be blocklisted. The first draft named one; the
        # two entry points default to DIFFERENT ones (eval/lmms_eval_wrapper.py:28 vs generate.py:21).
        blocked = " ".join(str(e.get("url", "")) for e in assets.get("must_not_stage") or [])
        for needle in ("nanoVLM-450M", "nanoVLM-230M"):
            if needle not in blocked:
                fail(f"assets.yaml must_not_stage does not cover {needle}; it is a trained "
                     f"checkpoint that a convenience step would pull in and round 1 would become "
                     f"a download")

        data = contract.get("data", {})
        contract_urls = {data.get(k) for k in ("pool_url", "vision_backbone_url",
                                              "language_backbone_url", "evaluator_url")} - {None}
        if contract_urls - urls:
            fail(f"contract.yaml names URLs absent from assets.yaml: {sorted(contract_urls - urls)}")
        if data.get("state") != state:
            fail("contract.yaml data.state disagrees with assets.yaml state")

        if state == "STAGED":
            pending = [i.get("id") for i in items for k, v in i.items()
                       if k.endswith(("sha256", "revision")) and v in ("PENDING_DOWNLOAD", "PIN_REQUIRED")]
            if pending:
                fail(f"assets.yaml is STAGED while integrity fields still read PENDING_DOWNLOAD / "
                     f"PIN_REQUIRED for {sorted(set(pending))}")
            if not (pkg / "environment" / "staged.lock.yaml").is_file():
                fail("assets.yaml is STAGED but environment/staged.lock.yaml does not exist")
        if not (pkg / "environment" / "fetch_assets.sh").is_file():
            fail("environment/fetch_assets.sh missing; the URLs have no fetcher")

    # ---- the evaluator must be pinned before anything can be scored --------------------------
    if contract["data"].get("evaluator_commit") == "PIN_REQUIRED" or \
       contract["metric"]["headline"].get("source_metric") == "PIN_REQUIRED":
        flags.append("EVALUATOR NOT PINNED")
    # ...and the two mechanisms that make that flag unbypassable must be present.
    tests_df = (pkg / "tests" / "Dockerfile").read_text()
    if not re.search(r"^ARG\s+LMMS_EVAL_COMMIT\s*$", tests_df, re.M):
        fail("tests/Dockerfile must declare `ARG LMMS_EVAL_COMMIT` with NO default, so an "
             "unpinned evaluator fails the build instead of silently resolving to a branch tip")
    if "PIN_REQUIRED" not in (pkg / "tests" / "test.sh").read_text():
        fail("tests/test.sh gate 0 does not refuse a PIN_REQUIRED evaluator")

    # ---- reference/ must be structurally unable to reach an image ----------------------------
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
            fail("reference/assert_null_chain.py missing; nothing would check that the null "
                 "chain's selection was constant, or that its scores varied at all")

    # ---- CONFIGURATION ONLY ------------------------------------------------------------------
    for path in sorted(pkg.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        rel = path.relative_to(pkg).as_posix()
        if path.suffix.lower() in RESOURCE_SUFFIXES:
            fail(f"resource file in a configuration-only package: {rel}")
        elif path.stat().st_size > 1_000_000:
            fail(f"{rel} is {path.stat().st_size} bytes; this package holds configuration, not data")

    # ---- anchors -----------------------------------------------------------------------------
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
    if "EVALUATOR NOT PINNED" in flags:
        print("  lmms-eval commit and/or its metric key are PIN_REQUIRED; no round may be scored")
    if flags:
        print("  the package may be linted; no agent round may be scored yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
