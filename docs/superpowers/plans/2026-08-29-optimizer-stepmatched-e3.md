# E3-Only Step-Matched Optimizer Task Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and fully validate a new E3-only optimizer AutoResearch Harbor task with four Work nodes and one independent Judge node.

**Architecture:** Refine the immutable six-rung task into a collision-free sibling, retaining the existing data, optimizer, checkpoint, and Judge machinery while reducing every scientific collection to E3. Work trains E3 on 32 ranks; Judge consumes the durable checkpoint on 8 ranks through the existing RSI-Harness broker and torchrun shim.

**Tech Stack:** Harbor schema 1.4, RSI-Harness, Bash, Python 3, pytest 9.1.1, PyTorch distributed, Megatron-Bridge/Megatron-LM, lm-evaluation-harness, Docker/Apptainer, IBM Blue Vela LSF adapter.

**Spec:** `docs/superpowers/specs/2026-08-29-optimizer-stepmatched-e3-design.md`

## Global Constraints

- Leave `task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2` unchanged.
- Create only the absent sibling `task_collect/zf_tasks/pre-training/optimizer_stepmatched_e3`.
- Preserve standard RSI-Harness task semantics; task code must not invoke LSF, `blaunch`, Apptainer, SSH, or host discovery.
- Work declares exactly 32 GPUs and Judge exactly 8 GPUs.
- E3 remains 1,384,584,448 parameters, 10,559,946,752 tokens, 40,283 final/scoring updates, GBS64/MBS2, sequence length 4096, and seed 0.
- A baseline loss is scoreable only when certified on the same 32-rank training to 8-rank Judge-replay path.
- Every production behavior change follows a witnessed RED then GREEN test cycle.
- Temporary `priority`/non-exclusive scheduling is run-local; restore shared Blue Vela defaults immediately after submission.

---

### Task 1: Create the E3 contract tests

**Files:**
- Create: `task_collect/zf_tasks/pre-training/optimizer_stepmatched_e3/tests/test_task_contract.py`
- Create: `task_collect/zf_tasks/pre-training/optimizer_stepmatched_e3/tests/test_authority_contract.py`

**Interfaces:**
- Consumes: the six-rung source task as read-only evidence.
- Produces: executable assertions for E3-only scale acceptance, 32/8 GPU resources, one full and one reduced E3 attempt, and 32-to-8 Judge reload semantics.

- [ ] Write tests with literal expectations for `SCALE_ORDER == ("E3",)`, `ANCHORS == ("E3",)`, E3's exact training constants, Work `gpus = 32`, and Judge `gpus = 8`.
- [ ] Run the focused pytest files before the sibling implementation exists and record the expected missing/old-contract failure.
- [ ] Add behavioral fixtures that stage E3 successfully and reject any non-E3 scale without asserting on source-text spelling alone.
- [ ] Run the focused tests again and retain the RED output before production files are changed.

### Task 2: Generate the complete sibling and implement the E3-only Work contract

**Files:**
- Create: `task_collect/zf_tasks/pre-training/optimizer_stepmatched_e3/task.toml`
- Create: `task_collect/zf_tasks/pre-training/optimizer_stepmatched_e3/instruction.md`
- Create: `task_collect/zf_tasks/pre-training/optimizer_stepmatched_e3/README.md`
- Create/modify within the sibling: `environment/Dockerfile`, `environment/materialize_project.sh`, `environment/source.lock.yaml`, `environment/project-overlay/**`, and `environment/task-tools/**`

**Interfaces:**
- Consumes: structured research requests accepted by `optimizer_scaling_task.py` and staged data below `RSI_SHARED_DATA_ROOT`.
- Produces: one 32-rank E3 run per full/reduced attempt and closed, durable checkpoints under `/workspace/output/attempts/<id>`.

- [ ] Reuse reviewed source assets without copying `.pytest_cache` or `__pycache__`, reserving the absent destination with no overwrite.
- [ ] Reduce scale tables and shell profiles to E3 while retaining its literal model, optimizer, data, and checkpoint constants.
- [ ] Set `task.toml` Work GPUs to 32 and Judge GPUs to 8; retain separate Work/Judge resources and portable task paths.
- [ ] Change the candidate-ladder runner to schedule E3 attempts against the Work pool without embedding scheduler or node names.
- [ ] Run the Task 1 tests and confirm GREEN; run shell syntax and Python compile checks.

### Task 3: Implement E3-only staging, baseline certification, and Judge evaluation

**Files:**
- Create/modify within the sibling: `environment/task-tools/stage_stepmatched.py`
- Create/modify within the sibling: `tests/validate_submission.py`, `tests/validate_baseline_certificate.py`, `tests/evaluate.py`, `tests/test.sh`, `tests/emit_reward.py`, `tests/policy_check.py`, and `tests/reduced_source_path.py`
- Create: `task_collect/zf_tasks/pre-training/optimizer_stepmatched_e3/environment/task-data/optimizer_adamh_baseline_e3.yaml`

**Interfaces:**
- Consumes: one full E3 attempt, one reduced E3 attempt, and a certificate binding the 32-rank Work identity to an 8-rank Judge replay.
- Produces: six Judge broker requests and exactly one exclusive-created `/logs/verifier/reward.json` after complete success.

- [ ] Write RED fixtures for rejecting an old 32-GPU-only loss certificate and accepting a certificate with Work world size 32 and Judge world size 8.
- [ ] Reduce submission/ablation manifests to exactly E3 and preserve changed-source and checkpoint-integrity gates.
- [ ] Make Paloma, optimizer reload, and replay explicitly request the single eight-GPU Judge subpool while retaining E3's fixed GBS64 scoring definition.
- [ ] Compute the single-rung geometric reward from task-owned certified baseline metrics; retain zero only for recognized candidate-invalid outcomes and no reward for infrastructure failure.
- [ ] Run focused RED/GREEN tests, then the complete sibling pytest suite.

### Task 4: Complete documentation, source locks, and generation validation

**Files:**
- Modify within the sibling: all Markdown, TOML, YAML, Python, Bash, and Dockerfile text assets.
- Modify: `task_collect/zf_tasks/pre-training/optimizer_stepmatched_e3/environment/source.lock.yaml`

**Interfaces:**
- Consumes: final sibling bytes.
- Produces: a self-contained, checksum-locked Harbor task whose docs match runtime behavior.

- [ ] Add the Harbor canary in real comment syntax to every text file and update the source-lock inventory from final bytes.
- [ ] Run sibling pytest, Bash `-n`, Python compile, and source-lock validation.
- [ ] Run `.agents/skills/harbor-task-refiner/scripts/validate_task.py` once with `--harness-root RSI-Harness` and fix every error or evidence-disposition every warning.
- [ ] Compile/dry-run through RSI-Harness and verify resolved Work 32 / Judge 8 without an allocation.
- [ ] Rehash the original source inventory and prove it is unchanged.

### Task 5: Independent review and corrections

**Files:**
- Review: source task, design spec, implementation plan, and complete sibling.
- Modify: only sibling files for bounded, evidence-backed corrections.

**Interfaces:**
- Consumes: static/compiler results and recovered E3 contract.
- Produces: one initial independent decision, at most one targeted re-review, and an auditable final review status.

- [ ] Read the refiner independent-review instructions immediately before review.
- [ ] Ask the existing independent reviewer to trace Work, staging, Judge, failure, and reward paths read-only.
- [ ] Apply one consolidated correction round if required, rerun affected validation, and request the single targeted re-review.

### Task 6: Blue Vela baseline certification and live end-to-end run

**Files:**
- Runtime artifacts only under the RSI-Harness `/proj` run/image/cache roots.
- Temporarily modify then restore: `RSI-Harness/src/rsi_harness/cluster/bluevela/profile.toml`.

**Interfaces:**
- Consumes: the compiled sibling and prepared E3/Paloma/tokenizer assets.
- Produces: native RSI-Harness `RUN_INFO.json`, Work/Judge controller artifacts, certified baseline metrics, terminal reward, and cluster logs.

- [ ] Verify asset readiness, image materialization, resolved group `grp_models`, queue policy, and exact 4-Work/1-Judge allocation before submission.
- [ ] Temporarily select `priority` and non-exclusive hosts for this authorized run; submit with `rsi-harness run <task> --cluster bluevela` and immediately restore normal/exclusive shared defaults.
- [ ] Confirm LSF assigns four disjoint Work hosts and one Judge host, Work launches 32 ranks, and Judge launches 8 ranks.
- [ ] Remain attached through E3 full/reduced training, checkpoint persistence, 32-to-8 Paloma/reload/replay, and terminal reward.
- [ ] On each failure, preserve the immutable run, add a focused RED regression, make one root-cause fix, and retry with a new UTC run ID.

### Task 7: Final verification and handoff

**Files:**
- Verify: sibling, source task, RSI-Harness tests, and native live artifacts.

**Interfaces:**
- Consumes: final code and fresh execution evidence.
- Produces: completion report with exact paths, job/run IDs, tests, topology, reward, limitations, and source immutability proof.

- [ ] Run the full sibling suite and all affected RSI-Harness Blue Vela tests fresh.
- [ ] Run lint/compile/static validator fresh and read all exit codes.
- [ ] Verify source hashes against the initial 53-file inventory and confirm no source-task mutation.
- [ ] Report completion only when the live run has a terminal successful Harness state and finite reward.
