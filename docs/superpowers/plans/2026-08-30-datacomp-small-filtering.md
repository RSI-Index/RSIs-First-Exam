# DataComp-small Filtering RSI Task Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a new portable DataComp-small filtering Harbor task through one complete RSI-Harness Blue Vela Work/Judge run with a finite reward.

**Architecture:** A schema-1.4 task declares 16 Work GPUs and 16 independent Judge GPUs while leaving all LSF, node, rendezvous, and Apptainer control to RSI-Harness. The shared image contains immutable DataComp/OpenCLIP sources and portable preparation helpers; phase-scoped profile binds expose sealed external assets at `/rsi-data/datacomp-small`, and the trusted Judge retrains and evaluates the submitted UID set.

**Tech Stack:** Harbor task schema 1.4, RSI-Harness, Python 3.12, Bash, PyTorch distributed/`torchrun`, pinned DataComp and OpenCLIP, LSF/Apptainer supplied only by the Blue Vela adapter.

**Spec:** `docs/superpowers/specs/2026-08-30-datacomp-small-filtering.md`

## Global Constraints

- New task path is `examples/harbor-task-agent/datacomp-small-filtering` and task ID is `rsi/datacomp-small-filtering`.
- Every generated task text file carries one canary line with GUID `8f3ea414-02bb-47fd-82a8-30ae0372f43e`.
- Preserve immutable source revisions and the fixed ViT-B/32, 12.8M, batch-4096, LR-5e-4, warmup-500, seed-0, 40-row/38-score contract.
- Work and Judge each receive 16 GPUs; do not reduce resource counts unless the user explicitly changes the proposal.
- Task-owned code never references LSF, `blaunch`, hostnames, rendezvous addresses, Apptainer, scheduler policy, GPU model selectors, or Blue Vela filesystem paths.
- Data preparation precedes GPU submission and publishes an atomic, verified readiness manifest.
- Default live validation uses `normal` plus exclusive `-x`; priority plus nonexclusive is an evidence-triggered fallback only.
- The shared worktree contains unrelated user edits, so execution records reviewed diffs and does not create repository commits.

---

### Task 1: Executable task contract

**Files:**
- Create: `examples/harbor-task-agent/datacomp-small-filtering/tests/test_task_contract.py`
- Create: `examples/harbor-task-agent/datacomp-small-filtering/tests/fixtures/candidate-valid.json`

**Interfaces:**
- Consumes: the approved resource, provenance, portability, candidate, and reward contracts in the spec.
- Produces: `pytest` assertions that compile the real task, run candidate validation on literal fixtures, and reject forbidden cluster authority.

- [ ] **Step 1: Write the failing compiler/resource test**

  Build a test that invokes `HarborTaskCompiler().compile()` on the destination and asserts task ID, 16 Work GPUs, 16 Judge GPUs, 32 CPUs, 262,144 MiB memory, 716,800 MiB storage, offline phases, and required work/judge asset declarations.

- [ ] **Step 2: Verify the compiler/resource test fails for the absent task**

  Run `uv run --project RSI-Harness pytest examples/harbor-task-agent/datacomp-small-filtering/tests/test_task_contract.py -v` and confirm failure names the missing `task.toml`.

- [ ] **Step 3: Add behavioral portability and candidate tests**

  Execute the real candidate validator with a hand-authored valid UID fixture and malformed/duplicate/out-of-universe fixtures. Inspect compiled launch artifacts and assert that the task has no task-owned cluster authority.

- [ ] **Step 4: Verify the new cases fail because their real executables do not exist**

  Re-run the same command and confirm failures are caused by missing task files and candidate validator, not test syntax.

### Task 2: Harbor task shell and immutable Environment

**Files:**
- Create: `examples/harbor-task-agent/datacomp-small-filtering/task.toml`
- Create: `examples/harbor-task-agent/datacomp-small-filtering/instruction.md`
- Create: `examples/harbor-task-agent/datacomp-small-filtering/README.md`
- Create: `examples/harbor-task-agent/datacomp-small-filtering/policy.yaml`
- Create: `examples/harbor-task-agent/datacomp-small-filtering/proposal.md`
- Create: `examples/harbor-task-agent/datacomp-small-filtering/environment/Dockerfile`
- Create: `examples/harbor-task-agent/datacomp-small-filtering/environment/source-lock.yaml`

**Interfaces:**
- Consumes: exact revisions and resources from the spec.
- Produces: one compilable shared image at `/workspace`, pinned source trees at `/opt/datacomp` and `/opt/open_clip`, and a task contract that declares `/rsi-data/datacomp-small` assets independently for Work and Judge.

- [ ] **Step 1: Implement the minimum schema-1.4 task declaration**

  Set numeric Work/Judge GPU counts to 16, fixed per-node CPU/memory values, offline runtime policies, phase-scoped asset readiness entries, and no cluster deployment fields.

- [ ] **Step 2: Implement the immutable source image**

  Start from a digest-pinned CUDA/PyTorch base, fetch the two source repositories at their exact commits during the networked build, install exact dependencies, remove repository remotes/build caches, copy only portable task tools, and set `/workspace` as the final image workdir.

- [ ] **Step 3: Document candidate and operator behavior**

  State the exact UID/provenance schema, mutable paths, fixed Judge protocol, normal `rsi-harness run ... --cluster bluevela` command, data preparation command, and evidence-triggered scheduler fallback without embedding a site path.

- [ ] **Step 4: Compile and run the contract tests**

  Run the Task 1 pytest command and `uv run --project RSI-Harness rsi-harness run examples/harbor-task-agent/datacomp-small-filtering --cluster bluevela --dry-run --agent codex --agent-auth local --model gpt-5.6-sol --reasoning-effort xhigh --json`; preserve the dry-run plan as validation evidence.

### Task 3: Portable asset preparation and baseline materialization

**Files:**
- Create: `examples/harbor-task-agent/datacomp-small-filtering/environment/task-tools/asset_contract.py`
- Create: `examples/harbor-task-agent/datacomp-small-filtering/environment/task-tools/prepare_assets.py`
- Create: `examples/harbor-task-agent/datacomp-small-filtering/environment/task-tools/materialize_baseline.py`
- Create: `examples/harbor-task-agent/datacomp-small-filtering/environment/task-tools/validate_candidate.py`
- Create: `examples/harbor-task-agent/datacomp-small-filtering/solution/solve.sh`

**Interfaces:**
- Consumes: `--root PATH`, pinned source data, 26 CommonPool metadata files, successful crawl outputs, evaluation assets, and official L/14 scores.
- Produces: atomically published `READY.json`; sealed metadata/crawl/eval/reference manifests; `/workspace/research/submission/selected_uids.npy`; `/workspace/research/submission/provenance.json`; exit code 0 for a valid candidate and 2 for a candidate-contract failure.

- [ ] **Step 1: Extend failing tests around asset and candidate boundaries**

  Use a temporary literal mini-pool to require deterministic successful-UID intersection, exact top-30% selection, duplicate rejection, foreign-UID rejection, cardinality/byte limits, digest verification, and no readiness marker after an interrupted preparation.

- [ ] **Step 2: Verify each boundary test fails before implementation**

  Run each named pytest node and confirm the missing function or wrong observable result is the reason.

- [ ] **Step 3: Implement preparation with atomic publication**

  Accept only explicit logical roots, verify the pinned metadata revision and exact expected inventory, stage into a sibling temporary directory, compute SHA-256 manifests, derive the build-realized UID universe and official top-30% intersection, fsync outputs, and rename `READY.json` last.

- [ ] **Step 4: Implement baseline and candidate tools**

  Make `solve.sh` call the baseline materializer idempotently without training, evaluation, tests access, reward writing, or submission. Make candidate validation stream UID data, reject malformed/duplicate/out-of-universe selections, and write a trusted immutable snapshot for Judge.

- [ ] **Step 5: Run asset/candidate tests and the full static suite**

  Run the task contract pytest, ShellCheck on shell files, Python compile checks, canary verification, and forbidden-token structural scans.

### Task 4: Trusted multi-node Judge

**Files:**
- Create: `examples/harbor-task-agent/datacomp-small-filtering/tests/test.sh`
- Create: `examples/harbor-task-agent/datacomp-small-filtering/tests/run_training.py`
- Create: `examples/harbor-task-agent/datacomp-small-filtering/tests/run_evaluation.py`
- Create: `examples/harbor-task-agent/datacomp-small-filtering/tests/reward.py`

**Interfaces:**
- Consumes: the Judge snapshot, sealed data assets, immutable `/opt/datacomp` and `/opt/open_clip`, and Harness-provided distributed environment.
- Produces: trusted checkpoint/evaluation scratch under `/logs/verifier`, exactly 40 complete task rows, exactly 38 finite aggregate inputs, and one atomic `{"reward": <finite Average>}` file only on complete success; candidate invalid produces exactly `{"reward": 0.0}`.

- [ ] **Step 1: Add failing Judge semantic tests**

  Require candidate-invalid versus infrastructure-failure separation, exact 40/38 completeness, rejection of NaN/Inf/duplicate/missing metrics, correct hand-calculated Average, atomic reward publication, and no use of candidate-authored scalar metrics.

- [ ] **Step 2: Verify the semantic tests fail for missing Judge code**

  Run the named pytest cases and confirm missing executables/functions cause the failures.

- [ ] **Step 3: Implement fixed distributed training**

  `test.sh` validates assets and candidate on the lead process, then invokes exactly `torchrun --nnodes 2 --nproc-per-node 8 /tests/run_training.py`. The entrypoint validates `WORLD_SIZE=16`, calls the pinned upstream training recipe with the sealed candidate, synchronizes completion, and publishes only trusted checkpoint inventory.

- [ ] **Step 4: Implement trusted evaluation and reward**

  Run the official evaluator against the fresh checkpoint, parse a closed metric-name set, require 40 result rows and 38 finite main metrics, recompute their arithmetic mean, and atomically publish reward. Bound Agent-visible output to safe aggregate status.

- [ ] **Step 5: Run the full task test suite**

  Run contract/unit tests, shell/Python checks, candidate-invalid smoke, and a controlled miniature Judge fixture that uses real scripts with a tiny local dataset while bypassing only the costly upstream train/eval subprocess through explicit test-only commands.

### Task 5: Static compiler and independent review gate

**Files:**
- Modify only files rejected by evidence in `examples/harbor-task-agent/datacomp-small-filtering/**`.

**Interfaces:**
- Consumes: complete task tree and clean automated checks.
- Produces: compiler success, a clean full static suite, an independent proposal-first review, and a separate adversarial review with no unresolved blocker.

- [ ] **Step 1: Run authoritative compilation and repository regression checks**

  Compile through RSI-Harness, run relevant Harness tests, audit all generated task text for the canary, and inspect `git diff --check` plus task-scoped diffs.

- [ ] **Step 2: Run proposal-first independent review**

  Give a reviewer the approved spec and destination without implementation hints. Require findings with file/line evidence for science, candidate semantics, trust, resources, portability, and migration.

- [ ] **Step 3: Run adversarial integration review**

  Give a distinct reviewer the accepted contract and task tree. Require pressure tests for fail-closed behavior, hidden topology assumptions, Work/Judge separation, incomplete assets, reward spoofing, and operational runability.

- [ ] **Step 4: Repair findings test-first and repeat both gates**

  For each confirmed defect, add a failing behavioral regression, implement the smallest repair, re-run all static checks, and request reviewer confirmation.

### Task 6: Data preparation and Blue Vela end-to-end validation

**Files:**
- Modify: `RSI-Harness/src/rsi_harness/cluster/bluevela/profile.toml` only to add phase-scoped read-only mappings for the prepared task data root.
- Create at runtime outside git: site-owned sealed DataComp-small data, content-addressed SIF cache, unique Harness run/log roots, Work artifacts, Judge artifacts, and reward.

**Interfaces:**
- Consumes: a statically accepted task, writable operator preparation mount, cluster credentials, and default Blue Vela profile.
- Produces: prepared `READY.json`, a resolved four-node dry-run plan, one fresh run identity, successful Agent/Work, `rsi-submit`, independent 16-GPU Judge, finite reward, and clean teardown.

- [ ] **Step 1: Read the execution-validation contract and perform read-only preflight**

  Validate profile resolution, queue/group access, auth file mode without reading secrets, free space, Apptainer/backend binaries, asset status, and exact four-node resource derivation before submitting any GPU job.

- [ ] **Step 2: Prepare and seal assets before GPU allocation**

  Run the task image's portable preparation command in a CPU allocation with a writable site root. Resume only idempotent downloads; reject partial sealed trees; verify `READY.json`, manifests, minimum counts, and baseline provenance before making normal binds read-only.

- [ ] **Step 3: Execute the ordinary full-scale Harness command**

  Run `uv run --project RSI-Harness rsi-harness run /u/yuetai/more_task/RSI-Index-Public/examples/harbor-task-agent/datacomp-small-filtering --cluster bluevela --agent codex --agent-auth local --model gpt-5.6-sol --reasoning-effort xhigh --max-submissions 1 --verbose` with default normal/exclusive scheduling.

- [ ] **Step 4: Monitor one coherent lifecycle**

  Follow the same process session and unique run tree through image build/cache, four-node allocation, Work launch, `rsi-submit`, Judge launch, 16-rank training, 40-task evaluation, reward publication, and teardown. Never stitch evidence across runs.

- [ ] **Step 5: Debug failures with fresh full-scale runs**

  Classify each failure as task defect, data defect, environment defect, or scheduler evidence. Add a failing regression before code repairs. Use priority/nonexclusive only after repeated pending evidence and keep all 16+16 GPUs. After any code or image change, start a fresh run identity.

- [ ] **Step 6: Verify final evidence**

  Confirm a single run contains Work success, submission record, independent Judge identity, complete trusted metrics, finite reward, durable native Harness logs, and no orphaned LSF job. Record exact run/log paths and final task diff without claiming success from partial stages.
