---
name: bluevela-harbor-task-agent
description: Use when an approved proposal must become a new RSI-Harness Harbor task and complete Work/Judge validation on IBM Blue Vela.
---

# Blue Vela Multi-Node Harbor Task Agent

Create one portable RSI-Harness schema-1.4 task. The task declares logical
Work/Judge resources and phase-visible assets; `rsi-harness run ... --cluster
bluevela` owns LSF, node discovery, Work/Judge pool partitioning, Apptainer,
site paths, and runtime placement.

This skill has one workflow and no mode switch: consume one approved proposal,
generate one absent task, validate it statically, then run and repair it through
real Work, native submission, independent Judge, and a finite reward. Do not
stop merely because generation, compilation, image build, or one failed attempt
finished.

## Required references

Read each reference completely immediately before its stage:

1. [references/repository-research.md](references/repository-research.md) before
   tracing source, data, baseline, and Harness compatibility evidence;
2. [references/cluster-contract.md](references/cluster-contract.md) before
   freezing resources and portability boundaries;
3. [references/environment-design.md](references/environment-design.md) before
   designing the image, data preparation, assets, and workspace state;
4. [references/verifier-design.md](references/verifier-design.md) before
   designing submission and Judge behavior;
5. [references/task-template.md](references/task-template.md) and
   [references/generation-validation.md](references/generation-validation.md)
   before confirmation and generation; and
6. [references/independent-review.md](references/independent-review.md) before
   the bounded independent review; and
7. [references/execution-validation.md](references/execution-validation.md)
   after static review passes and before any live action.

The references and bundled validator are the contract. Do not import the
retired normal-Harbor task-owned `cluster/bluevela.toml` architecture.

## Non-negotiable architecture

- Generate a normal `rsi/<slug>` schema-1.4 RSI-Harness task. Work GPUs are
  `[environment].gpus`; Judge GPUs are
  `[metadata.rsi_harness.verifier].gpus`.
- Do not generate `cluster/`, `baselines/`, an Apptainer definition, LSF
  scripts, host discovery, or a Blue Vela config inside the task. A baseline
  directory is allowed only when the scientific contract names a real
  task-owned portable artifact; it is never structural boilerplate.
- Task runtime code must not invoke or inspect `bsub`, `bjobs`, `bkill`,
  `blaunch`, `LSB_MCPU_HOSTS`, hostnames, Apptainer, queue/group policy, GPU
  model selectors, or site paths. It must not hard-code node counts derived
  from a particular cluster profile.
- The selected cluster profile is the sole authority for GPUs per node,
  queue/group, exclusivity, site binds, image/run/log roots, host exclusions,
  builder resources, and physical placement. Migrating sites means selecting a
  profile, not editing task code.
- Preserve the Harness single-node branch. Multi-node metadata must compile
  without changing behavior for tasks whose Work and Judge both fit one node.
- Do not treat a compiler pass, SIF build, Work start, checkpoint save, or
  shortened run as end-to-end evidence. Generation proceeds directly into
  validation.
- The approved proposal and request to execute this workflow authorize task
  generation and the exact resource policy below. A materially incomplete
  proposal is a blocker; do not invent science or resources.
- Default scheduling is the canonical `normal` queue with exclusive placement
  (`-x`). After recorded evidence that this allocation remains unavailable,
  use a run-private `priority` plus nonexclusive profile without changing the
  shared profile. A complete fallback run is final evidence and must not be
  repeated on `normal + -x`.
- Both scheduler policies run the exact Work/Judge GPU totals in `task.toml`.
  Reduce those totals only after an explicit user instruction naming the
  reduced scope; never infer a smaller validation from queue pressure.
- Never overwrite an unrelated user change.

## Workflow

### 1. Preflight the approved proposal

Require one approved proposal with immutable source, research question,
candidate deliverable, baseline, fixed evaluation and finite reward, editable
and prohibited scope, data/network policy, and evidence-based Work/Judge
budgets. Return a materially incomplete proposal rather than inventing values.

### 2. Establish repository and Harness evidence

Follow repository research. Resolve immutable source/model/data/baseline refs.
Inspect the project-local RSI-Harness compiler, Blue Vela resource planner,
multi-node fixture/tests, packaged profile interface, and a complete working
multi-node task when one exists. Compatibility evidence establishes interfaces,
not scientific values for the new task.

Record separately:

- contributor-confirmed science;
- immutable repository/data evidence;
- task declarations;
- profile-owned deployment values; and
- live facts still pending.

### 3. Resolve the portable resource contract

Follow the cluster contract. Freeze Work/Judge GPU totals, per-node CPU and
memory requests, shared workspace storage, timeouts, assets, and required
checkpoint/handoff semantics. Validate node arithmetic against the selected
profile during compiler/preflight validation; never copy its GPUs-per-node into
task logic.

`agent.timeout_sec` covers all Work activity, blocking Judge submissions, and
margin. `verifier.timeout_sec` covers one complete Judge execution including
reload/reshard and evaluation. The default scheduler policy is `normal + -x`;
`priority` without `-x` is the allocation fallback defined above, not a smaller
task. Either policy can supply final evidence when the complete declared
workflow and finite reward succeed.

### 4. Design Environment and data readiness

Follow environment design. Produce a Dockerfile-based Base/Work Environment.
Keep large site data out of the image and task; declare logical container paths
under `metadata.rsi_harness.assets`, with separate Work/Judge visibility.
Provide a portable preparation command only when raw data must be materialized
before normal runs. The selected profile maps site-owned sources to those
logical paths.

Freeze the complete post-build starting state, remove accidental bytecode,
caches, partial data, and build products, and keep candidate-owned paths inside
the snapshotted WORKDIR. All selected output/checkpoint state must be flushed to
disk before submission.

### 5. Design the Work-to-Judge contract

Follow verifier design. `tests/test.sh` remains the only scoring entry point;
`solution/solve.sh` only restores the approved baseline candidate. Work invokes
ordinary `rsi-submit`; it never selects Judge hosts or starts Judge itself.

When Judge consumes distributed checkpoints, define exact retained updates,
complete shard inventory, source/config lineage, optimizer-state requirements,
and rank-count-independent reload/reshard behavior. If reward depends on an
external live baseline authority, define a separate rewardless certification
bootstrap and a normal certified-baseline run. Do not impose this branch on
tasks whose baseline can be independently evaluated in one Judge run.

### 6. Freeze the approved contract

Record one compact execution contract derived from the approved proposal. It
includes:

- task identity, immutable evidence, research question, candidate, baseline,
  metric/reward, invalid-result behavior, and visible feedback;
- exact standard task tree and absent destination;
- Work/Judge GPU totals, CPU/memory/storage semantics, timeout arithmetic, and
  expected single-node or multi-node resolution under the selected profile;
- logical Work/Judge assets and any one-time data preparation;
- snapshot, checkpoint, submission, Judge reload/reshard, and optional baseline
  authority contract;
- network/provider requirements and hidden-state boundaries; and
- static checks, live Blue Vela gates, the default scheduler policy, and the
  allocation fallback policy.

Do not introduce a second mode or a second approval pause between generation
and validation. Pause only when the proposal is incomplete or later work needs
authority outside the approved contract.

### 7. Generate the package

Reserve one absent destination and follow the task template. Generate only
task-owned RSI-Harness files. All task text carries the Harbor canary. Use
logical container paths and evidence-backed task values; no site path, host,
group, queue, GPU model, or profile policy enters the package.

### 8. Static validation

Follow generation validation. Run the bundled validator once with the
project-local Harness checkout. Resolve every error and every warning before
review. This stage compiles and plans only; it performs no build, allocation,
Work, submission, Judge, or reward execution.

### 9. Bounded review

Use one initial read-only review, one consolidated correction, and one targeted
re-review. The generator owns edits. Finish as `PASS` only when static/compiler
validation and re-review are clean.

Continue directly after review; `PASS` is an internal gate, not a terminal
handoff.

### 10. Continue through live validation

Follow execution validation immediately after static review. Run the bundled
read-only preflight and continue through its fixed gates using the task's exact
resources. Start with `normal + -x`. When repeated scheduler evidence shows
that allocation remains unavailable, create a run-private `priority` profile
without `-x`, preflight it with that evidence, and continue the same workflow.

Repeated evidence means the same full-size normal job remains pending in at
least two timestamped scheduler observations separated by the declared wait
window, or terminates with an allocation-specific scheduler reason. Cancel
only that exact pending normal job and confirm it is no longer active before
submitting the fallback, so both policies cannot allocate concurrently.

Preserve every failed run, diagnose its exact boundary, make the smallest
source-controlled in-contract repair, and use a fresh run ID for each retry.
Modify source and add a regression only for a demonstrated source defect;
transient scheduler/node/storage faults get a fresh allocation without a fake
code change. A final retry starts from the frozen task state and never consumes
checkpoint/output state from an earlier run.
One successful full fallback run completes validation; do not schedule a
duplicate normal run. Final status is only:

- `END_TO_END_VALIDATED` after complete Work, native submission, independent
  Judge, and a finite reward; or
- `BLOCKED` when progress requires missing credentials/data/quota, different or
  larger resources, destructive external action, or a scientific decision
  outside the approved proposal.

## Red flags

Stop and correct the design if any of these appear:

- “The old `cluster/bluevela.toml` example is already proven.”
- “Put `blaunch` in the task so it controls placement.”
- “Blue Vela has eight GPUs per node, so write four nodes into the launcher.”
- “The SIF built, therefore multi-node is ready.”
- “Work trained, therefore Judge will work.”
- “The task was generated, so stop before live validation.”
- “The normal queue is pending, so silently shrink Work or Judge.”
- “A complete priority/no-`-x` success must be repeated on normal.”
- “One failed attempt is a terminal `FAIL`.”
- “A shortened run proves the unshortened task.”
- “Resume or combine interrupted attempts to create baseline authority.”

Each statement crosses an ownership or evidence boundary. Return to the
portable task contract and leave deployment behavior to Harness plus the
selected profile.
