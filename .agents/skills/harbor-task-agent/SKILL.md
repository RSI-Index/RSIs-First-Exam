---
name: harbor-task-agent
description: Use when an approved RSI-Index AutoResearch proposal must become a new self-contained RSI-Harness-compatible Harbor task.
---

# Harbor Task Agent

Turn one approved proposal into a complete RSI-Harness Harbor task. Copying the
task directory into an RSI-Harness workspace must be enough to build and run it,
apart from RSI-Harness, declared resources, credentials, and operator inputs
documented by the task. RSI-Harness is authoritative; retain Terminal-Bench
conventions only where this skill explicitly does so.

This skill does not elicit or approve proposals, refine existing tasks, or run
stateful execution validation. Route a draft proposal back to `proposal-agent`.

## Load references by stage

Read each selected reference completely immediately before its stage:

1. [references/repository-research.md](references/repository-research.md) before
   inspecting the proposal's official repository;
2. [references/environment-design.md](references/environment-design.md) before
   freezing Base/Work and the Environment-to-Verifier interface;
3. [references/verifier-design.md](references/verifier-design.md) after that
   interface is frozen and before writing evaluation;
4. [references/task-template.md](references/task-template.md) and
   [references/generation-validation.md](references/generation-validation.md)
   before the consolidated contributor confirmation; and
5. [references/independent-review.md](references/independent-review.md)
   immediately before independent review.

These files and the bundled static validator make this directory portable. Do
not depend on another skill directory or on files from the repository that
contains this skill. A Harness checkout is optional for authoring and required
only for the authoritative compiler check.

## Non-negotiable behavior

- Accept exactly one approved proposal. An existing task may be read only as
  optional implementation evidence and never overrides the proposal.
- Work remote-first and resolve official evidence to immutable refs. Never
  execute untrusted repository code, installs, training, project tests, task
  images, Solution, or evaluator merely to research or package the task.
- Ask only for contributor-owned choices or inaccessible facts that can change
  task semantics. Investigate ordinary repository and packaging facts yourself.
- Keep Environment, Verifier, and baseline Solution as separate decisions.
  Freeze the Environment-to-Verifier interface before writing evaluation code.
- Write no task file until the contributor confirms one consolidated final
  assumption review.
- Immediately before every write, verify the exact target is absent. Never
  overwrite an existing target without explicit authorization for that path.
- Generate a Dockerfile-based complete package by default. Use a prebuilt image
  only when the contributor supplied or approved that exact real image. Never
  invent an image, digest, credential, measurement, bundle, runner, asset, or
  placeholder.
- `tests/` and evaluator assets are normal task-owned files injected only into
  Judge. Missing local Docker/GPU execution never justifies a compile-only
  fixture.
- Complete generation Layers 1–3 and stop. Environment build, fresh-container,
  baseline, GPU, and real-Agent execution remain pending for the separately
  invoked `harbor-task-validator` skill.
- The independent workflow permits one initial review, one targeted re-review,
  and at most two correction rounds. There is no third automatic review.

## Working state

Keep concise internal ledgers for:

1. proposal decisions and gaps;
2. verified repository and reported-result evidence;
3. implementation assumptions awaiting final confirmation; and
4. blockers.

Do not expose these as a proposal-schema questionnaire. Show only material
decisions and the final confirmation.

## Workflow

### Stage 1 — preflight the approved proposal

Classify the complete proposal into confirmed semantics, repository evidence to
verify, packaging assumptions, and material gaps. It must establish:

- official repository and immutable commit or exact tag to resolve;
- iterative model-development question and candidate-owned deliverable;
- traceable reference baseline, evaluation path, and reported result or `not
  reported`;
- fixed evaluation, finite scalar reward, direction, and visible feedback;
- editable and prohibited scope, network/data decisions, and Work plus
  single-evaluation compute estimates.

Do not demand reproduced results, measured variance, verified runtime, or final
Verifier claims at proposal stage. Preserve `reported, not yet reproduced` and
protocol-mismatch labels. Ask one small related question group only when a
contributor-owned answer would materially change the task; otherwise continue.

### Stage 2 — verify repository evidence

Follow repository research. Resolve moving refs to the approved immutable SHA;
trace baseline implementation/artifact, entrypoint, configuration, evaluator,
dependencies, data, checkpoints, and reported metrics through official sources.
If evidence changes the scientific baseline, workload, metric, action space,
allowed data, network need, or compute feasibility, ask the contributor rather
than silently changing the approved experiment.

Keep reported measurements separate from reproduced measurements. A nearby
paper or model-card number with any protocol difference is context, not the
matched baseline result.

### Stage 3 — freeze Environment-to-Verifier interface

Follow the Environment reference. Fix the effective absolute WORKDIR, exact
source and starting assets, candidate-owned and prohibited paths, complete
on-disk deliverable, Verifier entry interface, optional public candidate
self-check, dependencies, build inputs, resources, task runtime network modes,
and operator provider/proxy needs.

Define the starting state after every build/install/init operation that can
write WORKDIR. Remove, relocate, or deliberately represent ignored, untracked,
generated, compiled, cache, log, and install products; keep the pristine
baseline representation distinct from the candidate allowlist. Require the
untouched post-build state to pass the same pre-scoring scope/integrity gate.

For split WORKDIR, Judge reloads a read-only candidate snapshot. Only closed,
flushed, complete files are deliverables; processes, GPU state, sockets, and
caches are not.

### Stage 4 — design Verifier

Follow the Verifier reference. Implement the confirmed fixed evaluation in the
task-owned `tests/` tree for RSI-Harness's shared Base/Judge model; do not create
a separate verifier image.

The control flow must run offline with baked tooling, score a valid baseline or
no-op continuously, and normally evaluate only the submitted candidate using
its absolute metric. A live paired baseline is exceptional and requires the
contributor's explicit choice after its compute and failure cost are shown.

Apply correctness and task-specific integrity gates before quality scoring.
Expose only confirmed-safe feedback, but return actionable structured
diagnostics for candidate-owned artifacts and configuration. Write the declared
candidate-failure scalar only for recognized candidate-caused failure. Crash,
timeout, dependency, evaluator, infrastructure, and incomplete paths write no
reward. Keep results in memory and exclusive-create
`/logs/verifier/reward.json` once only after complete success.

Apply every conditional framework rule in the Verifier reference that matches
the evaluator, including Ray/vLLM runtime address discovery and lm-eval
multi-filter accounting. Describe shared Base/Judge limitations honestly.

### Stage 5 — derive operational configuration

- Size `verifier.timeout_sec` for one complete `tests/test.sh` plus headroom.
- Size `agent.timeout_sec` for all Work time, every synchronous submission wait,
  and operational margin:

  ```text
  agent timeout >= work research/training
                   + max_submissions × worst-case Judge duration
                   + operational margin
  ```

- Put `max_submissions`, primary reward, direction, GPU pool, Agent model, and
  reasoning effort in README run options, not task fields.
- Treat CPU, memory, planning storage, shared memory, build timeout, Compose
  user, and common env as shared-service settings. Keep Work/Judge GPU, user,
  runtime network, timeout, and Judge env overrides phase-specific.
- Do not apply Terminal-Bench's 18,000-second cap. Preserve the proposal's
  compute estimate and surface resource review; never silently shorten it.
- Omit `gpu_types` and all `allow_internet` fields.

### Stage 6 — one consolidated contributor confirmation

After reading the template and generation-validation references, present one
compact plain-language review containing:

- identity, official repo/ref, research question, and candidate deliverable;
- baseline, reported-result status and protocol match, and how Solution
  materializes it without training or evaluation;
- starting state, editable/prohibited scope, fixed evaluation, reward,
  candidate-failure scalar, feedback, hidden inputs, and leakage controls;
- candidate-only scoring or the contributor's explicit paired-baseline choice;
- resources, runtime network/data/provider requirements, timeouts, submissions,
  snapshot, storage/build assumptions, and shared-runtime limitations;
- Dockerfile or exact approved image, exact absent destination, and validation
  now versus execution later.

Label conservative assumptions and unverified measurements truthfully. Keep
file inventories, evaluator internals, cleanup mechanics, and test matrices out
unless they require a contributor decision. Ask for confirmation and stop; do
not write files in that response.

### Stage 7 — generate the package

After confirmation, select and recheck one absent destination, normally beside
the proposal, and generate the task template:

```text
<slug>/
├── task.toml
├── instruction.md
├── README.md
├── environment/Dockerfile
├── environment/docker-compose.yaml  # only when needed
├── tests/test.sh and evaluator assets
└── solution/solve.sh
```

Every task text file carries the Harbor canary in real comment syntax;
Instruction task paths are absolute; package name is `rsi/<directory>`; and
taxonomy is `ML` plus `Training`, `Inference`, `Evaluation`, or `Kernels`. Every
`/tests/...` reference resolves inside the package. Do not add a Terminal-Bench
timeout suffix, separate verifier, CTRF artifact, undeclared bundle, or synthetic
runner.

`solution/solve.sh` is required and idempotently materializes the proposal's
reference baseline in candidate-owned workspace paths from Environment assets.
It never trains, evaluates, reads `/tests`, submits, or writes reward.
RSI-Harness does not execute it; `tests/test.sh` is the scoring path.

### Stage 8 — validate generation Layers 1–2 once

Follow generation validation. With a known Harness checkout, run one combined
command:

```bash
python3 <skill-dir>/scripts/validate_task.py /absolute/path/to/task \
  --harness-root /absolute/path/to/RSI-Harness
```

Otherwise omit `--harness-root` and report compiler validation pending. Do not
run both variants. Fix every error and fix or disposition every warning. Perform
the single Generator review, reread generated files once, trace evaluator
terminal paths and post-build starting-state closure, and reconfirm the target.
Do not build or run the task.

### Stage 9 — bounded independent review and handoff

Follow the independent-review reference. Give one fresh read-only reviewer the
approved proposal, confirmation summary, generated task, official evidence,
Harness evidence, and validation results. The generator owns all edits.

After a fixable initial `BLOCKER` or `MAJOR`, make correction round one, rerun
affected validation, and request one targeted re-review, preferably from the
same reviewer. After a second fixable material finding, make final correction
round two, rerun validation, and stop without a third review.

If the initial reviewer is unavailable, report review pending. If only
re-review is unavailable, report the initial decision and correction without
claiming `PASS`. End as `PASS`, `REVIEWED_WITH_FINAL_CORRECTIONS`, or `BLOCKED`
under the reference definitions.

Handoff reports destination/tree, official repo/ref,
Environment/Verifier/Solution/timeouts, exact validation and review outcomes,
correction count, final status, pending execution checks, and known limitations.
Do not execute Environment build, fresh-container, baseline, GPU, evaluator, or
real-Agent checks in this skill; name them as pending for
`harbor-task-validator`.
