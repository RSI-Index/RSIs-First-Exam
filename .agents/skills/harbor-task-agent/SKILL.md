---
name: harbor-task-agent
description: Use when an approved RSI-Index AutoResearch proposal must become an RSI-Harness-compatible Harbor task, or an existing Harbor task without a proposal needs standards-based refinement.
---

# Harbor Task Agent

Turn an approved AutoResearch proposal or an existing Harbor task into a
self-contained RSI-Harness task. Copying the task directory into an RSI-Harness
workspace must be enough to build and run it, apart from RSI-Harness, declared
resources, credentials, and operator inputs documented by the task.
RSI-Harness is authoritative; use Terminal-Bench conventions only where this
skill explicitly retains them.

## Load references by stage

Do not read every reference up front. Read each selected file completely before
acting on its stage.

1. For an existing task without an approved proposal, read
   [references/existing-task-refinement.md](references/existing-task-refinement.md)
   before inspecting it. Skip this reference in proposal mode.
2. Before inspecting the semantic source or remote repository, read
   [references/repository-research.md](references/repository-research.md).
3. Before fixing Base/Work, read
   [references/environment-design.md](references/environment-design.md).
4. After that interface is frozen and before authoring evaluation, read
   [references/verifier-design.md](references/verifier-design.md).
5. Before the contributor confirmation, read
   [references/task-template.md](references/task-template.md) and
   [references/validation-rules.md](references/validation-rules.md).
6. Immediately before independent review, read
   [references/independent-review.md](references/independent-review.md).

These references and scripts make the skill portable. Do not depend on files
from the repository that contains the skill. A Harness checkout is optional for
authoring and required only for the authoritative compiler check.

## Non-negotiable behavior

- Select exactly one semantic mode. An approved proposal controls when present;
  an existing task may then serve only as implementation evidence. Route a
  still-draft proposal back to proposal work. Refinement mode never invokes
  proposal-agent or creates a proposal artifact.
- Work remote-first and resolve official source evidence to immutable refs.
  Never execute untrusted repository code, installs, training, task images, or
  source-task Solution/Verifier merely to research or package the task.
- Keep Environment, Verifier, and baseline Solution as separate decisions.
  Freeze the Environment-to-Verifier interface before writing evaluation code.
- Ask only for contributor-owned choices or inaccessible facts. In refinement
  mode, recover semantics from the task and official evidence, decide ordinary
  implementation details, ask no proposal-field questions, and stop when a core
  semantic gap cannot be recovered.
- Write no task file until the contributor confirms one consolidated final
  assumption review.
- Never overwrite a target. Immediately before every write, check the exact
  path; if it exists, obtain permission for that directory or choose another
  path, then check again.
- Generate a Dockerfile-based complete package by default. Use a prebuilt image
  only when the contributor supplies or approves that exact real image. Never
  invent images, digests, credentials, measurements, external evaluator
  bundles, runners, assets, or placeholders.
- Treat `tests/` and its evaluator assets as normal task-owned files injected
  only into Judge. Lack of local Docker/GPU execution does not justify a
  compile-only fixture; report execution checks as pending.
- Do not claim a build, baseline/no-op, adversarial, GPU, or Agent check unless
  it actually ran.
- Complete Stage 9 before handoff. Never run Environment preflight or another
  execution layer automatically. First explain its network, disk, image,
  container, GPU, duration, retained-state, and cleanup effects; execute only
  after explicit contributor authorization in a later response.
- The independent workflow permits one initial review, one targeted re-review,
  and at most two generator correction rounds. There is no third review unless
  the contributor explicitly requests a deeper audit.

## Working state

Maintain concise internal ledgers for:

1. verified semantic-source and repository evidence;
2. confirmed proposal decisions or the recovered-task brief;
3. implementation assumptions awaiting final confirmation; and
4. blockers and material source-task conflicts.

Do not expose these as a schema questionnaire. Show only decisions needing
attention and the final review.

## Workflow

### Stage 1 — Approve one semantic source

In proposal mode, classify the complete proposal into confirmed semantics,
evidence to verify, packaging assumptions, and gaps. It must establish:

- official repository plus immutable commit;
- iterative model-development question and candidate-owned deliverable;
- traceable baseline, evaluation path, and reported result or `not reported`;
- fixed evaluation, finite scalar, direction, and visible feedback;
- editable/prohibited scope, network/data decisions, and Work plus
  single-evaluation compute estimates.

Do not require reproduced results, measured variance, verified runtime, or
final-verifier claims at proposal stage. Preserve `reported, not yet reproduced`
and protocol-mismatch labels. Ask one small related question group only when a
contributor-owned fact can change the task; otherwise continue.

In refinement mode, follow
[references/existing-task-refinement.md](references/existing-task-refinement.md):
inventory the source package, recover its internal task brief, research official
evidence, resolve material conflicts, and stop on an unrecoverable core gap. The
confirmed recovered-task brief becomes the approved semantic source for the
remaining stages.

### Stage 2 — Verify repository evidence

Follow [references/repository-research.md](references/repository-research.md).
Resolve moving refs to the approved immutable SHA; trace the baseline,
entrypoint, configuration, evaluator, dependencies, artifacts, and reported
metric through official sources. Investigate ordinary packaging facts yourself.

In proposal mode, ask when evidence would change the confirmed baseline or
evaluation. In refinement mode, disposition conflicts in the single final
review and stop when equally plausible core interpretations remain. Packaging
mechanics are Agent-owned decisions.

### Stage 3 — Freeze Environment-to-Verifier interface

Follow [references/environment-design.md](references/environment-design.md).
Record the effective absolute WORKDIR; exact source and starting assets;
candidate-owned and prohibited paths; fully materialized on-disk deliverable and
Verifier entry interface; optional public candidate self-check; dependency and
asset provenance; Work resources/build budget; common and Judge-only env; Agent
network mode; and any provider/proxy prerequisite.

The starting state is the complete WORKDIR after every build operation that can
write it. Close this state only after installs, generated metadata, compilation,
import probes, and initialization. A Git commit alone does not represent ignored
or untracked products: remove, relocate, or deliberately represent them, keep
that baseline distinct from the candidate allowlist, and require the untouched
post-build workspace to pass Judge's pre-scoring scope/integrity gate.

For split WORKDIR, design Judge reload against a read-only snapshot. Candidate
artifacts must be closed, flushed, and complete before submission; processes,
GPU state, sockets, and caches are not deliverables. The `environment/` build
context must contain no `tests/`, Solution, hidden inputs, credentials, or task
volumes.

### Stage 4 — Design Verifier

Follow [references/verifier-design.md](references/verifier-design.md). Build the
confirmed fixed evaluation in the task-owned `tests/` tree for RSI-Harness's
shared Base/Judge model; do not create a separate verifier image.

The resulting control flow must:

- run offline with tooling already installed;
- score a valid baseline/no-op continuously and, by default, evaluate only the
  submitted candidate using its absolute metric;
- require explicit contributor choice before exceptional live paired-baseline
  scoring;
- apply correctness and task-specific integrity gates before quality scoring;
- expose only confirmed-safe stdout/stderr feedback while giving actionable,
  structured diagnostics for candidate-owned failures;
- finalize the declared candidate-failure scalar only for a recognized
  candidate-caused failure, never for evaluator or infrastructure failure; and
- retain results in memory and directly create
  `/logs/verifier/reward.json` exactly once after a complete successful outcome,
  leaving no reward on crash, timeout, infrastructure, or incomplete evaluation.

Treat candidate code and Work-modified system state as untrusted. Apply every
conditional framework contract in `verifier-design.md` that matches the actual
evaluator, and describe shared-environment limitations honestly.

### Stage 5 — Derive operational configuration

- Set `verifier.timeout_sec` for one complete `tests/test.sh` run plus headroom.
- Set `agent.timeout_sec` for all Work time, every synchronous submission wait,
  and margin:

```text
agent timeout >= Work research/training
                 + max_submissions × worst-case Judge time
                 + operational margin
```

- Document `max_submissions`, primary reward, direction, GPU pool, model, and
  reasoning effort as README run options, not task fields.
- Treat CPU, memory, planning storage, shared memory, build timeout, Compose
  user, and common env as shared-service settings; keep supported phase GPU,
  user, network, timeout, and Judge env overrides separate.
- Do not apply Terminal-Bench's 18,000-second cap. If full submissions are very
  expensive, suggest fewer submissions or a contributor-approved proxy; never
  silently shorten the experiment.
- Omit `gpu_types` and every `allow_internet` field. Use the current Harness
  Work/Judge GPU and network fields defined by the references.

### Stage 6 — Obtain final contributor confirmation

After reading the template and validation references, present one compact,
plain-language review covering:

- identity, repository/ref, recovered or proposed research semantics;
- baseline, reported-result status and protocol match, plus how
  `solution/solve.sh` materializes it without training or evaluation;
- starting state, deliverable, editable/prohibited scope;
- fixed evaluation, reward/aggregation/direction/units, candidate-failure
  scalar, diagnostics, feedback, hidden inputs, and leakage controls;
- candidate-only scoring or the contributor's explicit live-pairing choice;
- optional public candidate check and Judge-only checks when material;
- Work/Judge resources, network/data/provider requirements, timeouts,
  submissions, WORKDIR/snapshot, storage/build assumptions, and shared-runtime
  limitations;
- Dockerfile or exact approved image; and validation now versus execution later.

Label conservative assumptions as assumptions. Keep inventories, evaluator
internals, cleanup mechanics, and test matrices out unless they require a
contributor decision. Ask for confirmation and stop; do not write in that
response. In refinement mode this is the only question before generation and
also names source-task conflicts, their dispositions, the collision-free sibling
destination, and the promise that the source stays unchanged.

### Stage 7 — Generate package

After confirmation, select and recheck one absent destination. Proposal mode
defaults beside the proposal. Refinement mode defaults to a collision-free
sibling, using `<source-slug>-refined` only when it satisfies the three-token
limit.

Generate [references/task-template.md](references/task-template.md):

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

Every task text file carries the exact Harbor canary in a valid comment;
Instruction task paths are absolute; taxonomy is `ML` plus `Training`,
`Inference`, `Evaluation`, or `Kernels`; every `/tests/...` reference resolves
inside the package. Do not add a Terminal-Bench timeout suffix, separate
Verifier, CTRF artifacts, undeclared bundle, or synthetic runner.

`solution/solve.sh` is required and idempotently materializes the approved
reference baseline in candidate-owned workspace paths from Environment assets.
It never trains, evaluates, reads `/tests`, submits, or writes reward.
RSI-Harness does not execute it; `tests/test.sh` is the scoring path.

### Stage 8 — Validate Layers 1–2 once

Follow [references/validation-rules.md](references/validation-rules.md). If a
known Harness checkout exists, run the combined command once:

```bash
python3 <skill-dir>/scripts/validate_task.py /absolute/path/to/task \
  --harness-root /absolute/path/to/RSI-Harness
```

Otherwise omit `--harness-root` and report the authoritative compiler pending.
Do not run both variants. Fix every error; fix or disposition each warning.
Then perform the single Generator final review defined by Layer 2, including
semantic fidelity, evaluator terminal paths, post-build starting-state closure,
and truthful pending checks. Reread the actual generated files once and confirm
the destination did not change. Do not run execution Layers 4–5 here.

### Stage 9 — Independent review and handoff

Follow [references/independent-review.md](references/independent-review.md).
Dispatch one fresh read-only reviewer with the complete mode-specific evidence.
The generator owns corrections. After a fixable initial `BLOCKER` or `MAJOR`,
make correction round one, rerun affected validation, and request one targeted
re-review, preferably from the same reviewer. After a second fixable material
finding, make the final correction round and rerun validation, then stop; there
is no third review.

If an independent reviewer is unavailable initially, report review pending. If
only the re-review is unavailable, report the initial decision and correction
without claiming `PASS`. End truthfully as `PASS`,
`REVIEWED_WITH_FINAL_CORRECTIONS`, or `BLOCKED` under the reference definitions.

Handoff reports the destination/tree, repository/ref, Environment/Verifier/
Solution/timeouts, exact validation results, review decisions and dispositions,
correction count, final status, pending execution checks, and known security or
reproducibility limitations.

### Post-handoff execution — Layers 4–5

Follow [references/validation-rules.md](references/validation-rules.md).
Layer 4 first plans one combined Environment preflight and discloses its
requirements; it executes only after explicit contributor authorization in a
later response. Layer 5 contains separately authorized scientific, GPU, and
end-to-end checks. Neither layer is part of default generation.
