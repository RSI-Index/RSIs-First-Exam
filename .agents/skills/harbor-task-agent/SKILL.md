---
name: harbor-task-agent
description: Use when converting an approved RSI-Index AutoResearch proposal into a self-contained RSI-Harness-compatible Harbor task.
---

# Harbor Task Agent

Turn an approved AutoResearch proposal into a self-contained, runnable RSI-Harness task. Copying the generated task directory into an RSI-Harness workspace must be sufficient to build and run it, apart from RSI-Harness itself, the declared GPUs, provider credentials, and other operator inputs explicitly documented in the task. RSI-Harness is the authoritative runtime contract. Terminal-Bench conventions are only useful where this skill explicitly preserves or adapts them.

## Load references by stage

Do not read every reference up front.

1. Before inspecting a proposal or remote repository, read [references/repository-research.md](references/repository-research.md) completely.
2. Before deciding the Base/Work environment, read [references/environment-design.md](references/environment-design.md) completely.
3. After the Environment interface is fixed and before authoring the task-owned evaluation, read [references/verifier-design.md](references/verifier-design.md) completely.
4. Before the final contributor assumption review, read [references/task-template.md](references/task-template.md) and [references/validation-rules.md](references/validation-rules.md) completely.
5. After generation and validation, immediately before dispatching the independent reviewer, read [references/independent-review.md](references/independent-review.md) completely.

The references are part of this portable skill. Do not require files from the repository that happens to contain the skill. RSI-Harness itself is optional for static authoring and required only for the authoritative compiler check.

## Non-negotiable behavior

- Start from an approved proposal. If the scientific question, baseline, evaluation, action space, or compute decision is still draft, route it back to proposal work instead of silently defining it here.
- Work remote-first. Verify the official repository and immutable commit from remote evidence; use a local checkout only when the contributor explicitly points to one or remote inspection is insufficient.
- Never execute untrusted repository code, training, installation hooks, or task images merely to research or package the task.
- Keep Environment, Verifier, and baseline Solution as separate decisions. Freeze the Environment-to-Verifier interface before writing Verifier logic.
- Ask only for contributor-owned choices or inaccessible facts that can materially change the task. Explain operational questions in plain language; contributors need not know Harbor internals.
- Do not write any task file until the contributor has confirmed one consolidated final assumption review.
- Never overwrite an existing target. Immediately before every write, check the exact target path. If any target exists, obtain explicit permission for that exact directory or choose a new target, then check again.
- Generate a Dockerfile-based, self-contained package by default. Use a prebuilt image only when the contributor explicitly supplies or approves that exact real image. Never invent an image registry, tag, digest, external evaluator bundle, runner, or asset location.
- `tests/` and every evaluator asset it needs are normal task-owned files. Harness-only Judge injection keeps them unavailable to Work at runtime; it does not imply a separately delivered private bundle.
- Do not downgrade an ordinary deliverable to a compile-only fixture because Docker, GPUs, or a full evaluation cannot be run during authoring. Produce a complete task and report unperformed execution checks as pending. Create a deliberately incomplete fixture only when the contributor explicitly requests a fixture or compiler demo.
- Stop instead of leaving placeholders, invented credentials, fabricated measurements, missing task-owned assets, or an evaluator that cannot implement the confirmed protocol.
- Do not claim Docker, Oracle/baseline, no-op, adversarial, GPU, or real-agent checks unless they were actually run.
- Never run Environment preflight or any other execution check automatically.
  Finish Stage 9 and hand off the task first. Explain the exact network, Docker
  disk, image/container, GPU, time, and cleanup implications, then wait for
  explicit contributor authorization in a later response before execution.
- Complete the bounded independent-review workflow in Stage 9 before handoff. The generator's own reread, static validation, and compiler check do not replace the initial independent review. Do not exceed two independent review passes or two generator correction rounds unless the contributor explicitly requests a deeper audit.

## Working state

Maintain four concise internal ledgers:

1. verified proposal and repository evidence;
2. contributor-confirmed decisions;
3. implementation assumptions awaiting final confirmation; and
4. blocking gaps.

Do not turn these ledgers into a schema questionnaire. The contributor should see only the decisions that need attention and the final consolidated assumption review.

## Workflow

### Stage 1 — Proposal preflight

Read the complete proposal and classify every statement as confirmed task semantics, repository evidence to verify, packaging assumption, or gap.

Require at least:

- official repository URL and resolved 40-character commit SHA;
- iterative model-development question and candidate-owned deliverable;
- traceable baseline, its evaluation path, and its officially reported result or an explicit `not reported` status;
- fixed evaluation protocol, scalar metric, direction, and visible-feedback boundary;
- explicit editable scope and prohibited actions;
- Work and single-evaluation compute estimates;
- network and additional-data decisions.

Do not require baseline reproduction, measured variance, verified runtime, or claimed final-verifier behavior at proposal stage. Preserve “reported, not yet reproduced” labels and never turn a nearby but mismatched result into a matched baseline claim.

If a required author name/email, license permission, contributor-supplied prebuilt image or asset, registry access, runtime provider route, or other contributor-owned fact is missing, ask one small related question group. Otherwise continue without interaction. Do not manufacture a dependency on an external evaluator asset when the evaluator can be included under `tests/`.

### Stage 2 — Repository evidence

Follow [references/repository-research.md](references/repository-research.md). Resolve moving refs to the proposal SHA, verify that material baseline, entrypoint, config, evaluator, and dependency paths exist at that SHA, and search official repository documentation, releases, model cards, papers, and author-published result artifacts for the reported baseline value.

Investigate packaging facts yourself. Do not ask the contributor to locate ordinary files, commands, dependencies, or existing benchmark behavior that remote repository evidence can answer.

If the repository contradicts the proposal in a way that changes the baseline or evaluation, show the evidence and ask the contributor. Packaging inconveniences that do not change task semantics are implementation decisions.

### Stage 3 — Environment design

Read [references/environment-design.md](references/environment-design.md). Design the clean Base image and persistent Work state before writing the task-owned tests.

Fix and record this interface:

- effective WORKDIR: `/workspace` is the default convention, but any evidenced existing absolute POSIX path is valid; `/` deliberately selects full-rootfs snapshots;
- exact source state and starting assets visible to Work;
- candidate-owned paths and prohibited paths;
- the fully materialized on-disk candidate and commands/interfaces Verifier may invoke after starting clean from the snapshot;
- when nontrivial candidate artifacts admit useful public-only checks, the exact optional read-only candidate self-check command and the checks that remain Judge-only;
- pinned runtime dependencies and immutable asset provenance;
- Work GPU count, CPU, memory, storage estimate, shared memory, and build budget;
- common `[environment.env]` values and any Judge-only `[verifier.env]` overrides;
- Agent network policy and any exact provider/proxy reachability requirement.

Treat the starting workspace as the filesystem state after every image-build
operation that can write the WORKDIR, including editable installs, package
metadata generation, compilation, import checks, and repository
initialization. Close that state before Verifier design. A Git commit alone is
not a complete baseline when ignored or untracked files can exist: remove or
relocate build byproducts, deliberately represent the accepted post-build
state, and keep the candidate modification allowlist separate. The unchanged
post-build workspace must satisfy the same pre-scoring scope and integrity gate
that Judge will apply after an Agent submission.

For a non-root WORKDIR, Judge mounts that directory read-only. Confirm that candidate loading and evaluation need no cache, compilation output, checkpoint update, or scratch write there. Docker commit captures files, not live processes or GPU memory: before `rsi-submit`, candidate code/config/checkpoints must be closed, flushed, and complete on disk. Judge must start any local serving process afresh and reload the candidate from the snapshot.

The Environment build context is exactly `environment/`. It must not contain or copy `tests/`, `solution/solve.sh`, hidden inputs, expected answers, or private baseline artifacts. Do not author task volumes or Docker `VOLUME` instructions.

### Stage 4 — Verifier design

Only after the interface above is stable, read [references/verifier-design.md](references/verifier-design.md). Design the task-owned fixed evaluation against the confirmed protocol.

RSI-Harness uses a clean Base/Judge derived from the shared Environment plus a snapshot of Work. It does not support Harbor's separate verifier image. The generated task contains its complete `tests/` tree; RSI-Harness withholds that tree from Work and injects it into Judge at submission time. This runtime visibility rule is not an external bundle requirement. Use `tests/test.sh`, not `tests/Dockerfile`.

The Verifier must:

- run entirely without network or runtime package installation;
- print all safe feedback and metric diagnostics to stdout/stderr;
- create no task-authored intermediate result files;
- keep the reward in memory and write `/logs/verifier/reward.json` directly exactly once, only after complete successful evaluation;
- emit one finite primary scalar named `reward`, with the direction declared in the README run command;
- leave no reward file on timeout, crash, incomplete evaluation, or infrastructure failure;
- give a valid continuously scored baseline/no-op result when the baseline itself is valid;
- evaluate only the selected candidate during a normal candidate submission and use its absolute metric as reward; do not rerun the reference baseline merely to print a live delta;
- enforce correctness and task-specific anti-cheat controls before performance/quality scoring;
- report candidate-owned artifact, configuration, and checkpoint failures with complete structured diagnostics that identify the failing absolute path, field when applicable, and failed condition, plus safe expected/actual values and a repair hint when useful;
- reserve vague or redacted diagnostics for hidden evaluation information and evaluator internals, not for errors in files the Agent owns; and
- finalize the contributor-confirmed candidate-failure scalar for a validly detected candidate failure, chosen from the declared score domain and direction rather than hard-coding `0.0`; leave no reward for validator crashes or infrastructure failures.

When the fixed evaluator uses Ray, vLLM data parallelism, multi-node vLLM, or
another vLLM execution path that needs a routable host address, the task owns
that framework-specific bootstrap. It must derive the disposable Judge
container's IPv4 at runtime and set `VLLM_HOST_IP` before initializing or
spawning Ray/vLLM. Never bake an IP into task configuration or fall back to an
unspecified address; address-discovery failure is infrastructure failure with
no reward. Do not make this a Harness-global requirement for unrelated tasks.

Every submission exposes the complete `tests/test.sh` stdout/stderr stream to Work at `/run/rsi-harness/feedback/agent-N.log`, plus a footer containing round, status, reward, optional score, exit code, timeout flag, duration, remaining submission budget, and any error. `rsi-submit --list` exposes submission history. Therefore stdout/stderr is the intentional Agent-visible feedback channel and must contain only the confirmed safe feedback—never hidden cases, gold answers, secrets, or undeclared per-example details. Do not confuse the bounded in-memory/report `output_limit_bytes` field with this separately captured durable file: the feedback log is the complete stream. There is no feedback-hidden final Judge phase inside RSI-Harness, so never promise one; if a proposal requires hidden final evaluation, distinguish an external final evaluation from the in-Harness development Judge and obtain contributor confirmation.

Treat candidate code and all Work-modified system state as untrusted. State the shared-environment isolation limitation honestly; do not claim the protection of an independent verifier image.

### Stage 5 — Operational configuration

Derive but do not hide run-time choices:

- `verifier.timeout_sec` covers one complete `/tests/test.sh` evaluation plus headroom;
- `agent.timeout_sec` covers all Work research/training, every synchronous `rsi-submit` Judge wait, and operational headroom;
- the Terminal-Bench 18,000-second timeout cap does not apply;
- `max_submissions`, `primary_reward`, `score_direction`, GPU pool, model, and reasoning effort are RSI-Harness run options documented in `README.md`, not invented task fields.
- `cpus`, `memory_mb`, `storage_mb`, `shm_size`, and `build_timeout_sec` describe the shared main service used by both Work and Judge; Compose/base user is common, while supported phase user overrides, GPU allocation, network, phase timeout, and Judge-only env overrides remain phase-specific.

Use this conservative budget relation:

```text
agent timeout >= total Work research/training time
                 + max_submissions × worst-case Judge time
                 + operational margin
```

If every full submission takes many hours, suggest fewer submissions or the contributor-approved fixed proxy for development. Never shorten the scientific experiment without confirmation.

Omit `gpu_types` and every `allow_internet` field. Declare Work GPUs with `[environment].gpus` and Judge GPUs with `[metadata.rsi_harness.verifier].gpus`.

### Stage 6 — Final contributor assumption review

Read [references/task-template.md](references/task-template.md) and [references/validation-rules.md](references/validation-rules.md). Present one compact, plain-language review containing every material assumption:

- task slug and `rsi/<slug>` package name;
- repository identity and exact ref;
- reference baseline, reported result and source status, whether it matches the Judge protocol, and how `solution/solve.sh` materializes a separately scoreable baseline without training or evaluation;
- starting state, editable scope, prohibited actions, and final deliverable;
- fixed evaluation, correctness gate, scalar reward, direction, aggregation, units, candidate-failure scalar, actionable candidate-owned diagnostics, and exactly visible feedback;
- confirmation that candidate submissions score the candidate alone, or the explicit proposal requirement and runtime-drift evidence that justify an exceptional live paired baseline;
- when useful for a complex deliverable, the exact optional public candidate self-check command and which checks remain Judge-only;
- hidden/public inputs and leakage controls;
- Work and Judge GPU counts and whether they can reuse the same caller pool;
- Agent and Verifier network/data policy, including provider/proxy prerequisites;
- Agent timeout, per-submission Verifier timeout, their wall-clock calculation, and maximum submissions;
- effective WORKDIR, split/read-only versus full-rootfs snapshot mode, materialized deliverable/reload path, and any snapshot storage cost;
- common CPU, memory, storage, shared-memory, build-timeout and environment assumptions, plus Judge-only overrides;
- shared Base/Judge limitation, actual full submission feedback/footer, and Verifier trust boundary;
- confirmation that the required `solution/solve.sh` materializes the baseline workspace state while RSI-Harness leaves scoring to `tests/test.sh`;
- Dockerfile build by default, or the exact contributor-supplied prebuilt image when explicitly chosen, including a no-declared-volumes image preflight for the latter;
- validation that will be run now versus Docker/GPU/execution checks left pending.

Call out conservative assumptions as assumptions, not facts. Ask the contributor to confirm or correct the entire review. Do not write in the same response that asks for confirmation.

This is a contributor decision review, not an implementation audit. Keep it compact and explain only choices or assumptions that could materially change the task. Keep file inventories, evaluator internals, process/thread cleanup details, line-level evidence, and validation-test matrices out of the contributor review unless the contributor must decide among alternatives because of them.

### Stage 7 — Generate the package

After explicit confirmation, choose one destination, defaulting to `<slug>/` beside the proposal unless the contributor specified another location. Recheck existence immediately before writing.

Generate the contract in [references/task-template.md](references/task-template.md):

```text
<slug>/
├── task.toml
├── instruction.md
├── README.md
├── environment/
│   ├── Dockerfile                  # default; omit only for an explicit real prebuilt image
│   └── docker-compose.yaml        # only when a supported Compose field is needed
├── tests/
│   ├── test.sh
│   └── complete task-owned evaluator/assets
└── solution/
    └── solve.sh                   # required baseline materializer
```

Every task text file must contain the exact Harbor canary string in a comment. Keep all paths in `instruction.md` absolute. Use ML taxonomy `Training`, `Inference`, `Evaluation`, or `Kernels`. Do not add Terminal-Bench's standard timeout suffix, separate-verifier files, CTRF artifacts, or a full “optimal” solution. Every literal `/tests/...` reference must resolve to a file or directory included in the generated task. Do not reference undeclared `/tests/private`, an out-of-package evaluator bundle, or a synthetic runner.

`solution/solve.sh` is required. It idempotently materializes the proposal's reference baseline in the candidate-owned workspace, using starting assets already present in the Environment. It does not train, evaluate, access `/tests`, submit, or write reward. RSI-Harness does not execute it; `tests/test.sh` remains the only scoring path. For a no-op baseline, the script restores or clears candidate-owned state so the Judge takes its declared baseline path.

### Stage 8 — Validate (Layers 1–2)

Complete Layers 1–2 in [references/validation-rules.md](references/validation-rules.md): the static validator, the authoritative compiler when its checkout is available, and one Generator final review. Do not split proposal fidelity, evaluator control flow, and starting-state closure into repeated rereads; cover them together in that final review.

Choose one validation command. If no known RSI-Harness checkout is available,
run the included standard-library validator and report the compiler as pending:

```bash
python3 <skill-dir>/scripts/validate_task.py /absolute/path/to/task
```

If the contributor or workspace provides a known RSI-Harness checkout, use one
combined command instead; it runs the same static checks once and then invokes
the read-only authoritative compiler. Do not first repeat the static-only
command, and do not guess a checkout path from a stale editable installation:

```bash
python3 <skill-dir>/scripts/validate_task.py /absolute/path/to/task \
  --harness-root /absolute/path/to/RSI-Harness
```

Fix every error. Review warnings against task evidence; fix them or document why they are intentional. Reread the actual generated files once and verify the destination did not change. Compilation without Docker/GPU execution is valid evidence only for the compiler portion; it never justifies missing build files, evaluator assets, or runtime commands.

Use the single Layer 2 review checklist for proposal fidelity, reward control
flow, starting-state closure, and truthful pending checks. Do not repeat those
as separate audits. Do not perform Layers 4–5 during Stage 8.

### Stage 9 — Independent review and handoff (Layer 3)

Read [references/independent-review.md](references/independent-review.md). The reviewer is always read-only; the generating Agent owns every correction.

Use this fixed review budget:

1. Dispatch one fresh Agent that did not generate or edit the task for the full initial review. Give it the approved proposal, contributor-confirmed assumptions, generated task, repository evidence, known RSI-Harness checkout or documentation, and exact validation results. Do not prime it with the generator's preferred conclusion or ask it to modify files.
2. If the initial decision has a fixable `BLOCKER` or `MAJOR`, make correction round one and rerun every affected validation. Resume the same reviewer for one targeted re-review when possible. If that Agent is unavailable, use a fresh reviewer but provide the first report, correction dispositions, touched paths, and updated validation evidence so the pass remains a re-review rather than a new open-ended audit.
3. The re-review is the second and final independent review. If it reports a fixable `BLOCKER` or `MAJOR`, make correction round two and rerun affected validation, then stop. Do not dispatch a third reviewer. If the final correction cannot be completed or essential evidence remains unavailable, stop as `BLOCKED`.

If an independent Agent is unavailable for the initial pass, stop and report review as pending instead of substituting generator self-review. If it is unavailable only for the re-review, report the first decision, completed correction, and missing re-review rather than claiming `PASS`.

End Stage 9 with one truthful status:

- `PASS`: the initial review or re-review returned `PASS`; any accompanying minor findings were fixed or explicitly dispositioned.
- `REVIEWED_WITH_FINAL_CORRECTIONS`: the final allowed correction round addressed the second review's findings and affected validation passed, but those final edits were not independently re-reviewed. Never describe this status as reviewer `PASS`.
- `BLOCKED`: a material finding or evidence gap remains after the allowed workflow.

Do not start another automatic review/fix loop. A contributor may explicitly request a separate deeper audit later.

At handoff, report:

- generated destination and file tree;
- repository/ref actually used;
- Environment, Verifier, baseline Solution, and timeout decisions;
- exact static/compiler commands and outcomes;
- both independent-review decisions when applicable, every finding and disposition, the number of correction rounds, and the final Stage 9 status;
- remaining execution checks not performed;
- any known security or reproducibility limitation.

### Post-handoff execution (Layers 4–5)

Follow Layers 4–5 in [references/validation-rules.md](references/validation-rules.md).
They begin only after the Stage 9 handoff, never by default. Layer 4 is one
combined Environment preflight with a planning-and-authorization gate. Its
inspection container uses a private internal bridge so no-network Judge
addressing can be inspected without public egress. Layer 5
contains separately authorized scientific and end-to-end execution checks.
