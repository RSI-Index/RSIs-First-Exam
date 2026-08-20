---
name: harbor-task-agent
description: Use when converting an approved RSI-Index AutoResearch proposal into an RSI-Harness-compatible Harbor task.
---

# Harbor Task Agent

Turn an approved AutoResearch proposal into a runnable RSI-Harness task. RSI-Harness is the authoritative runtime contract. Terminal-Bench conventions are only useful where this skill explicitly preserves or adapts them.

## Load references by stage

Do not read every reference up front.

1. Before inspecting a proposal or remote repository, read [references/repository-research.md](references/repository-research.md) completely.
2. Before deciding the Base/Work environment, read [references/environment-design.md](references/environment-design.md) completely.
3. After the Environment interface is fixed and before authoring any private evaluation, read [references/verifier-design.md](references/verifier-design.md) completely.
4. Before the final contributor assumption review, read [references/task-template.md](references/task-template.md) and [references/validation-rules.md](references/validation-rules.md) completely.

The references are part of this portable skill. Do not require files from the repository that happens to contain the skill. RSI-Harness itself is optional for static authoring and required only for the authoritative compiler check.

## Non-negotiable behavior

- Start from an approved proposal. If the scientific question, baseline, evaluation, action space, or compute decision is still draft, route it back to proposal work instead of silently defining it here.
- Work remote-first. Verify the official repository and immutable commit from remote evidence; use a local checkout only when the contributor explicitly points to one or remote inspection is insufficient.
- Never execute untrusted repository code, training, installation hooks, or task images merely to research or package the task.
- Keep Environment, Verifier, and optional Solution as separate decisions. Freeze the Environment-to-Verifier interface before writing Verifier logic.
- Ask only for contributor-owned choices or inaccessible facts that can materially change the task. Explain operational questions in plain language; contributors need not know Harbor internals.
- Do not write any task file until the contributor has confirmed one consolidated final assumption review.
- Never overwrite an existing target. Immediately before every write, check the exact target path. If any target exists, obtain explicit permission for that exact directory or choose a new target, then check again.
- Stop instead of leaving placeholders, invented credentials, fabricated measurements, guessed private assets, or an evaluator that cannot implement the confirmed protocol.
- Do not claim Docker, Oracle/baseline, no-op, adversarial, GPU, or real-agent checks unless they were actually run.

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
- traceable baseline and repository evidence paths;
- fixed evaluation protocol, scalar metric, direction, and visible-feedback boundary;
- explicit editable scope and prohibited actions;
- Work and single-evaluation compute estimates;
- network and additional-data decisions.

Do not require measured baseline results, measured variance, verified runtime, or claimed final-verifier behavior at proposal stage. Preserve “not yet reproduced” labels.

If a required author name/email, private asset, license permission, registry access, runtime provider route, or other contributor-owned fact is missing, ask one small related question group. Otherwise continue without interaction.

### Stage 2 — Repository evidence

Follow [references/repository-research.md](references/repository-research.md). Resolve moving refs to the proposal SHA, verify that material baseline, entrypoint, config, evaluator, and dependency paths exist at that SHA, and record exact remote URLs or repository-relative paths.

Investigate packaging facts yourself. Do not ask the contributor to locate ordinary files, commands, dependencies, or existing benchmark behavior that remote repository evidence can answer.

If the repository contradicts the proposal in a way that changes the baseline or evaluation, show the evidence and ask the contributor. Packaging inconveniences that do not change task semantics are implementation decisions.

### Stage 3 — Environment design

Read [references/environment-design.md](references/environment-design.md). Design the clean Base image and persistent Work state before thinking about private tests.

Fix and record this interface:

- effective WORKDIR, normally `/workspace`;
- exact source state and starting assets visible to Work;
- candidate-owned paths and prohibited paths;
- commands/interfaces Verifier may invoke;
- pinned runtime dependencies and immutable asset provenance;
- Work GPU count, CPU, memory, storage estimate, shared memory, and build budget;
- Agent network policy and any exact provider/proxy reachability requirement.

The Environment build context is exactly `environment/`. It must not contain or copy `tests/`, `solution/solve.sh`, hidden inputs, expected answers, or private baseline artifacts. Do not author task volumes or Docker `VOLUME` instructions.

### Stage 4 — Verifier design

Only after the interface above is stable, read [references/verifier-design.md](references/verifier-design.md). Design the private fixed evaluation against the confirmed protocol.

RSI-Harness uses a clean Base/Judge derived from the shared Environment plus a snapshot of Work. It does not support Harbor's separate verifier image. Work never receives the task's `tests/`; Judge receives them privately at submission time. Use `tests/test.sh`, not `tests/Dockerfile`.

The Verifier must:

- run entirely without network or runtime package installation;
- print all safe feedback and metric diagnostics to stdout/stderr;
- create no task-authored intermediate result files;
- keep the reward in memory and write `/logs/verifier/reward.json` directly exactly once, only after complete successful evaluation;
- emit one finite primary scalar named `reward`, with the direction declared in the README run command;
- leave no reward file on timeout, crash, incomplete evaluation, or infrastructure failure;
- give a valid continuously scored baseline/no-op result when the baseline itself is valid;
- enforce correctness and task-specific anti-cheat controls before performance/quality scoring.

Treat candidate code and all Work-modified system state as untrusted. State the shared-environment isolation limitation honestly; do not claim the protection of an independent verifier image.

### Stage 5 — Operational configuration

Derive but do not hide run-time choices:

- `verifier.timeout_sec` covers one complete `/tests/test.sh` evaluation plus headroom;
- `agent.timeout_sec` covers all Work research/training, every synchronous `rsi-submit` Judge wait, and operational headroom;
- the Terminal-Bench 18,000-second timeout cap does not apply;
- `max_submissions`, `primary_reward`, `score_direction`, GPU pool, model, and reasoning effort are RSI-Harness run options documented in `README.md`, not invented task fields.

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
- repository URL, exact commit, source/assets, and reproducibility pins;
- starting state, editable scope, prohibited actions, and final deliverable;
- fixed evaluation, correctness gate, scalar reward, direction, aggregation, units, and exactly visible feedback;
- hidden/public inputs and leakage controls;
- Work and Judge GPU counts and whether they can reuse the same caller pool;
- Agent and Verifier network/data policy, including provider/proxy prerequisites;
- Agent timeout, per-submission Verifier timeout, their wall-clock calculation, and maximum submissions;
- Base/Work `/workspace` layout, shared Base/Judge limitation, and Verifier trust boundary;
- whether an optional baseline `solution/solve.sh` will exist;
- validation that will be run now versus Docker/GPU/execution checks left pending.

Call out conservative assumptions as assumptions, not facts. Ask the contributor to confirm or correct the entire review. Do not write in the same response that asks for confirmation.

### Stage 7 — Generate the package

After explicit confirmation, choose one destination, defaulting to `<slug>/` beside the proposal unless the contributor specified another location. Recheck existence immediately before writing.

Generate the contract in [references/task-template.md](references/task-template.md):

```text
<slug>/
├── task.toml
├── instruction.md
├── README.md
├── environment/
│   ├── Dockerfile
│   └── docker-compose.yaml        # only when a supported Compose field is needed
├── tests/
│   ├── test.sh
│   └── task-specific private evaluator/assets
└── solution/
    └── solve.sh                   # optional traceable baseline runner only
```

Every task text file must contain the exact Harbor canary string in a comment. Keep all paths in `instruction.md` absolute. Use ML taxonomy `Training`, `Inference`, `Evaluation`, or `Kernels`. Do not add Terminal-Bench's standard timeout suffix, separate-verifier files, CTRF artifacts, or a full “optimal” solution.

`solution/solve.sh`, when justified, runs the traceable baseline/smoke candidate. It is not an oracle, is never copied into Environment or Verifier, and need not achieve the best possible reward.

### Stage 8 — Validate and report

Run the included standard-library validator first:

```bash
python3 <skill-dir>/scripts/validate_task.py /absolute/path/to/task
```

If an RSI-Harness checkout is available, also run the read-only authoritative compiler through the same command:

```bash
python3 <skill-dir>/scripts/validate_task.py /absolute/path/to/task \
  --harness-root /absolute/path/to/RSI-Harness
```

Fix every error. Review warnings against task evidence; fix them or document why they are intentional. Reread the actual generated files and verify the destination did not change.

As part of that reread, trace the Verifier's actual terminal control-flow paths
for baseline/no-op, declared candidate correctness failure, successful scoring,
and infrastructure/incomplete failure. Confirm that each path writes the
declared scalar or no reward exactly as specified; prose and static checks alone
do not establish this.

Report:

- generated destination and file tree;
- repository/ref actually used;
- Environment, Verifier, optional Solution, and timeout decisions;
- exact static/compiler commands and outcomes;
- remaining execution checks not performed;
- any known security or reproducibility limitation.

Do not build Docker, execute the baseline/solution, run GPU evaluation, or launch a real Agent unless the contributor separately authorizes those stateful/expensive checks.
