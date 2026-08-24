# Validation rules

Post-generation validation has five layers. Passing an earlier layer never
proves a later one.

## Layer 1 — Included static validator

When no known Harness checkout is available, run:

```bash
python3 <skill-dir>/scripts/validate_task.py /absolute/path/to/task
```

It is read-only and uses the Python standard library. It checks required files and metadata, required Instruction sections, required executable baseline Solution, task identity/taxonomy, language-appropriate text canaries, absolute/effective WORKDIR declarations, major RSI-Harness unsupported fields, the supported Compose subset, obvious placeholder image references, dependency pins, suspicious Git-baseline/ignored-file scope closure, absolute instruction paths, missing task-owned `/tests/...` assets, Verifier network/install rules, common direct Python/shell intermediate writes, direct exclusive reward creation, README run options and a necessary timeout lower bound, and selected solution/test reference alignment. For common Ray/vLLM distributed launch patterns it also warns when no runtime `VLLM_HOST_IP` assignment is visible and rejects a statically configured or hard-coded IPv4. For `lm-evaluation-harness` it warns when logged-sample rows are used as a document-completeness count or when `n-samples` effective-count handling is not visible. This is conservative static evidence; it cannot prove address filtering, import order, subprocess propagation, failure cleanup, per-document filter grouping, or metric selection.

`ERROR` must be fixed. A `WARNING` needs evidence-backed review; warnings do not automatically make a task invalid. Static pattern and AST checks are conservative and cannot prove Docker buildability, full reward control flow, evaluator correctness, anti-cheat security, statistical validity, or successful GPU execution.

Every UTF-8 task text file must use a comment-capable format so the canary can
remain a real comment. Do not author commentless JSON text assets; use a
comment-capable source format or a genuinely binary asset and parse it
accordingly. `docker-compose.yml` is invalid because RSI-Harness recognizes only
`environment/docker-compose.yaml`.

## Layer 2 — RSI-Harness compiler and Generator final review

When a Harness checkout is available, use this as the single validation
command; it performs Layer 1 once and then the compiler portion of Layer 2:

```bash
python3 <skill-dir>/scripts/validate_task.py /absolute/path/to/task \
  --harness-root /absolute/path/to/RSI-Harness
```

The validator invokes `HarborTaskCompiler.compile(...)` with the reward, direction, and maximum-submission settings parsed from README. Compilation reads the task but does not build Docker, allocate GPUs, use network, run solution/tests, or mutate the task.

Do not replace the static layer with compilation. Harbor's parser may accept unknown metadata, and the compiler intentionally does not lint canaries, dependency pins, Dockerfile hygiene, reward lifecycle, or proposal semantics.
Do not also run the static-only command first; that would duplicate Layer 1.

After compilation, perform one Generator final review rather than several
overlapping rereads. Check all of the following together:

- repository URL and exact SHA match the approved proposal;
- baseline, workload, metric, direction, units, aggregation, and budget did not change;
- Instruction names the baseline, reported result/status, and fixed comparison;
  README gives concise official evidence and labels protocol differences;
  `solution/solve.sh` materializes that same baseline; Judge scores that same
  baseline/no-op under the fixed protocol;
- normal candidate submissions evaluate only the candidate and use its
  absolute metric; a live baseline pass appears only after the contributor
  explicitly chose it over the recommended candidate-only design after seeing
  its compute and failure cost; proposal words such as `matched` are not enough;
- public/hidden feedback boundary and leakage controls are implemented as confirmed;
- starting artifacts, editable scope, prohibited actions, and deliverable align;
- every Docker build/install/init operation that can mutate WORKDIR precedes
  starting-state closure, and the pristine post-build state is accepted by the
  same scope/integrity gate used for submissions, including ignored and
  untracked paths;
- Work/Judge resources, network/data policy, proxy needs, timeouts, and submissions match the final assumption review;
- the effective WORKDIR and snapshot mode match the image, candidate is fully materialized on disk, and split-WORKDIR evaluation is designed for read-only reload;
- complete Judge stdout/stderr and Harness footer visibility match the contributor-confirmed feedback contract;
- candidate-owned artifact/configuration/checkpoint failures emit a stable,
  actionable structure with specific code, absolute path, exact field and
  condition where applicable, and safe expected/actual/hint values; broad
  umbrella errors do not hide distinct repairs, while hidden evaluation data
  and internals remain protected;
- the candidate-failure scalar is explicitly derived from the declared score
  domain and direction, and a validator/internal failure cannot reach that
  reward path;
- conditionally, a useful public candidate self-check is an exact read-only
  absolute command baked into Environment, uses only public candidate
  invariants, writes no reward, and matches the independent Judge gate's
  diagnostic contract on generated non-hidden fixtures; simple tasks are not
  required to have one;
- every evaluator helper/input referenced under `/tests/...` is present in the generated task, with no undeclared external evaluator bundle or runner dependency;
- evaluator control flow implements valid baseline scoring, declared candidate
  correctness failure, successful final reward, and no reward for
  infrastructure/incomplete failure; and
- conditionally, distributed Ray/vLLM startup derives one valid Judge IPv4 at
  runtime before initialization, propagates it to every relevant child, never
  falls back to `0.0.0.0`, and leaves no reward on discovery/startup failure;
  and
- conditionally, `lm-evaluation-harness` completeness uses
  `n-samples[task].effective`; logged samples are uniquely grouped by
  `(doc_id, filter)`, every document has exactly the declared filters, one
  filter is selected explicitly for consumption, and the synthetic
  single-task/multi-filter regression passes; and
- no measured result, variance, runtime, or successful verification was invented.

The validator and compiler cannot prove this mapping because neither
reinterprets scientific intent or executes the evaluator. Trace the evaluator
from entrypoint to reward write and trace Dockerfile operations into the scope
gate during this one review.

## Layer 3 — Independent task review

After Layers 1–2, follow Stage 9 in `SKILL.md` and
[independent-review.md](independent-review.md). This is one bounded independent
review workflow, not another Generator audit: one initial review, at most one
targeted re-review, no third review, and no more than two Generator correction
rounds. Execution-only uncertainty belongs in Layers 4–5 unless the reviewer
can cite a concrete delivered defect and reachable consequence.

## Layer 4 — Authorized Environment preflight

This layer is stateful and potentially expensive. It occurs only after
the bounded independent review and task handoff. Never run it automatically
or in the same response that first presents its requirements.

For Environment preflight, first run the skill-owned planner without
`--execute`:

```bash
python3 <skill-dir>/scripts/preflight_task.py /absolute/path/to/task \
  --required-free-gb <conservative Agent estimate>
```

Explain the reported build/pull network endpoints, unresolved dynamic package
repositories, Docker data root and free-space requirement, image/container
names, expected duration, GPU requirement, retained state, and cleanup impact.
Stop and wait for explicit contributor authorization. Only in a later response
may the stateful mode run:

```bash
python3 <skill-dir>/scripts/preflight_task.py /absolute/path/to/task \
  --required-free-gb <same approved estimate> \
  --execute --acknowledge-authorized
```

This creates an untouched, no-GPU inspection container on a new private Docker
`--internal` bridge and mounts the task's tests read-only. That topology matches
the no-network Judge property relevant to container-local IPv4 discovery: the
container has a dynamic private address but no external route. It never runs
`tests/test.sh`, Solution, training, evaluation, submission, or reward writing.
The Agent then performs only the task-specific read-only diagnostics needed to
establish the starting-state gate and, when the Verifier uses distributed
Ray/vLLM, confirms that runtime address resolution returns one valid non-
loopback IPv4 without initializing the framework. Container/image/network
cleanup is another state-changing action and requires authorization.

Layer 4 combines Docker build or approved-image retrieval, image metadata
preflight, fresh untouched container creation, and the no-Agent
starting-state/scope gate. Report those outcomes together; do not duplicate
them as separate layers. It does not establish a baseline score or scientific
runtime result.

## Layer 5 — Separately authorized full execution

Authorize and report each requested check independently:

1. Baseline/no-op submission: valid continuous baseline score, safe feedback, no leaked cases.
2. Required external/manual baseline Solution materializer: restores the declared baseline workspace without training/evaluation, never presumed full score or Harness-executed.
3. Negative controls: prohibited edits, evaluator tampering, hard-coded cases, fabricated outputs, dependency/path changes, missing/incomplete evaluation.
4. Repeated deterministic/reliability runs and variance characterization.
5. Real GPU full evaluation within Verifier timeout and resource limits.
6. Real multi-round Agent run with submission budget, timeout, GPU release/reuse, feedback, and retained workspace.

Do not label a task “runtime-verified” solely because it passed Layers 1–4.
After Layer 3, a complete self-contained package may be reported as “statically
valid, Harness-compilable, and independently reviewed; execution checks
pending.” Missing Docker/build definitions, evaluator assets, or real runtime
commands make the package incomplete rather than a valid compile-only fixture.

## Terminal-Bench rule disposition

RSI-Harness is authoritative. Apply the referenced Terminal-Bench checks as follows:

| Reference rule | RSI AutoResearch disposition |
|---|---|
| Canary in comments | Keep; exact canary in every task text file using that format's actual comment syntax. |
| Dockerfile must not copy solution/tests | Keep and strengthen: Environment contains no private/answer material. |
| apt sanity | Keep warnings for missing update/cleanup; forbid apt version pins. |
| `FROM --platform` | Keep prohibition. |
| Absolute instruction paths | Keep for task file/component paths. |
| Shared solution/test output refs | Keep; document any shared expected artifact in instruction/task. |
| test.sh Python isolation | Adapt: no runtime venv/install; exact tooling is baked into shared Environment. |
| Required metadata/taxonomy | Adapt: category `ML`; subcategory `Training`, `Inference`, `Evaluation`, or `Kernels`. |
| 18,000-second timeout cap | Drop. Long research is valid; use wall-clock timeout formula and contributor confirmation. |
| Terminal-Bench instruction suffix | Drop; RSI-Harness appends submission guidance. |
| GPU type allowlist | Drop and omit `gpu_types`; caller/scheduler chooses physical GPUs. |
| `allow_internet` true/false | Keep omission; use current `network_mode`/`allowed_hosts`. |
| slug length | Keep the at-most-three-token naming convention. |
| package name | Adapt to exact `rsi/<task-directory>`. |
| separate verifier | Invert: forbid it; RSI-Harness uses shared Base/Judge and injects the task-owned tests only into Judge. |
| verifier tooling baked | Adapt: bake it into Environment, not tests image. |
| verifier network fetch | Keep prohibition; Verifier is no-network. |
| pip pinning | Keep exact pins with requirements/local/revisioned-VCS exceptions. |
| pytest pins | Use `pytest==9.1.1` when pytest is used; `pytest-json-ctrf==0.5.2` only if actually used. |
| bare `nproc` | Keep prohibition. |
| Compose host binds | Keep and strengthen: no task-authored volumes at all; allow only Harness's exact NVIDIA reservation structure under `deploy`. |

## Harness-specific rejection list

Also reject:

- schemas other than 1.4, Windows, missing GPU Work requirement;
- multi-step tasks/rewards, sidecars, unsupported Compose keys or contexts;
- task volumes, artifacts, collect hooks, solution env;
- independent Verifier environment/image or `tests/Dockerfile`;
- MCP, skills_dir, healthchecks, TPU;
- multiple or explicit GPU types for this task program;
- wildcard runtime allowlists when exact provider/data hosts can be named;
- Verifier reward written early, legacy `reward.txt`, runtime fetch/install, or task-authored intermediates;
- distributed Ray/vLLM without runtime Judge IPv4 derivation, with a hard-coded
  `VLLM_HOST_IP`, with a `0.0.0.0` fallback, or with address/bootstrap failure
  incorrectly converted into a candidate score.
- `lm-evaluation-harness` document completeness inferred from logged-sample row
  count, or multi-filter samples consumed without explicit `doc_id` grouping
  and filter selection.
- obvious fake image references and any literal `/tests/...` dependency absent from the task's own `tests/` tree.

Do not require an image digest. Prefer a Dockerfile and accept real stable tags; when a contributor supplies a digest, preserve and verify it rather than synthesizing one. A prebuilt-image path is an explicit contributor decision, not a fallback for unavailable execution checks.

Do not reject a read-only path reference merely because it contains `/tmp`. Reject actual task-authored result, feedback, or control-flow writes. Unavoidable library/runtime scratch outside a read-only WORKDIR still requires manual review and must not persist task results, protected data, or cross-round state.

## Timeout review

Do not warn merely because a timeout exceeds 18,000 seconds.

Check instead:

```text
Verifier timeout >= worst complete scoreable test.sh run + headroom

Agent timeout >= all Work research/training
                 + max_submissions × worst blocking Judge duration
                 + operational margin
```

`agent.timeout_sec` is wall-clock. It continues while Work is paused and the synchronous `rsi-submit` waits for Judge. `verifier.timeout_sec` starts only for `/tests/test.sh`; snapshot/creation/injection/cleanup happen outside it. A 12-hour Work training phase followed by a 12-hour Judge evaluation consumes roughly 24 hours of Agent budget for that cycle.

If proposal compute exceeds its normal planning reference, preserve the contributor's estimate and flag resource review; do not silently cap the task.
