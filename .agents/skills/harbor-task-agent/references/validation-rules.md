# Validation rules

Validation has four layers. Passing an earlier layer never proves a later one.

## Layer A — Proposal-to-task semantics

Review manually before and after generation:

- repository URL and exact SHA match the approved proposal;
- baseline, workload, metric, direction, units, aggregation, and budget did not change;
- public/hidden feedback boundary and leakage controls are implemented as confirmed;
- starting artifacts, editable scope, prohibited actions, and deliverable align;
- Work/Judge resources, network/data policy, proxy needs, timeouts, and submissions match the final assumption review;
- the effective WORKDIR and snapshot mode match the image, candidate is fully materialized on disk, and split-WORKDIR evaluation succeeds read-only;
- complete Judge stdout/stderr and Harness footer visibility match the contributor-confirmed feedback contract;
- every evaluator helper/input referenced under `/tests/...` is present in the generated task, with no undeclared external evaluator bundle or runner dependency;
- no measured result, variance, runtime, or successful verification was invented.
- evaluator control flow, not merely its documentation, implements every
  declared terminal outcome: valid baseline score, candidate correctness-fail
  score when defined, successful final reward, and no reward for
  infrastructure/incomplete failure.

The included validator cannot prove this mapping because it does not reinterpret scientific intent.

Reread the evaluator from entrypoint to reward write for this layer. A clean
static report is not evidence that an exception path implements the confirmed
score semantics.

## Layer B — Included static validator

Run:

```bash
python3 <skill-dir>/scripts/validate_task.py /absolute/path/to/task
```

It is read-only and uses the Python standard library. It checks required files and metadata, task identity/taxonomy, language-appropriate text canaries, absolute/effective WORKDIR declarations, major RSI-Harness unsupported fields, the supported Compose subset, obvious placeholder image references, dependency pins, absolute instruction paths, missing task-owned `/tests/...` assets, Verifier network/install rules, common direct Python/shell intermediate writes, direct exclusive reward creation, README run options and a necessary timeout lower bound, and selected solution/test reference alignment.

`ERROR` must be fixed. A `WARNING` needs evidence-backed review; warnings do not automatically make a task invalid. Static pattern and AST checks are conservative and cannot prove Docker buildability, full reward control flow, evaluator correctness, anti-cheat security, statistical validity, or successful GPU execution.

Every UTF-8 task text file must use a comment-capable format so the canary can
remain a real comment. Do not author commentless JSON text assets; use a
comment-capable source format or a genuinely binary asset and parse it
accordingly. `docker-compose.yml` is invalid because RSI-Harness recognizes only
`environment/docker-compose.yaml`.

## Layer C — Authoritative RSI-Harness compiler

When a Harness checkout is available:

```bash
python3 <skill-dir>/scripts/validate_task.py /absolute/path/to/task \
  --harness-root /absolute/path/to/RSI-Harness
```

The validator invokes `HarborTaskCompiler.compile(...)` with the reward, direction, and maximum-submission settings parsed from README. Compilation reads and hashes the task but does not build Docker, allocate GPUs, use network, run solution/tests, or mutate the task.

Do not replace the static layer with compilation. Harbor's parser may accept unknown metadata, and the compiler intentionally does not lint canaries, dependency pins, Dockerfile hygiene, reward lifecycle, or proposal semantics.

## Layer D — Execution checks

These are separate, stateful, and potentially expensive. Run only when authorized and report each independently:

1. Docker/Compose image build and image preflight.
2. Baseline/no-op submission: valid continuous baseline score, safe feedback, no leaked cases.
3. Optional external/manual baseline Solution helper: traceable smoke/baseline behavior, never presumed full score or Harness-executed.
4. Negative controls: prohibited edits, evaluator tampering, hard-coded cases, fabricated outputs, dependency/path changes, missing/incomplete evaluation.
5. Repeated deterministic/reliability runs and variance characterization.
6. Real GPU full evaluation within Verifier timeout and resource limits.
7. Real multi-round Agent run with submission budget, timeout, GPU release/reuse, feedback, and retained workspace.

Do not label a task “runtime-verified” solely because it compiled. A complete self-contained package may be reported as “statically valid and Harness-compilable; execution checks pending.” Missing Docker/build definitions, evaluator assets, or real runtime commands make the package incomplete rather than a valid compile-only fixture.

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
- Verifier reward written early, legacy `reward.txt`, runtime fetch/install, or task-authored intermediates.
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
