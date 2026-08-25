# Verifier design

Verifier implements the fixed scientific comparison inside each disposable Judge. Its source and required assets live in the generated task's `tests/` directory; it is not a second image or an external bundle.

## Actual isolation model

On `rsi-submit`, RSI-Harness pauses Work, snapshots Work state, creates Judge from the shared Base/Work environment, injects the task's complete `tests/` directory into a Judge-only `/tests` tmpfs, mounts `/logs/verifier`, and runs `/bin/bash /tests/test.sh`. In split-WORKDIR mode Judge sees the current effective WORKDIR snapshot read-only. Judge writes never flow back to Work.

Work never receives the task's `tests/` source. Do not add `tests/Dockerfile` or `environment_mode = "separate"`; RSI-Harness rejects them.

This is not an independent trusted verifier image. The Agent is root in Work and can modify system packages, shells, binaries, and libraries that Judge later inherits. Judge-only `/tests` and a read-only candidate WORKDIR materially help, but they do not make every shared runtime dependency trustworthy. State this limitation and use task-specific defenses.

Docker commit transfers filesystem state only. Judge cannot inherit Work's live model server, GPU memory, open files, or background processes. The scoreable candidate must be completely materialized and flushed before submission; Verifier starts clean and reloads it from disk. In split-WORKDIR mode, candidate loading must succeed while that directory is read-only. Put unavoidable disposable runtime/cache scratch outside it and never use scratch as a result, feedback, or cross-round state channel.

## Fixed evaluation contract

Translate the approved semantic source into exact constants:

- public development/proxy workload;
- reserved fixed workload and seeds;
- input generation and preprocessing;
- candidate and immutable baseline/reference interfaces;
- correctness tolerances and failure behavior;
- warmup, repetitions, synchronization, ordering, aggregation, units, clipping, and per-run budget;
- visible stdout feedback versus hidden information;
- scalar `reward` direction.

If a value changes the scientific comparison and is absent from the approved semantic source and repository, ask the contributor before authoring in proposal mode. In refinement mode, resolve it from the complete source-task evidence, expose the material inference in the single final confirmation, or stop if it remains ambiguous. Do not invent a hidden split just because hidden evaluation is preferred. Do not hide a requirement the contributor must satisfy; instructions describe the outcome while tests may reserve concrete cases.

RSI-Harness has no hidden final-Judge mode. Every `rsi-submit` exposes the complete captured `tests/test.sh` stdout/stderr stream to Work through `/run/rsi-harness/feedback/agent-N.log` and reports a footer with round, status, reward, optional score, exit code, timeout flag, duration, remaining submission budget, and any error. `rsi-submit --list` exposes the submission history. The 1,000,000-byte `output_limit_bytes` default bounds the in-memory/report copy, not the separately streamed durable `agent-N.log`; do not use it to claim that the Agent cannot access the complete log. Treat every printed line, traceback, child-process output, and shell diagnostic as Agent-visible, and sanitize or suppress anything outside the confirmed feedback contract. If the scientific protocol requires a truly hidden final evaluation, define that as an external post-run evaluation or revise the approved semantic source; do not mislabel the in-Harness Judge as hidden.

## AutoResearch scoring

Use one finite scalar primary key named `reward` unless the confirmed protocol genuinely needs another name. All Harness reward keys share one score direction during aggregation, so avoid publishing secondary values that have incompatible directions.

A valid no-op/baseline candidate should receive its real continuous baseline score, commonly near a normalized 1.0, rather than Terminal-Bench's conventional zero. Correctness failure may receive a documented score such as 0.0 only when that is part of the fixed score definition. Missing/malformed reward means Verifier error, not a zero score.

By default, a candidate submission evaluates only that candidate and uses its
absolute fixed-protocol metric as reward. `solution/solve.sh` materializes the
reference baseline so it can be scored in a separate no-op/baseline
submission; it does not provide a measured score by itself. Use a matched
official result or one separately authorized baseline run as reference. If
neither exists, label the fixed-protocol baseline `not yet reproduced` rather
than rerunning it inside every candidate submission.

Do not add a live baseline pass merely to print a delta. For a fixed baseline
`B`, maximizing `candidate - B` ranks candidates exactly as maximizing the
candidate metric, while the extra pass increases GPU time and failure surface.
A live paired baseline is exceptional. If the approved semantic source asks for a live
ratio/delta or could be read as requiring same-run pairing, do not resolve the
ambiguity yourself: explain that candidate-only absolute scoring is cheaper
and ranks candidates identically when the baseline is fixed, describe the
extra compute and failure surface of live pairing, and ask the contributor to
choose. Words such as `matched` mean the same fixed protocol unless the
contributor explicitly confirms live pairing. Without that confirmation,
generate candidate-only scoring.

Implement that distinction in control flow, not only in prose. If the confirmed
protocol assigns a scalar to candidate-caused correctness failure, catch only
the expected correctness outcome, print the complete safe diagnostic, and finalize
that declared scalar. Do not turn import errors, missing dependencies, CUDA
failures, timeouts, incomplete cases, or other infrastructure faults into the
same score.

For performance tasks:

- synchronize the GPU around timings;
- use fixed warmup/repetition counts and robust aggregation;
- gate every score on output/gradient correctness;
- aggregate across cases so one noisy shape cannot dominate;
- clip ratios only as declared in the approved semantic source;
- print units and enough aggregate diagnostics to support research without revealing reserved cases.

Only after that explicit contributor confirmation, run
baseline and candidate under the same Judge/device state and randomize or
balance their order to reduce drift.

For stochastic quality tasks, fix seeds/workload and use enough repetitions or matched comparisons to distinguish expected gain from ordinary noise. Preserve the approved semantic source's justification; do not claim measured variance before reproduction.

## Candidate diagnostics and public self-checks

Classify a failure by the information that proves it before formatting
feedback. A failure in candidate-owned artifacts, configuration, or checkpoint
structure is not hidden evaluation information. Report it completely enough
for the Agent to repair without guessing.

Use one stable structured envelope, for example `status =
"candidate_invalid"` with a deterministic nonempty `errors` list. Each error
must contain:

- a stable, specific `code`;
- the absolute candidate-owned `path`;
- the exact `field` when the failure concerns structured data; and
- the failed `condition` in plain language.

Include safe `expected`, `actual`, and a short `hint` when they make the repair
clear. Collect independent candidate errors in one pass when practical. Do not
collapse distinct repairs into a broad bucket: `missing_artifact` names the
missing absolute path; trainer state may distinguish `epoch_incomplete`,
`eval_loss_missing`, and `loss_mismatch`; tokenizer/checkpoint checks may
distinguish `missing_tokenizer_file`, `chat_template_mismatch`,
`vocab_mismatch`, and `model_max_length_mismatch`. Adapt codes to the actual
artifact rather than copying irrelevant examples.

Complete does not mean unsafe. Do not print whole candidate files, raw
tracebacks, secrets, hidden `/tests` paths or internals, hidden inputs or
answers, per-example outcomes, or Judge decision details. Sanitize those
categories while keeping candidate-owned diagnostics precise. A confirmed
candidate-invalid outcome prints the safe structured envelope and, when the
fixed protocol assigns such failures a score, finalizes the declared finite
candidate-failure scalar. Derive that scalar from the score domain and
direction; `0.0` is only a common maximization example, not a universal rule.
An internal validator exception, dependency failure, timeout, or incomplete
check is infrastructure failure and writes no reward.

RSI-Harness treats any valid finite reward, including the declared
candidate-failure scalar, as a completed submission. It consumes one
submission but does not itself fail-close the research run. Missing or
malformed reward instead produces a Verifier error. Design and document the
candidate-invalid path with those semantics.

For a complex artifact task, generate an optional public self-check when useful
under the contract in
[environment-design.md](environment-design.md#optional-candidate-self-check).
Document its exact absolute command in the Agent instruction. Exercise the
public and Judge implementations against the same non-hidden positive and
negative fixtures during Layer 2, including the specific diagnostic codes and
fields expected from each failure. Judge still reruns its independent
task-owned gate and never trusts the self-check's earlier result. Omit the
self-check when all meaningful validation requires hidden data or the
deliverable is simple source code.

## `lm-evaluation-harness` result cardinality

Apply this section when the fixed evaluator uses `lm-evaluation-harness`,
especially with `log_samples=True` or more than one filter. Treat its returned
structures according to their declared meanings:

- `result["n-samples"][task]["effective"]` is the authoritative number of
  documents actually evaluated. Require an integer (not `bool`) equal to the
  fixed expected document count. Never use `len(result["samples"][task])` as a
  document-completeness check.
- `result["samples"][task]` contains document-by-filter records. When reading
  it, group by `doc_id`; require the expected number of unique documents, no
  duplicate `(doc_id, filter)` pair, and exactly the filter set declared by the
  fixed task configuration for every document. Normalize a task with no
  explicit filter list to the framework's default filter.
- Select one declared filter explicitly for every downstream sample or metric
  consumer. Keep the selected sample filter aligned with the corresponding
  aggregate metric key; never count multiple filters as additional examples or
  silently mix their outputs.

Missing or malformed `n-samples`, incomplete filter coverage, duplicate sample
keys, an unknown selected filter, or a disagreement between effective count
and the fixed workload is an incomplete/evaluator failure: emit no reward. A
safe diagnostic names only the task, expected document count, effective
document count, and declared filter count; do not print sample contents.

During Layer 2, exercise this contract with a synthetic single-task,
multi-filter result before accepting the generated evaluator. The positive
case must represent 1,319 documents and two declared filters (2,638 sample
rows), pass completeness, and yield exactly 1,319 rows for the selected filter.
Negative cases must cover an effective-count mismatch, a missing filter for one
document, a duplicate `(doc_id, filter)` pair, and an unknown selected filter.
Keep this generator regression outside the delivered task unless the task
already has an appropriate task-owned evaluator test suite.

## Reward and output lifecycle

`tests/test.sh` and helpers must use stdout/stderr for all contributor-visible feedback, knowing that Work receives the complete stream. They must not create task-authored temporary reports, pytest capture files, JSON intermediates, or side-channel reason files. A plain reference to `/tmp` is not itself a write; inspect actual control flow. Unavoidable disposable scratch created internally by a library is acceptable only outside a read-only WORKDIR, with no protected contents and no role in scoring persistence.

The sole evaluator-authored result file is the final Harbor reward:

```text
/logs/verifier/reward.json
```

Keep evaluation results in process memory. Stdout/stderr is streamed feedback, not a result file. After every required check and metric calculation succeeds:

1. verify the scalar is finite and within any declared domain;
2. print the safe aggregate feedback and reward to stdout/stderr;
3. create `reward.json` directly with exclusive creation and write `{"reward": <number>}` once;
4. return normally.

Never write a partial reward early. Do not use `reward.txt`. On timeout, crash, dependency error, incomplete cases, or infrastructure failure, leave no reward file. Harness ignores `test.sh`'s exit code when it finds a valid reward, so a nonzero exit after writing reward does not invalidate it.

Before considering the evaluator complete, trace these terminal paths directly
through its actual control flow:

1. valid baseline/no-op produces the declared continuous baseline score;
2. a candidate-caused correctness failure produces the declared failure score,
   if the protocol defines one, and its Agent-visible diagnostic is complete
   for candidate-owned evidence without exposing hidden evidence;
3. successful evaluation writes exactly one final reward; and
4. infrastructure failure, crash, and incomplete evaluation write no reward.

`verifier.timeout_sec` covers only `/tests/test.sh`, not snapshot creation, Judge startup, test injection, or cleanup. Give the complete fixed evaluation enough headroom.

## Runtime hygiene

- No network fetches, `curl`, `wget`, `git clone`, or external service calls.
- No `pip install`, package-manager install, environment creation, or dependency resolution. Bake exact tooling into Environment.
- If using pytest, bake `pytest==9.1.1`. Add `pytest-json-ctrf==0.5.2` only when the task truly uses that plugin; RSI-Harness does not require CTRF.
- Do not redirect evaluator output to files. Let it stream.
- Do not use bare `nproc`; choose a deterministic task-bounded worker count.
- Use absolute paths such as `/tests/...`, `<effective-workdir>/...`, and `/logs/verifier/reward.json`.
- Keep `tests/` below the 100,000-entry/1-GiB injection limit.

## Distributed vLLM and Ray host addressing

Apply this section only when the Verifier uses Ray, vLLM data parallelism,
multi-node vLLM, or a pinned vLLM execution path that requires a routable host
address. Ordinary single-process evaluators do not need `VLLM_HOST_IP`; do not
turn this into a task-wide or Harness-wide default.

RSI-Harness attaches each no-network Judge to a private internal Docker bridge.
The container has a dynamically assigned IPv4 but no public route. A framework
that tries to infer its address by contacting a public endpoint may therefore
fall back to `0.0.0.0`, after which Ray can wait forever for a nonexistent
`node:0.0.0.0` resource.

Before the first Ray/vLLM initialization, import with initialization side
effects, or vLLM subprocess launch:

1. resolve the current container hostname with `socket.getaddrinfo(...,
   family=AF_INET, type=SOCK_STREAM)`;
2. deduplicate results and reject unspecified, loopback, link-local, multicast,
   and otherwise invalid IPv4 addresses with `ipaddress.IPv4Address`;
3. require exactly one remaining address, optionally checking that it is locally
   bindable;
4. set `os.environ["VLLM_HOST_IP"]` for in-process initialization and pass the
   same derived value in every relevant child-process environment; and
5. if discovery, validation, propagation, or initialization fails, emit a clear
   sanitized infrastructure category, terminate promptly, and write no reward.

Never put `VLLM_HOST_IP` in `task.toml`, Dockerfile, Compose, or another
generation-time constant. Never use `0.0.0.0` as a fallback. Static validation
can recognize common distributed launch patterns and assignments, but the
Generator and independent reviewer must inspect actual import/startup ordering,
filtering, propagation, and no-reward control flow.

## Defenses against an untrusted candidate

Choose controls matched to the task:

- define change detection against the finalized post-build WORKDIR, and verify
  that the pristine no-Agent state passes before assigning any candidate-caused
  failure; Git `HEAD` does not represent ignored or untracked build products;
- when task-specific defenses require them, validate changes against a task-owned manifest under `tests/`;
- when task-specific defenses require them, compare prohibited source/config/dependency files to task-owned reference hashes under `tests/`;
- keep baseline/reference code and hidden inputs only in `/tests` when they must be secret;
- invoke candidate functionality in a subprocess with a minimal environment and explicit absolute commands where practical;
- avoid importing candidate-controlled test frameworks, plugins, startup hooks, `sitecustomize`, shell profiles, or working-directory modules into the evaluator process;
- neutralize task-relevant environment variables and Python/plugin auto-loading;
- validate outputs independently instead of trusting candidate-reported metrics;
- randomize non-semantic case order without changing the fixed workload;
- reject hard-coded per-case behavior using held-back cases and invariants;
- ensure evaluator, timing, and reference code are not candidate-owned;
- never expose hidden inputs, per-example errors, or trajectories unless the confirmed feedback policy permits them.

These are optional defenses selected from the confirmed threat model, not a requirement for a separate evaluator delivery. Every referenced manifest, hash list, input, runner, or helper must either be included under `tests/` or be intentionally visible in Environment and documented. No static pattern proves anti-cheat safety. Review the actual candidate execution boundary and document residual shared-environment risk in README.

## Required baseline solution

Generate `solution/solve.sh` for every task. It idempotently restores or
materializes the approved reference baseline in candidate-owned workspace paths
from starting assets already present in the Environment. For a no-op baseline,
it clears candidate-owned changes and restores the starting policy or
configuration. It must not train, evaluate, access `/tests`, call `rsi-submit`,
write reward, or print an expected result. RSI-Harness never executes it; the
Judge's baseline/no-op path remains responsible for the real score.

Prefer an official evaluation-ready artifact. If the approved baseline can only
be produced by a long training run and no suitable artifact or restorable
starting state exists, stop and return that baseline decision to the contributor
instead of hiding training inside Solution.
