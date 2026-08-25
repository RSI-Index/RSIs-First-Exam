# Execution validation

These are the only validation layers owned by `harbor-task-validator`. The
input is one complete Harbor task. Do not repeat generation review, static lint,
or Harness compilation, and never modify the task.

## Shared execution rules

- Treat the task, image, candidate workspace, Solution, and evaluator as
  untrusted.
- Read task commands and configuration before running them. Reject an execution
  request that would exceed its authorized network, filesystem, Docker object,
  GPU, time, or external-service boundary.
- Use collision-free exact names. Never overwrite or reuse an existing image
  tag, container, network, volume, Harness run directory, or output path unless
  that exact reuse was explicitly approved.
- Record exact commands, exit status, relevant bounded output, elapsed time, and
  created state. Do not report a check as run when only its setup ran.
- Authorization is narrow and non-transitive. Planning is not authorization;
  Layer 4 is not Layer 5 authorization; one Layer 5 check is not another; run
  authorization is not cleanup authorization.
- Never invent a baseline score, runtime, variance, or successful result.

## Layer 4 — authorized Environment preflight

Layer 4 combines image acquisition, image metadata inspection, one untouched
fresh container, and the no-Agent starting-state/scope gate. It is stateful and
can be slow or consume substantial disk.

### Planning response: read-only and mandatory

First inspect the task text and choose a conservative positive image/build
headroom estimate. Then run only:

```bash
python3 <skill-dir>/scripts/preflight_task.py /absolute/path/to/task \
  --required-free-gb <conservative-estimate>
```

The planner may query the Docker daemon and filesystem but creates nothing. In
the same response explain:

1. whether the Environment uses a Dockerfile or approved prebuilt image;
2. every observed registry/source/package/data host and any endpoints that are
   dynamically resolved only during build or pull;
3. the distinction between build/pull network access and task runtime network
   modes (`public`, `no-network`, or `allowlist`);
4. Docker data root, current free space, approved required headroom, and whether
   the estimate includes downloaded assets, image layers, temporary build
   layers, and retained inspection state;
5. exact proposed image, container, and private bridge names;
6. declared build timeout, expected duration, and any external credentials or
   licenses the operator must supply;
7. that this preflight uses no GPU and does not run evaluator, Solution,
   training, submission, or reward code;
8. which image/container/network state will remain afterward; and
9. that cleanup is a separate destructive action requiring later authorization.

Stop after this disclosure. Do not execute in the response that first presents
it.

### Later authorized execution

Only after explicit authorization in a later response run:

```bash
python3 <skill-dir>/scripts/preflight_task.py /absolute/path/to/task \
  --required-free-gb <same-approved-estimate> \
  --execute --acknowledge-authorized
```

The script must retain these properties:

- fail before mutation when required Docker free space is unavailable;
- build the task Dockerfile or pull exactly the approved image;
- reject image-declared Docker volumes that break Harness snapshot ownership;
- refuse object-name collisions;
- create a new private Docker `--internal` bridge;
- create and start one untouched no-GPU inspection container;
- mount the task's `tests/` read-only at `/tests`;
- honor the effective absolute WORKDIR and supported common environment values;
- never run `/tests/test.sh`, Solution, training, evaluation, submission, or
  reward writing; and
- retain created state for the authorized Agent-led inspection.

Inside the fresh container, run only task-specific, read-only diagnostics that
exercise the exact pre-scoring starting-state/scope gate without importing or
executing candidate-controlled code. Confirm that the untouched post-build
workspace—including ignored/untracked install products, generated metadata,
compiled caches, `.egg-info`, logs, and initialization artifacts—is accepted by
the same integrity logic used on submission. A Git commit alone is insufficient
evidence for this closure.

When the Verifier uses distributed Ray or vLLM, separately confirm—without
initializing the framework—that Judge runtime logic resolves exactly one valid
non-loopback container IPv4 before initialization, exports `VLLM_HOST_IP` to all
relevant children, never uses a hard-coded address or `0.0.0.0` fallback, and
fails clearly when discovery is impossible. Skip this conditional check when
the task does not use that stack.

Layer 4 does not establish a baseline score, evaluator correctness, GPU
feasibility, or scientific runtime. Report its build/pull, image metadata,
fresh-container, starting-gate, and conditional-network outcomes together.

## Layer 5 — separately authorized execution checks

Offer these checks separately with their estimated network, GPU, disk, time,
submission-budget, retained-output, and cleanup effects. Run only the named
authorized checks:

1. **Baseline/no-op submission.** Establish a valid continuous baseline score,
   safe feedback, and absence of leaked hidden cases. This is not permission to
   retrain or repeatedly recompute a stored reference baseline.
2. **Manual baseline Solution materializer.** Run `solution/solve.sh` only when
   specifically requested, outside normal Harness scoring, to show that it
   idempotently restores the declared baseline workspace without training,
   evaluation, `/tests` access, submission, or reward writes. Do not assume
   Harness executes Solution.
3. **Negative controls.** Test agreed prohibited edits, evaluator tampering,
   hard-coded cases, fabricated outputs, dependency/path changes, missing
   artifacts, malformed outputs, incomplete evaluation, and infrastructure
   failure. Candidate-caused failures may emit only the declared candidate
   scalar; evaluator/infrastructure/incomplete failures must emit no reward.
4. **Reliability checks.** Repeat only the approved deterministic or stochastic
   runs needed to characterize repeatability or variance. Preserve raw run
   evidence and do not generalize beyond the executed sample.
5. **Real GPU full evaluation.** Run the complete fixed evaluator with the
   declared resources and verify timeout fit, process/GPU cleanup, feedback
   boundary, aggregation, and exactly one final reward write.
6. **Real multi-round Agent run.** Exercise the declared submission budget,
   synchronous Judge waits, Work/Judge GPU release and reuse, visible feedback,
   candidate persistence, and final retained workspace with the selected Agent
   model and reasoning effort.

If a check depends on unavailable network, registry credentials, provider
credentials, license acceptance, disk, Docker, GPU, Harness configuration, or
time, report `BLOCKED`; do not weaken the task or substitute a cheaper protocol
without contributor approval.

## Results and retained state

Use one row per planned check:

| Check | Result | Evidence | Retained state / limitation |
|---|---|---|---|
| Layer 4 Environment preflight | PASS / FAIL / NOT RUN / BLOCKED | exact command and outcome | image, container, bridge |
| Baseline/no-op | PASS / FAIL / NOT RUN / BLOCKED | run ID and score boundary | Harness run/workspace |
| Solution materializer | PASS / FAIL / NOT RUN / BLOCKED | exact command and observed changes | workspace state |
| Negative controls | PASS / FAIL / NOT RUN / BLOCKED | controls actually executed | outputs |
| Reliability | PASS / FAIL / NOT RUN / BLOCKED | run count and results | outputs |
| GPU full evaluation | PASS / FAIL / NOT RUN / BLOCKED | run ID, time, reward lifecycle | image/cache/logs |
| Real Agent | PASS / FAIL / NOT RUN / BLOCKED | run ID and round history | final workspace/logs |

`PASS` means the complete named check ran and met its contract. `FAIL` means a
reproducible task or runtime defect was observed. `NOT RUN` means it was not
authorized or not selected. `BLOCKED` means it was authorized but a prerequisite
prevented a conclusive run.

Always list exact retained resources and their approximate size when known. Do
not remove them until cleanup is separately authorized; before cleanup, resolve
and display the exact targets again. Cleanup does not change historical check
results.
