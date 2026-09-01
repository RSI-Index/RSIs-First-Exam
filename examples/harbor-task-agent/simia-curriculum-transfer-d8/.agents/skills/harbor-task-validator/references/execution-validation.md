# Execution validation

This reference defines the two required execution gates for one complete
RSI-Harness Harbor task. The task may be repaired after the contributor approves
the initial resource envelope; RSI-Harness and unrelated repositories remain
out of scope.

## Shared rules

- Treat the task, image, candidate workspace, Solution, and evaluator as
  untrusted.
- Use collision-free names. Never overwrite or reuse pre-existing Docker
  objects, run directories, or outputs without explicit authorization.
- Record exact commands, exit status, bounded output, elapsed time, reward
  lifecycle, and created state.
- Initial authorization covers both required gates, task-local repairs and
  reruns within the disclosed envelope, and cleanup of superseded resources
  created by this validation run. It does not authorize deleting pre-existing
  or final retained resources.
- Never invent a baseline score, runtime, variance, or successful result.

## Initial planning and authorization

Read the full task before running task code. Calculate peak Docker headroom from
evidence rather than a round guess:

1. inspect the local or registry-reported uncompressed base-image size;
2. total declared model, dataset, checkpoint, source, package, and build-context
   assets from manifests or authoritative remote metadata;
3. include download/extraction duplication, temporary build layers and caches,
   the final image, inspection state, baseline logs, and one repair rebuild when
   plausible; and
4. state the calculation, uncertainty, safety margin, Docker data root, and
   current free space.

Also disclose build/pull hosts and dynamic endpoints, runtime network modes,
Judge GPU count and approved devices, CPU/memory/shm needs, build and verifier
timeouts, expected duration, credentials or licenses, exact proposed resource
names, automatic repair scope, retained state, and validator-owned intermediate
cleanup.

Run only the read-only planner in this response:

```bash
python3 <skill-dir>/scripts/preflight_task.py /absolute/path/to/task \
  --required-free-gb <calculated-peak-headroom>
```

Stop for one explicit approval of the complete envelope. After approval, do not
ask again while actions remain inside it.

## Gate 1: Environment acceptance

Execute the approved preflight:

```bash
python3 <skill-dir>/scripts/preflight_task.py /absolute/path/to/task \
  --required-free-gb <same-approved-headroom> \
  --execute --acknowledge-authorized
```

The preflight must:

- fail before mutation when Docker free space is below the approved headroom;
- build the task Dockerfile or pull exactly the approved image;
- reject image-declared volumes that break Harness snapshot ownership;
- create a private internal bridge and one untouched inspection container
  without requesting GPUs or launching a GPU workload;
- mount task-owned `tests/` read-only and honor the effective absolute WORKDIR,
  user, shared environment, and Compose shm settings; and
- retain the image, container, and bridge for Agent-led inspection.

Inside that fresh container, the validator Agent performs task-specific,
read-only diagnostics. Exercise the public pre-scoring starting-state and
editable-scope gates without running the evaluator or importing
candidate-controlled code. Confirm that the untouched post-build workspace,
dependencies, assets, ignored/untracked install products, generated metadata,
compiled caches, `.egg-info`, and logs are accepted. A Git commit alone is not
starting-state evidence.

GPU device files may remain visible because of an image or host default
runtime. Inherited device visibility is informational, not a Gate-1 failure.
Gate 1 must not request or reserve GPUs, launch GPU processes, or materially
consume GPU memory.

When the verifier uses distributed Ray or vLLM, confirm without initializing it
that runtime code resolves exactly one valid non-loopback Judge IPv4, exports
`VLLM_HOST_IP` before framework startup, and fails clearly instead of falling
back to `0.0.0.0`.

Gate 1 passes only when the image and untouched starting state satisfy every
applicable check.

## Gate 2: Baseline Judge acceptance

Use the Gate-1 image and unchanged baseline workspace. Run the complete fixed
`/bin/bash /tests/test.sh` exactly once under the task's declared Judge
constraints:

- task-owned tests mounted or injected read-only at `/tests`;
- effective WORKDIR and baseline candidate state preserved;
- declared verifier user, environment, network mode, CPU, memory, shm, GPU
  count, and timeout;
- writable disposable scratch/cache paths and an isolated verifier log
  directory; and
- no undeclared network, secrets, host binds, or external services.

Before launch, disclose the exact GPU devices and retained output names already
covered by the initial authorization. Capture complete bounded stdout/stderr and
the process exit status. After completion verify:

- the full declared workload completed within timeout;
- stdout exposes the promised feedback and no hidden examples, answers,
  per-example decisions, or protected evaluator details;
- the declared finite scalar is written exactly once to
  `/logs/verifier/reward.json` or the supported scalar fallback;
- incomplete, infrastructure, timeout, or malformed-reward paths did not get
  misreported as success; and
- Judge processes exited and GPUs were released.

This single run is the baseline, complete evaluator, and GPU execution check.
Do not repeat it under a second “GPU full evaluation” label. It validates the
task-specific side of Harness submissions; the generic `rsi-submit`, snapshot,
and Judge orchestration belongs to RSI-Harness and is not re-tested per task.

## Automatic repair loop

If either gate fails because of a concrete task defect:

1. preserve the exact failure evidence;
2. reproduce the smallest failing behavior and identify its root cause;
3. add an external or task-owned regression test when practical;
4. make the smallest repair inside the task without changing its scientific
   contract;
5. rerun syntax checks and RSI-Harness compilation, plus any available static
   task validator affected by the edit; and
6. rerun the failed gate and every later required gate.

Do not request another confirmation for this loop while network, disk, GPU,
time, mutation, and cleanup stay inside the initial envelope. Superseded Docker
objects created by this validation run may be removed under that approval;
resolve their exact identities first and never delete pre-existing or final
retained resources.

Stop with `BLOCKED` and request direction when a fix would change the scientific
contract, exceed the envelope, require unavailable credentials/licenses, or
modify RSI-Harness or another repository. Use `FAIL` only when a reproducible
task defect remains after safe in-scope repair is exhausted.

## Non-default diagnostics

Do not perform reliability/variance runs or a real Agent run. A Solution
materializer or targeted negative control is diagnostic-only: run it only when
explicitly requested or necessary to understand a required-gate failure. It is
not part of default acceptance and does not appear as `NOT RUN` in the result.

## Final result

Return one status:

- `EXECUTION READY`: both required gates passed after any reported repairs;
- `FAIL`: a reproducible task defect remains; or
- `BLOCKED`: an external prerequisite or decision prevents completion.

Report evidence for Gate 1 and Gate 2, repairs, limitations, and every retained
image, container, network, log, output, and approximate size. Do not list
unselected optional checks. Final retained-resource cleanup remains a separate
explicit operation unless it was included in the initial envelope.
