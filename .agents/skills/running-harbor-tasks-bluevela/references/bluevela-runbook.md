# Blue Vela runbook

Use this runbook for an already self-contained RSI-Harness Harbor task. The
packaged Blue Vela profile owns site paths and LSF policy; use a copied profile
TOML for another IBM/LSF cluster instead of editing adapter code.

## 1. Resolve inputs without guessing

Take `TASK` from the request or discover it in the current workspace. Resolve
`HARNESS` from the current Git worktree because this repo-scoped skill keeps
RSI-Harness at the project root. Do not select a sibling checkout:

```bash
TASK=/absolute/path/to/harbor-task
REPO_ROOT="$(git rev-parse --show-toplevel)"
HARNESS="$REPO_ROOT/RSI-Harness"

test -f "$HARNESS/pyproject.toml" || {
  echo "project-local RSI-Harness is unavailable: $HARNESS" >&2
  exit 1
}
```

Require `task.toml`, `instruction.md`, `environment/Dockerfile`, and
`tests/test.sh`. When a task README exists, read it for the primary reward,
score direction, model, reasoning effort, and any deliberately bounded
submission policy. Otherwise take those operator choices from the user or
existing run configuration. Do not invent them. The current Blue Vela adapter
requires a local Docker build context; a prebuilt-image-only task needs an
adapter change before execution.

Perform only lightweight login-node checks:

```bash
cd "$HARNESS"
which bsub
which bjobs
bqueues -u
test -x /proj/datasets/interns/yuetai/rsi-nemotron/apptainer-env/bin/apptainer
gquota ~
gquota /proj/datasets
```

Let the adapter's exact-name collision check protect submission; do not inspect
or operate on unrelated shared-account jobs. Before changing a shared profile
or launcher, inspect only exact job IDs known to use it. Never edit a cached
SIF, checksum sidecar, frozen manifest, or live run; a source/build-context
change must create a new content-addressed image. Never copy credentials into
the task, image, logs, or run artifacts.

## 2. Compile and resolve the final allocation

Use the same scientific options for dry-run and execution. This example assumes
the task contract names `reward` and maximizes it; replace those two values only
when the task says otherwise.

```bash
cd "$HARNESS"

RUN_ARGS=(
  --cluster bluevela
  --agent codex
  --model "${RSI_MODEL:?set RSI_MODEL}"
  --reasoning-effort "${RSI_REASONING_EFFORT:?set RSI_REASONING_EFFORT}"
  --agent-auth local
  --primary-reward reward
  --score-direction maximize
)

uv run rsi-harness run "$TASK" "${RUN_ARGS[@]}" --dry-run
```

The dry-run is a read-only plan, not a smoke job. Confirm it reports:

- Work and Judge GPU counts matching the compiled task;
- total GPUs equal to Work plus Judge;
- one-node `exclusive_process` allocation;
- explicit CPU, memory, local `/tmp`, build walltime, and run walltime;
- unique GPFS run/log paths; and
- either a verified SIF cache hit or one automatic build submission.

The dry-run's hypothetical run ID and paths are not the real run's identity.
Discard them after checking the plan; resolve and monitor the IDs and paths
created by the subsequent non-dry-run command.

Do not add `--gpus`; it is a local selector and is rejected in cluster mode.
For `gpus = "all"`, require a numeric cluster-profile override rather than
guessing. If the requested total exceeds the profile's GPUs per node, stop with
the compiler error instead of silently reducing the task.

Task metadata may describe a required accelerator generation, but the current
packaged profile requests a GPU count rather than an LSF model constraint. If
the scientific contract truly requires a specific GPU model, extend or select
an authorized profile before claiming compatibility.

## 3. Start the production-shaped long run

Remove only `--dry-run` and run in a durable managed terminal session:

```bash
cd "$HARNESS"
uv run rsi-harness run "$TASK" "${RUN_ARGS[@]}"
```

Do not add a separate smoke allocation. On a cache miss the same command first
submits a CPU build using the profile's large build resources, creates the
Docker archive and SIF in compute-node `/tmp`, copies the completed SIF to GPFS,
and verifies its SHA256. It then submits the final GPU job and waits.

Record from the command and `RUN_INFO.json`:

- exact build and run job IDs;
- run ID and run directory;
- frozen source commit/diff;
- SIF path and SHA256;
- Work/Judge/total GPU counts; and
- expected standard artifacts.

The Engine must use the LSF-provided ordered `CUDA_VISIBLE_DEVICES`: first the
declared Work count, then the declared Judge count. Apptainer must receive the
selectors explicitly under containment and run with NVIDIA support. The
compiled Work/Judge counts determine the adapter topology; the runtime
lifecycle then enforces pausing, isolation, and allowed overlap.

## 4. Use the first allocation as hardware evidence

The first real allocation is already the long-running attempt. Use its exact
LSF record and logs to verify the actual host and hardware:

- `bjobs -l <owned-job-id>` identifies the execution host, and the driver's
  full `nvidia-smi` output identifies the assigned GPU model and driver;
- the standard log leaf's `run-plan.json` proves the exact authorized selectors,
  Work/Judge slices, and Judge mode;
- CUDA/NCCL/Ray/vLLM output proves the communication path actually used; and
- Engine events prove native Work, submission service, Judge, and feedback
  flow—not a Harbor CLI run.

If startup fails, follow the debugging reference. A frozen allocation cannot be
patched in place: apply one minimal tested fix under source control, leave the
failed run immutable, and execute the same final command again with a new run
ID. Once an allocation is healthy, do not kill it to start a second “real” job.

## 5. Keep exact ownership boundaries

Monitor only manifest-recorded decimal job IDs:

```bash
bjobs -l "$JOB_ID"
```

Use `bkill "$JOB_ID"` only after confirming the ID, job name, run directory,
and frozen source all belong to the current failed attempt. Never cancel by
partial name, wildcard, queue, or shared account.

Long quiet periods during SIF staging, checkpoint loading, training, or Judge
evaluation are normal. Poll the managed controller and exact logs without
holding a blocking wait longer than the collaboration environment allows.
