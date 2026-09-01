# Portable Environment and data design

The task ships a Dockerfile-based RSI-Harness Environment. Blue Vela's adapter
turns that build context into a content-addressed SIF and supplies all
site-owned binds. The task never ships an Apptainer definition or build script.

## Base and WORKDIR

- Use schema 1.4 and one shared Base/Work/Judge image.
- Freeze an existing absolute WORKDIR, normally `/workspace`; keep it separate
  from Engine mounts such as `/tests`, `/logs/verifier`, and
  `/run/rsi-harness/staging`.
- Pin source and non-system dependencies by immutable evidence. Do not bake the
  task's tests, solution, credentials, hidden answers, or external authority
  into the image.
- Finish every install/build/generation step that mutates WORKDIR before
  recording the pristine starting state.
- Remove incidental `__pycache__`, `*.pyc`, partial downloads, build caches,
  logs, and test outputs. The build context digest must change only for
  intentional source changes.

Default to root Work/Agent identity unless the workload requires another user.
Judge authority comes from task-owned `/tests` and profile-owned read-only
mounts, never filesystem ownership inside the shared image.

## Large data and preparation

Large datasets, checkpoints, model caches, and baseline authority live outside
the task and SIF. Task code consumes logical container paths, preferably under
`/rsi-data`. The selected profile maps its own source roots to those paths.

For every external asset, add a phase-scoped
`[[metadata.rsi_harness.assets]]` declaration with:

- exact logical path;
- `phase = "work"` or `"judge"`;
- file/directory kind; and
- a real minimum size or entry-count readiness predicate.

Declare the same path twice when both phases require it. Do not assume Work
binds are visible to Judge.

When raw data needs preprocessing, provide a portable, idempotent preparation
command in the image. It takes only logical input/output roots and writes an
atomic READY manifest after validating complete outputs. The README states that
an operator runs it once with a writable site-owned mount before normal task
runs. Normal Work/Judge binds remain read-only.

Preparation is not hidden inside Agent startup. A missing or partial asset must
fail before allocation or before training, not after hours of GPU work.

## Multi-node runtime boundary

Task launchers may invoke framework-native distributed commands such as
`torchrun`, but they consume environment supplied by Harness. They do not:

- read LSF variables or discover hosts;
- choose rendezvous host/IP, node rank, or physical GPU indices from site data;
- invoke remote shells, LSF, or Apptainer; or
- assume a GPU model or GPUs-per-node.

Harness maps one logical Work command across the frozen Work pool and later
runs Judge on its separate pool. Task code may validate `WORLD_SIZE`,
`RANK`, and framework-local variables against its scientific contract; it may
not construct the cluster allocation.

## Workspace and checkpoint durability

The mutable workspace and outputs are on Engine-owned shared storage. Before a
long command, estimate:

- source seed bytes/file count;
- build/SIF cache peak;
- retained checkpoints and snapshots;
- Judge reload scratch; and
- final logs/artifacts.

Use atomic files and explicit completion markers. A checkpoint is complete only
after every expected rank shard, metadata file, tracker, and sync barrier is
durable. Keep the Judge-required scoring and replay-start checkpoints; pruning
must be derived from the evaluator contract.

The Agent-facing command should be synchronous and heartbeat-capable. README
operator guidance tells long-polling agents to wait on the same shell session
with large polling intervals instead of emitting commentary or creating new
sessions. This is an execution-efficiency rule, not cluster logic in the task.

## Review

Before Verifier design, confirm the untouched image can create the workspace,
all logical assets are declared by phase, Work outputs are fully on disk before
submission, Judge can reload them read-only, and no task text contains site
paths or deployment policy.
