# End-to-end Blue Vela execution validation

Use this stage immediately after generating and statically reviewing the task
from its approved proposal. Treat task bytes, image contents, cluster nodes,
checkpoints, and candidate output as untrusted until their gate passes.

## Approved workflow envelope

Run the bundled preflight read-only:

```bash
PYTHONDONTWRITEBYTECODE=1 <harness-python> \
  <skill-dir>/scripts/preflight_task.py /absolute/task \
  --harness-root /absolute/RSI-Harness \
  --cluster bluevela --json
```

Record its exact JSON plus credentials, data preparation, expected duration,
and retained-storage additions it cannot infer. The already-approved proposal
and workflow request cover:

- the named task/Harness/profile digests;
- build and run resource plans;
- the selected queue/group/exclusivity and site roots;
- data preparation, SIF build/cache, and final allocation;
- task-local or Harness-local repairs and fresh retries that remain inside the
  envelope; and
- cleanup only of exact resources created by this validation.

Do not add a second approval pause between this preflight and live execution.
Pause only when the envelope reveals missing authority outside the approved
proposal.

## Fixed gate order

```text
STATIC_READY
  -> PROFILE_READY
  -> IMAGE_READY
  -> ALLOCATION_READY
  -> WORK_READY
  -> SUBMISSION_READY
  -> JUDGE_READY
  -> END_TO_END_VALIDATED
```

Do not infer a later gate from an earlier one.

### STATIC_READY

Require clean bundled static validation, authoritative Harness compilation, and
the selected profile's exact resource plan. Run the Harness Blue Vela
single-node and multi-node regression suites before changing adapter behavior.
The task source and canonical profile digests become immutable evidence inputs.

### PROFILE_READY

Start with the ordinary run command and canonical profile selected by
`--cluster bluevela`. Do not pass local-only `--gpus`. Confirm:

- profile group/queue belong to the user and are not guessed from a GPU model;
- whole-node Work/Judge arithmetic and disjoint multi-node pools;
- per-node CPU, memory, local tmp, total shared storage, and walltime;
- phase-scoped binds cover every declared asset;
- long validation uses an authorized production policy that can survive its
  duration; and
- shared profile bytes are unchanged from the approved digest.

The default profile must select queue `normal` with exclusive placement (`-x`).
When recorded scheduler evidence shows that this allocation remains
unavailable, copy the canonical profile to the exact run's private state,
change only its queue to `priority` and exclusivity to false, and preflight it
with `--scheduler-policy allocation-fallback --allocation-evidence <evidence>`.
Never edit the task or canonical profile to carry this fallback. A complete
fallback run has the same acceptance criteria and needs no duplicate normal
run.

Allocation evidence is either:

- the same full-size normal job in a pending state in at least two timestamped
  scheduler observations separated by the declared wait window; or
- that job terminating with a scheduler-reported allocation-specific reason.

Record job ID, timestamps, status, and scheduler reason. Before fallback
submission, cancel only that exact normal job if it remains pending and confirm
it is no longer active. Never leave normal and fallback submissions able to
allocate concurrently.

Pass `--allocation-evidence` one JSON object with `normal_job_id` and either
`declared_wait_seconds` plus two or more `observations`, or one `terminal`
object. Each observation carries `observed_at`, `status = "PEND"`, and
`reason`; a terminal object additionally carries `allocation_specific = true`.

After a demonstrated bad node, a copied run-specific profile may additionally
exclude only that exact host for a new attempt; preserve both attempts and keep
the canonical profile unchanged.

### IMAGE_READY

Let Harness use its compute-node build/cache path. Require a content-addressed
SIF, matching digest on every allocated node, sufficient compute-local build
space, and no bytecode/cache-derived accidental image key. Do not build on the
login node or modify cached SIFs/sidecars.

### ALLOCATION_READY

Keep the invoking Harness controller alive. Record exact run/build job IDs,
run/log roots, SIF digest, profile digest, ordered hosts/IPs, GPU selectors, and
Work/Judge pool digest. Node probes must validate every allocated node: GPUs,
GPFS, SIF, required InfiniBand, and local tmp. A node probe failure gets bounded
retries, then a fresh allocation; it is not ignored.

Require workspace seed completion before Agent start. Seed the full project
tree with a bounded, observable method suitable for shared storage; compare
file count/bytes or a recorded inventory. A partial or timed-out seed is an
infrastructure failure.

### WORK_READY

Use the exact Work/Judge GPU totals declared by `task.toml` under either
scheduler policy. Confirm the declared distributed world size, all ranks,
collective path, data readiness, stable progress, and expected throughput.
Queue pressure never authorizes reduction. A smaller run requires an explicit
user instruction naming its scope and the resulting evidence claims only that
scope unless the approved task contract itself is changed.

For checkpointed workloads, inspect every retained checkpoint for its complete
rank shard count, metadata, tracker, total bytes, and barrier completion. Work
must reach its declared terminal state. A shortened run proves only the exact
bounded plumbing it exercised and never certifies a longer scientific
workload.

Long Agent commands stay in one shell session. Poll that session at long
intervals and avoid textual commentary between polls so orchestration does not
consume the Agent's context/token budget.

### SUBMISSION_READY

Work completion alone is insufficient. Require:

- the expected submission manifest exists and passes task validation;
- `rsi-submit` is called through native Harness;
- the immutable snapshot completes;
- submission/lease state reaches Judge dispatch; and
- source, config, task, image, profile, allocation, checkpoint, and snapshot
  identities remain linked.

### JUDGE_READY

Require Judge to run on the planned independent pool, with its phase assets and
read-only authority. Verify complete checkpoint reload/reshard across the
declared Work/Judge world sizes, optimizer/state serialization when relevant,
the full evaluator, safe feedback, and clean timeout/error classification.

Judge success requires nonempty native Harness feedback/report artifacts. A
single Judge log line, process start, or endpoint load is not sufficient.

### END_TO_END_VALIDATED

Require all of:

- login-side Harness command exits zero with completed status;
- owned LSF jobs finish successfully;
- `RUN_INFO.json` and `final_result.json` agree on the immutable run;
- at least one completed submission has nonempty Agent, feedback, and report
  artifacts;
- `/logs/verifier/reward.json` contains the declared finite reward;
- Work/Judge selectors and pools match the resource plan; and
- remote processes, GPUs, temporary state, and exact-run cleanup are complete.

These requirements are identical for `normal + -x` and the recorded
`priority` allocation fallback without `-x`. Success under either policy is
final.

## Conditional baseline certification

When the task declares external live baseline authority, end-to-end acceptance
has two distinct runs:

1. one fresh, uninterrupted certification attempt produces Judge-owned
   evidence and a rewardless certificate;
2. after validating and installing that certificate in the profile-owned
   authority root, one normal certified-baseline run produces a finite reward.

Certification evidence binds one run root, allocation, task/source/SIF/profile
digest tuple, Work/Judge topology, checkpoint lineage, evaluator outputs, and
`resume_count = 0`. An interrupted attempt cannot be resumed, copied, or
stitched into authority. Ordinary resume smoke remains diagnostic.

Tasks without this authority branch need only their normal scoreable baseline
run.

## Repair and retry

For a failure, read the exact boundary logs and state one evidence-backed
hypothesis. If evidence demonstrates a source defect, add a focused regression,
make the smallest source-controlled repair, and retry with a new run ID. If it
demonstrates a transient scheduler, node, storage, or other infrastructure
fault, preserve the evidence and retry with a fresh allocation without making
a fictitious source change. Frozen task/run/image artifacts are never patched
in place.

Every final retry starts from the frozen generated task state. Do not seed it
with checkpoints, workspace output, snapshots, or partial submissions from an
earlier run. This no-stitching rule applies to ordinary scoreable tasks as well
as external baseline certification.

Repairs to generic Harness behavior must keep both single-node and multi-node
tests green. Task repairs cannot change science. A changed task, Harness, or
canonical profile digest invalidates downstream live evidence and restarts from
the earliest affected gate.

Continue through failed attempts rather than returning terminal `FAIL`. Pause
only for new credentials, quota/permissions, external data, larger or different
resources, new network destinations, destructive cleanup, or a
scientific-contract decision; report that state as `BLOCKED`.
