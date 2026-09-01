# Work, submission, and Judge design

RSI-Harness owns the lifecycle:

```text
Work/Agent -> flushed workspace -> rsi-submit -> immutable snapshot
           -> independent Judge phase -> /tests/test.sh -> finite reward
```

The task defines candidate and evaluator semantics only. It never launches,
selects, or communicates with physical Judge nodes.

## Baseline Solution

`solution/solve.sh` idempotently restores or materializes the approved baseline
candidate inside WORKDIR. RSI-Harness does not execute it automatically. It
does not train, evaluate, invoke `rsi-submit`, read `/tests`, or write reward.

State baseline evidence truthfully as reported, reproduced, not reported, or
protocol mismatch.

## Submission contract

Instruction and README define the complete on-disk candidate. Before
`rsi-submit`, Work must close files, synchronize ranks, flush checkpoints and
metadata, and stop processes that own mutable state. Processes, sockets, CUDA
state, and unflushed buffers do not cross the snapshot boundary.

The Agent calls ordinary `rsi-submit`; no task-specific multi-node CLI exists.
Work never reads scheduler state or tells Harness where Judge runs.

## Distributed checkpoint contract

When Judge loads distributed Work artifacts, freeze:

- source/config/attempt identity and exact training-update coordinates;
- expected shard count and metadata/tracker files;
- checkpoint completeness and digest/inventory rules;
- optimizer/model state requirements;
- Work and Judge world sizes;
- exact scoring and replay-start checkpoints; and
- supported rank-count-independent load/reshard path.

Judge reconstructs candidate source from the immutable snapshot, validates
lineage, and performs a real reload. Endpoint model reload does not prove
optimizer serialization or intermediate trajectory measurements; test those
separately when reward depends on them.

## Terminal verifier

`tests/test.sh` is the only scoring entry point. It runs offline under the
declared Judge GPU and timeout contract and writes
`/logs/verifier/reward.json` exactly once, only after complete success, with
one finite `reward` field.

Keep every reward operand in trusted Judge state or independently recompute it.
Candidate summaries and candidate-produced metrics are diagnostics, not reward
authority. Candidate-invalid behavior uses only the proposal-approved finite
scalar. Infrastructure, scheduler/container, asset, timeout, crash, incomplete
checkpoint, missing submission, and ambiguous paths write no reward.

Stdout/stderr is Agent-visible feedback. Emit bounded safe aggregates and stable
candidate-owned diagnostics; withhold hidden examples, answers, credentials,
per-case results, raw unsafe child streams, and evaluator internals.

## Conditional external baseline authority

Use this branch only when reward requires baseline measurements that cannot be
independently recomputed within every normal Judge run.

1. Provide a separate operator-only baseline-certification submission type.
2. Require unchanged baseline source and one complete Work-to-Judge attempt.
3. The certification run emits no candidate reward; Judge creates a certificate
   bound to protocol, task/source/image/profile, Work/Judge world sizes,
   checkpoint lineage, evaluator metrics, and evidence digest.
4. Store accepted authority outside the task and mount it read-only to Judge.
5. Run the normal certified baseline afterward and require a finite reward.

An interrupted, resumed, or stitched certification cannot create authority.
Resume may be supported for ordinary research or a dedicated resume smoke, but
selected certification evidence has one fresh run root, one allocation
identity, and `resume_count = 0`.

## Completion evidence

Work success is not task acceptance. End-to-end validation requires a staged
submission, immutable snapshot, Judge launch on its planned pool, complete
reload/evaluation, finite reward, durable Harness artifacts, and clean teardown.
