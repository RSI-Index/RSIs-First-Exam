# Generation and offline validation

Generation is collision-safe and performs no stateful task execution. It may
parse, validate, and compile task bytes against the project-local Harness.

## Sequence

1. From the approved proposal, verify the destination root is absent,
   create it once, and record its inventory. Before each initial file write,
   verify that exact path is absent.
2. Generate the complete schema-1.4 package. Record modes and content digests.
3. Run one combined static/compiler validation:

   ```bash
   PYTHONDONTWRITEBYTECODE=1 <harness-python> \
     <skill-dir>/scripts/validate_task.py /absolute/task \
     --harness-root /absolute/RSI-Harness --json
   ```

4. Perform the Generator semantic review without editing: scientific contract,
   post-build starting state, logical assets, Work/Judge resources, timeout and
   storage arithmetic, snapshot/checkpoint handoff, reward provenance, failure
   classification, and safe feedback.
5. Obtain one initial independent read-only review.
6. Verify no external inventory/content change, apply one consolidated
   correction, rerun the combined validator once when files changed, and then
   obtain one targeted re-review. No third review.

## Required static outcomes

- schema 1.4 and package name `rsi/<slug>`;
- valid Work and Judge GPU declarations;
- project-local RSI-Harness compiler pass;
- no task-owned `cluster/` or Apptainer definition;
- no LSF/remote-host/site-path logic in task runtime files;
- Dockerfile-based build context and complete pristine starting state;
- phase-scoped readiness declarations for logical external assets;
- executable baseline Solution and terminal Verifier boundaries;
- documented ordinary `--cluster bluevela` command without `--gpus`;
- per-node CPU/memory and Work/Judge timeout semantics documented; and
- finite reward written only after complete Judge success.

Warnings are unresolved work, not optional noise. Fix or explicitly return a
task-defining decision to the contributor.

## Continue after the offline gate

Do not build a SIF, call a scheduler, inspect live assets, run Solution, start
Work, call `rsi-submit`, run Judge, or write reward during generation. The
unified skill moves directly to live preflight after generation/review passes.
The approved proposal and workflow request already define the resource
envelope; do not introduce a generation-only stop.

A static pass never supports wording such as “cluster ready,” “multi-node
works,” or “baseline certified.”
