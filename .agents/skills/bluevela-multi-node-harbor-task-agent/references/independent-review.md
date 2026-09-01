# Bounded independent review

The reviewer is read-only. Give it the approved proposal, contributor
confirmation, immutable evidence, selected Harness/profile identity, generated
task, static/compiler/preflight output, and Generator findings. Handoff claims
are context, not proof.

## Review questions

Verify that the package:

- preserves the approved science, baseline, action space, evaluation, reward,
  data/network boundary, and candidate-invalid behavior;
- uses schema 1.4 Work/Judge metadata and the ordinary RSI-Harness lifecycle;
- contains no task-owned cluster/LSF/Apptainer/hostname/site-path logic;
- leaves GPUs-per-node, node count, queue/group, exclusivity, binds, placement,
  host exclusions, and storage roots to the selected profile;
- preserves single-node compatibility and resolves valid whole-node arithmetic
  for the selected multi-node profile;
- declares every external logical asset separately for Work and Judge and
  documents preparation without embedding site paths;
- budgets per-node CPU/memory, shared workspace, build, Agent, and Judge time;
- freezes a complete starting state without bytecode/cache contamination;
- defines complete durable checkpoint/snapshot/submission/Judge reload
  contracts when applicable;
- traces each reward operand to Judge authority or independent recomputation;
- prevents Work-only or shortened-smoke evidence from becoming certification;
- keeps allocation fallback changes run-private, preserves the declared task
  resources unless the user explicitly changes scope, and prevents
  interrupted/resumed evidence from being stitched into a clean baseline
  certificate; and
- exposes only safe aggregate feedback.

Classify findings as `BLOCKER`, `MAJOR`, or `MINOR` with exact evidence.

## Bound

Each review cycle has one initial review, one generator-owned consolidated
correction, and one targeted re-review. A remaining material finding prevents
live execution and begins a fresh evidence/repair cycle. It becomes `BLOCKED`
only when resolving it needs a scientific decision or authority outside the
approved proposal; an internally repairable finding is not a terminal state.

After static review passes, the same unified skill may enter live validation,
but the review itself never runs cluster commands.
