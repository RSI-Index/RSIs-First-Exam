# Existing Harbor Task Refinement Mode

## Purpose

Extend `harbor-task-agent` with a second input mode for refining an existing
Harbor task that has no approved AutoResearch proposal. The skill must recover
enough task semantics from the package and its cited sources to rebuild it to
current RSI-Harness standards without requiring the user to author or complete
a proposal.

This is not a lint-only task review. The existing task is evidence and reusable
material, not an authoritative specification.

## Mode selection

The skill accepts two entry modes:

1. **Proposal mode:** an approved proposal is the semantic source of truth. The
   current proposal-to-task workflow remains unchanged.
2. **Existing-task refinement mode:** an existing Harbor task is supplied with
   no approved proposal. The skill recovers an internal task brief and uses it
   in place of the proposal for the remainder of the workflow.

If both are supplied, the approved proposal controls task semantics and the
existing task is only an implementation reference.

Refinement mode does not call `proposal-agent`, create `proposal.md`, or expose
a proposal schema to the user.

## Semantic recovery

The Agent completely inspects the existing task, including `instruction.md`,
`task.toml`, README, Environment, Solution, Verifier, evaluator assets, and all
referenced commands and paths. It then researches the official repository,
immutable ref, released artifacts, documentation, model cards, papers, and
author-published results needed to verify the intended experiment.

The Agent maintains an internal recovered-task brief covering:

- iterative research question and candidate-owned deliverable;
- reference baseline, baseline materialization, reported result, and evidence;
- fixed evaluation protocol, scalar reward, direction, aggregation, and visible
  feedback;
- starting state, editable scope, prohibited actions, and candidate-failure
  behavior;
- data, network, resource, timeout, and submission assumptions; and
- Environment-to-Verifier interface and required task-owned assets.

The brief is internal working state, not a generated artifact.

## Evidence and conflict rules

RSI-Harness is the hard runtime contract. No individual file in the source task
is automatically authoritative about research intent.

The Agent uses the actual evaluator and Solution to determine current behavior,
the instruction and README to recover intended behavior, and official sources
to verify repository facts, baselines, metrics, data, and artifacts. It records
contradictions explicitly and distinguishes an implementation defect from a
deliberate task choice before changing behavior.

When evidence is incomplete, the Agent makes conservative implementation
decisions and labels unverified claims honestly. It never invents measured
baseline results, runtime, variance, provenance, hidden evaluation behavior, or
scientific intent.

The Agent stops instead of claiming a standards-compliant refinement when it
cannot reliably establish any core semantic requirement: the iterative
AutoResearch objective, a traceable baseline, a fixed evaluation and scalar
reward, the candidate deliverable/action boundary, or the source identity
needed to build the task. Ordinary packaging and implementation decisions do
not require user input.

## Interaction and generation

Refinement mode does not conduct proposal rounds or ask the user to fill missing
fields. Before writing, it presents one compact review of the recovered task
semantics, material inferences, and planned behavioral corrections. The user
confirms or rejects that review but is not expected to supply proposal content.

After confirmation, the Agent generates a new sibling directory named
`<source-slug>-refined` by default. It never overwrites the source task unless
the user explicitly authorizes that exact destination. Existing files and
assets may be reused only after review; each is preserved, rewritten, or
discarded according to the recovered task brief and current skill rules.

The result must be a complete, self-contained RSI-Harness Harbor task, including
a required `solution/solve.sh` that idempotently materializes the reference
baseline without evaluating it. Instruction remains concise and Agent-facing;
README contains the operational detail, provenance, validation status, and
known limitations useful to a human operator.

## Workflow integration

Add input-mode selection before the current Stage 1:

- proposal mode continues through the existing proposal preflight;
- refinement mode replaces proposal preflight with existing-task inventory,
  repository research, semantic recovery, and blocking-gap analysis.

Both modes then converge on the existing Environment, Verifier, operational
configuration, final assumption review, generation, validation, independent
review, and post-handoff execution stages. References and checklists should use
“approved semantic source” where a rule applies equally to an approved proposal
or a contributor-confirmed recovered-task brief.

The independent reviewer receives the original task, recovered-task brief,
confirmation summary, refined task, repository evidence, and validation
results. It verifies both current RSI-Harness compliance and faithful recovery
of the intended AutoResearch experiment. The existing limit of one initial
review, at most one targeted re-review, and no third automatic review remains
unchanged.

## Validation and execution boundary

The refined task follows the same validation layers as proposal mode:

1. static task validation;
2. authoritative RSI-Harness compilation and one Generator final review;
3. bounded independent review;
4. optional Environment preflight; and
5. optional scientific and end-to-end execution.

Environment preflight, Docker build, GPU evaluation, baseline execution, and a
real Agent run are never started automatically. After handoff, the Agent first
explains the required network mode, Docker disk, images/containers, GPUs, time,
and cleanup impact and waits for explicit authorization.

## Skill verification

Verify the new mode with a representative existing task that has no proposal
and contains intentional contradictions across its instruction, baseline
Solution, and Verifier. The forward test must demonstrate that the Agent:

- enters refinement mode without invoking proposal-agent;
- researches and recovers an internal task brief;
- asks for only the single final confirmation;
- leaves the source directory unchanged;
- generates a new `-refined` task;
- resolves the seeded contradictions according to evidence;
- passes the static validator and RSI-Harness compiler when available; and
- completes the normal bounded independent-review workflow.

Generated forward-test tasks remain temporary and are not committed unless the
user explicitly requests an example artifact.
