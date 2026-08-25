# Positional encoding across a 550M-to-2.545B scaling ladder

## Research Question

Can one scale-general positional-information or attention-scope mechanism,
trained only at context length 4096, improve length generalization over the
locked Llama-3-scaled RoPE control at **every** model scale from 550M through
2.545B without materially degrading short-context quality?

Design one mechanism and define its behavior beyond position 4095. The same
scale-specific mechanisms is not a solution.

## Reference Baselines

The locked scaling ladder E0-E5 model and training configs are defined in
`/task-tools/scaling_ladder_profile.sh` and loaded by
`/task-tools/run_scaling_ladder.sh`. Model, data and token order, optimizer and
learning rates, batch, schedule, seed, topology, precision, and evaluators are
fixed.

| Rung | Model | Parameters | Training tokens | Updates | Batch | GPUs |
|---|---|---:|---:|---:|---|---:|
| E0 | d1152-L12 | 550,337,664 | 2,904,358,912 | 44,317 | GBS16 / MBS2 | 8 |
| E1 | d1408-L15 | 837,007,744 | 3,612,672,000 | 55,125 | GBS16 / MBS2 | 8 |
| E2 | d1536-L16 | 998,036,992 | 4,982,571,008 | 38,014 | GBS32 / MBS4 | 8 |
| E3 | d1792-L18 | 1,384,584,448 | 10,559,946,752 | 40,283 | GBS64 / MBS2 | 32 |
| E4 | d2048-L21 | 1,934,716,160 | 14,805,106,688 | 56,477 | GBS64 / MBS2 | 32 |
| E5 | d2304-L23 | 2,544,614,912 | 18,617,466,880 | 35,510 | GBS128 / MBS1 | 128 |

All rungs use TP1 / PP1 and pure data parallelism. The 32- and 128-GPU rungs
use NCCL with the host RDMA devices exposed inside Apptainer. The task runner
rejects a topology, recipe, or data change.

### Baseline metrics

Trusted evaluation values, full baseline trajectories, and training-loss
curves are in
`/task-data/positional_adamh_baselines.json`. E0-E3 use their final
evaluations; E4 uses update 30,000 for all three metrics; E5 uses Paloma at
update 35,000 and LongPPL at update 35,510.

## Evaluation and success

Only Paloma and official LongPPL determine quality. The verifier reloads all
six staged checkpoints with the submitted source and reruns both evaluators.

- Paloma measures short-context quality over 16 subsets. Training evaluates it
  every 10,000 updates and at the final update. Final micro and macro BPB must
  each be at most 1% above the matching baseline.
- LongPPL is primary. The locked offline protocol evaluates 50 GovReport
  documents at 16,384-32,768 source tokens with the model trained only at
  4096. Training evaluates it at the final update; its value must strictly
  improve the matching baseline at every rung.

A valid submission uses one exact source inventory across E0-E5, keeps
parameters within 102% of each control, supplies every exact-final-update
checkpoint, and has matching source, run-contract, and policy evidence. The
six runs may execute concurrently with no scheduling prerequisite, but all six
must pass with one identical source hash.

Reward is zero unless every gate passes. Otherwise, for baseline and candidate
LongPPL `B_i` and `L_i`, reward is:

`exp(mean(log(B_i / L_i)))` for `i = E0 ... E5`.

Higher is better; every valid score is greater than 1.

## Workspace

| Item | Path / value |
|---|---|
| Writable source | `/app/project` |
| Experiment code | `/app/project/examples/training/positional_encoding` |
| Locked profiles | `/task-tools/scaling_ladder_profile.sh` |
| Parallel candidate runner | `/task-tools/run_candidate_ladder.sh` |
| Single-rung runner | `/task-tools/run_scaling_ladder.sh` |
| Task state tool | `/task-tools/scaling_ladder_task.py` |
| Output root | `/app/output` |
| Hardware pool | 32 nodes × 8 H100 = 256 GPUs in one LSF allocation |

The outer allocation is made once. Do not submit nested `bsub` jobs. Each
runner invocation atomically leases disjoint whole nodes from that pool and
releases them on exit.

Resource use per invocation is:

- E0, E1, E2: 1 node / 8 GPUs each;
- E3, E4: 4 nodes / 32 GPUs each;
- E5: 16 nodes / 128 GPUs each.

Therefore the pool can hold up to 32 small-rung runs, 8 E3/E4 runs, or 2 E5
runs concurrently, with mixed combinations whose total is at most 32 nodes.
One complete E0-E5 candidate occupies 216 GPUs and can run concurrently in the
single allocation, leaving 40 GPUs for up to five additional 8-GPU screening
runs. The six-scale task therefore should normally be launched in parallel.

## What you may change

Change only positional information or attention scope. Candidate hooks belong
in `examples/training/positional_encoding/runtime/positional_candidate.py`, or
the other source globs listed in `/app/task/policy.yaml`.

The mechanism may depend continuously on architectural quantities such as
layer index divided by layer count, head dimension, hidden width, or sequence
position. It may not branch on `E0`-`E5`, exact parameter counts, exact hidden
sizes, or a lookup table identifying the six benchmark models.

Do not modify data, tokenizer, model dimensions, optimizer, either AdamH LR,
schedule, objective, precision, seed, 4096 training length, token budgets,
batches, TP/PP/DP topology, launcher, evaluator, task tools, or trusted
pretraining entrypoint. Longer-sequence training, extra tokens, external
models, and post-training are forbidden.

## Persistent experiment loop

For each candidate source:

1. Inspect `/app/output/experiments.jsonl` and retained results. Write a
   falsifiable mechanism hypothesis, its behavior after position 4095, its
   expected scale behavior, and a useful ablation to
   `/app/output/hypotheses/<candidate>.md`.
2. Implement the mechanism. Validate a forward crossing position 4095,
   checkpoint reload, parameter count, and that the formula is scale-general.
3. Launch all six scales from the same unchanged source. This uses 216 GPUs:

   ```bash
   /task-tools/run_candidate_ladder.sh \
     --candidate-id <candidate> \
     --hypothesis-file /app/output/hypotheses/<candidate>.md
   ```

   It creates `<candidate>-e0` through `<candidate>-e5`. The host lease prevents
   GPU overlap. Before editing the next candidate, confirm that all six
   attempts emitted `run_contract.json`. Each contract freezes the exact
   source overlay; staging later rejects any mixed hashes. Use the single-rung
   runner directly for E0-only screening or a missing-rung retry.
4. Inspect each result:

   ```bash
   python /task-tools/scaling_ladder_task.py summarize \
     --attempt /app/output/attempts/<attempt-id>
   ```

5. If `/app/project` now contains another parallel candidate and you need to
   repeat or inspect an earlier source, restore any one of its attempts:

   ```bash
   python /task-tools/scaling_ladder_task.py restore-attempt-source \
     --attempt /app/output/attempts/<passing-attempt-id>
   ```

6. If any rung fails, retain its hypothesis, contract, status, and evaluator
   JSONs, then release bulky checkpoint state:

   ```bash
   python /task-tools/scaling_ladder_task.py prune \
     --attempt /app/output/attempts/<rejected-attempt-id>
   ```

   Analyze the cross-scale failure. You may retry only that rung with the same
   restored source, or edit the mechanism materially and launch a new
   six-scale set. E0-only screening is allowed when you deliberately want a
   cheap signal before committing another complete set.
7. After the same source passes E0-E5, stage the complete ladder using its E5
   attempt as the anchor:

   ```bash
   python /task-tools/scaling_ladder_task.py stage \
     --attempt /app/output/attempts/<candidate>-e5 \
     --hypothesis-file /app/output/hypotheses/<candidate>.md
   ```

   Staging records all six checkpoints and evaluator results, snapshots the
   exact source, and replaces an incumbent only when aggregate LongPPL gain
   improves. Staging never authorizes a final answer: continue the experiment
   loop until `/run-contract/RESEARCH_DEADLINE_UTC`.

Do not stop because one mechanism failed, because one complete ladder passed,
because training is expensive, or because the next idea is uncertain. Review
the retained evidence, choose the most informative falsifiable experiment, and
continue until the trusted deadline.

## Research deadline and handoff

The trusted deadline is `/run-contract/RESEARCH_DEADLINE_UTC`. Stage the first
passing ladder as the incumbent, but continue improving until the deadline;
replace it only with a complete ladder with higher aggregate LongPPL reward.
Do not stop or provide a final handoff early.

Near the deadline, avoid runs that cannot finish and use the remaining time for
audits and validation. At the deadline, leave the best complete incumbent
staged and audit its six checkpoints, metrics, source snapshot, and ledger. If
no ladder passes, report exhaustion; never fabricate missing results. Exiting
early returns zero.

An interrupted allocation preserves GPFS outputs for explicit continuation but
is not task completion.

## Required artifacts

- `/app/output/submission.json` describing E0-E5;
- six exact-final-update checkpoints from one source hash;
- six final Paloma and six final LongPPL result files;
- `/app/output/HYPOTHESIS.md`;
- `/app/output/experiments.jsonl` with every measured rung;
- retained hypotheses and evaluator evidence for rejected candidates;
- submitted source changes under `/app/project`.
