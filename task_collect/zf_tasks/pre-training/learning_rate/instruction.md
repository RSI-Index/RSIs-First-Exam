# Learning-rate scheduling across a 550M-to-2.545B scaling ladder

## Research Question

Can one scale-general learning-rate schedule improve the Paloma and
training-loss trajectories over the locked Warmup-Stable-Decay (WSD) baseline
across 550M to 2.545B parameters without changing the optimizer, peak learning
rate, training tokens, or compute?

Design one deterministic schedule

\[
m_t=S_\phi(u_t),\qquad u_t=t/T_i,\qquad 0\le m_t\le 1,
\]

applied as a shared multiplier to the two existing AdamH learning-rate
branches' locked peak LRs. It replaces the WSD curve; it is not multiplied on
top of an already WSD-shaped LR. The same equation, constants, and source
inventory must run unchanged across E0-E5. The candidate observes only
normalized progress: it has no rung identity, absolute step or horizon,
training feedback, evaluation feedback, or persistent state.

This is schedule discovery, not scalar retuning. A valid method introduces a
mathematically explicit schedule mechanism such as an alternative warmup,
hold, decay, restart, cycle, or interaction among normalized-progress phases.
Changing only the locked peak learning rates, AdamH parameters, or independent
per-rung constants is not a new scale-general schedule.

## Reference Baselines

The locked E0-E5 scientific profiles are defined in
`/task-tools/lr_schedule_ladder_profile.sh` and loaded by
`/task-tools/run_lr_schedule_ladder.sh`. Model, initialization, tokenizer,
data and order, sequence length, tokens, updates, batch, topology,
model-compute precision, objective, AdamH optimizer, seed `0`, and evaluators
are fixed. WSD is the only scored reference schedule.

| Rung | Model | Parameters | Training tokens | Updates | Batch | GPUs |
|---|---|---:|---:|---:|---|---:|
| E0 | d1152-L12 | 550,337,664 | 2,904,358,912 | 44,317 | GBS16 / MBS2 | 8 |
| E1 | d1408-L15 | 837,007,744 | 3,612,672,000 | 55,125 | GBS16 / MBS2 | 8 |
| E2 | d1536-L16 | 998,036,992 | 4,982,571,008 | 38,014 | GBS32 / MBS4 | 8 |
| E3 | d1792-L18 | 1,384,584,448 | 10,559,946,752 | 40,283 | GBS64 / MBS2 | 32 |
| E4 | d2048-L21 | 1,934,716,160 | 14,805,106,688 | 56,477 | GBS64 / MBS2 | 32 |
| E5 | d2304-L23 | 2,544,614,912 | 18,617,466,880 | 35,510 | GBS128 / MBS1 | 128 |

All rungs use context 4096, TP1 / PP1 / CP1, and pure data parallelism. The
locked WSD control warms linearly from zero to one over the first 10% of
updates, remains at one, and decays linearly to zero over the final 20%.
AdamH transforms, parameter routing, branch peak LRs, and the ratio between
the two branches are frozen.

### Baseline metrics

Trusted run identity, scoring/final updates, every Paloma event, compact
100-update fixed-window loss curves, active-training cost, run state, and
provenance are in `/task-data/lr_schedule_baselines.json`. The file is the
offline source of truth; do not query external experiment tracking or invent
missing points.

## Evaluation and success

### Paloma Evaluation
TODO:


### Training loss Evaluation
TODO:

### Reward
TODO:

## Workspace

| Item | Path / value |
|---|---|
| Writable source | `/app/project` |
| Candidate module | `/app/project/examples/training/lr_schedule/runtime/lr_schedule_candidate.py` |
| Locked profiles | `/task-tools/lr_schedule_ladder_profile.sh` |
| Parallel runner | `/task-tools/run_lr_schedule_candidate_ladder.sh` |
| Single-rung runner | `/task-tools/run_lr_schedule_ladder.sh` |
| Task state tool | `/task-tools/lr_schedule_task.py` |
| Trusted evidence | `/task-data/lr_schedule_baselines.json` |
| Output root | `/app/output` |
| Hardware pool | 32 nodes × 8 H100 = 256 GPUs |

E0-E2 use one node each, E3-E4 use four nodes each, and E5 uses sixteen
nodes. One full ladder occupies 27 nodes / 216 GPUs. The outer allocation is
created once; do not submit nested `bsub` jobs. A trusted recovery allocation
may expose only the 27 nodes needed for one full ladder. Node quarantine and
wave order are operational facts, not candidate changes.

## What you may change

Change only `lr_schedule_candidate.py`. Its `build_schedule()` returns an
object implementing:

```python
def multiplier(self, progress: float) -> float:
    ...
```

The schedule may use any deterministic continuous or piecewise functional
form of normalized progress, including alternative warmup, hold, decay,
restart, or cycle structure. It must return one finite value in `[0, 1]`
shared by both locked LR branches.

Do not use rung, model shape, parameter count, absolute step/horizon, tokens,
batch, GPUs, seed, hostname, time, environment metadata, filesystem data,
training loss, gradients, optimizer diagnostics, baseline trajectories,
Paloma, or evaluator results as schedule inputs. Do not add state or feedback.
Do not change peak LR, parameter-group ratios, optimizer transforms, model,
data, objective, precision, training horizon, topology, evaluator, launcher,
or task tools. Parameter averaging, rollback, checkpoint selection, extra
updates, and post-training are outside the edit surface.

Before a scored ladder, write `/app/output/LR_SCHEDULE.md` with the exact
schedule equations, closest reduced control, term-by-term mechanism claim,
predicted helpful/harmful phases, and a preregistered mechanism-on/off
experiment. Run full and reduced controls at E0, E2, and E5 and create:

```json
{
  "version": 1,
  "anchors": {
    "E0": {"full_attempt_id": "...", "reduced_attempt_id": "..."},
    "E2": {"full_attempt_id": "...", "reduced_attempt_id": "..."},
    "E5": {"full_attempt_id": "...", "reduced_attempt_id": "..."}
  }
}
```

Save this as `/app/output/mechanism_ablation.json`. Full IDs must be selected
ladder runs; reduced source must differ from full source.

## Known rejected result

Treat pure tuning of WSD warmup/decay fractions as a tuning baseline, not a
discovered mechanism. Multipliers above one are rejected because they change
the locked peak LR and confound schedule shape with base-LR transfer. Preserve
any measured tuning-only result in the experiment ledger as negative evidence.

## Persistent experiment loop

Use monotonic candidate IDs such as `lr-001-phased-decay`; never reuse an
attempt directory.

1. Inspect baseline histories, the experiment ledger, and deadline. Write one
   exact mechanism hypothesis and ablation to
   `/app/output/hypotheses/<candidate>.md`.
2. Implement the schedule and validate boundary values, finiteness, range,
   continuity where claimed, determinism, shared use by both LR branches, and
   exact checkpoint-resume behavior.
3. Use E0-E2 screens when useful, then run the preregistered full versus
   reduced controls at E0, E2, and E5.
4. Freeze one source and launch all scales:

   ```bash
   /task-tools/run_lr_schedule_candidate_ladder.sh \
     --candidate-id <candidate> \
     --seed 0 \
     --hypothesis-file /app/output/hypotheses/<candidate>.md
   ```

5. Inspect attempts with
   `python /task-tools/lr_schedule_task.py summarize --attempt <attempt>`.
   Record metrics, mechanism verdict, and decision in
   `/app/output/experiments.jsonl`.
6. Preserve hypotheses, contracts, and evaluator outputs for failures; prune
   only bulky checkpoints.
7. Stage the first unchanged-source ladder that passes all gates using its E5
   attempt, then replace it only with a higher-reward complete ladder.

## Research deadline and handoff

Each rung has research budget `B_i = 6C_i`, where `C_i` is its trusted WSD
active-training cost. The outer allocation lasts 72 hours, the agent envelope
68 hours, and the trusted scientific deadline 66 hours. Stage the first
passing ladder but keep searching until the deadline. At 66 hours, stop
training and source changes; use the next two hours for ledger audit, freeze,
and handoff. The final four allocation hours are reserved for trusted reloads
and cleanup. Failures after update 1 consume budget.

At the deadline, retain the strongest complete incumbent and audit the
schedule mechanism, ablation, six checkpoints, metrics, source, and ledger. If
none passes, report exhaustion. Exiting before the research deadline returns
zero. Resume preserves the original deadline; interruption is not completion.

## Required artifacts

- `/app/output/submission.json` and six exact-final seed-0 checkpoints from one
  source hash;
- `/app/output/LR_SCHEDULE.md` and `/app/output/mechanism_ablation.json`;
- full and reduced E0/E2/E5 checkpoints and evaluator evidence;
- exact submitted source and candidate-owned configuration;
- raw and fixed-window loss histories, LR traces, and all Paloma results;
- `/app/output/HYPOTHESIS.md` and `/app/output/experiments.jsonl`;
- retained hypotheses, contracts, policy/reload reports, and evaluator evidence
  for rejected candidates.
