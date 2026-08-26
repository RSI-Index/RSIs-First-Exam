# RLM Step 4 baseline result

This is the unchanged 60-update baseline run from LSF job 238576. It is the
Step 4 reproduction result, with zero optimization attempts.

## Training reward curve

The per-step `Reward` emitted by the repository orchestrator is the mean reward
of that step's OOLONG `spam` training rollouts after the configured filters. It
is not the checkpoint evaluation metric on `trec_coarse`.

- Raw 60-point curve: [`STEP4_BASELINE_TRAIN_REWARD.csv`](./STEP4_BASELINE_TRAIN_REWARD.csv)
- Mean over all 60 updates: `0.658143333`
- Updates 1–20 mean: `0.672135`
- Updates 21–40 mean: `0.664445`
- Updates 41–60 mean: `0.637850`

The source log labels the first completed update as `Step 0`. The CSV therefore
keeps both `orchestrator_step=0..59` and the less ambiguous
`updates_completed=1..60`. Checkpoint 20 is after rows 1–20, checkpoint 40 is
after rows 21–40, and checkpoint 60 is after rows 41–60.

## Checkpoint evaluation reward curve

All points use the repository OOLONG `trec_coarse` 25-example evaluation,
TP2×DP2 inference, model-default sampling, the unchanged OOLONG scorer, and the
same 600-second per-rollout cutoff used by the baseline final eval.

| Checkpoint | Raw OOLONG reward | Rows | Errors | Timeouts | State |
|---:|---:|---:|---:|---:|---|
| 20 | `0.3607127179228175` | 25 | 0 | 0 | post-hoc eval, LSF job 252617 |
| 40 | `0.36000000000172944` | 25 | 0 | 1 | post-hoc eval, LSF job 252618 |
| 60 | `0.360077631820896` | 25 | 0 | 2 | completed in the baseline run |

- Machine-readable curve: [`STEP4_BASELINE_CHECKPOINT_EVAL.csv`](./STEP4_BASELINE_CHECKPOINT_EVAL.csv)
- Combined plot: [`STEP4_BASELINE_REWARD_CURVES.svg`](./STEP4_BASELINE_REWARD_CURVES.svg)
- The exact `(example_id, answer, info)` sequence is identical at all three
  checkpoints; its canonical JSON SHA-256 is
  `7316462003ccd66373c0801bc2dcce5d4156e1e84ded05c4ea805a5d7f296efc`.
- Step 20 and 40 were evaluated post hoc from their saved merged checkpoints.
  Step 60 is the final-checkpoint eval already performed inside the baseline
  run. The evaluation dataset, scorer, TP2×DP2 topology, model-default
  sampling, completion/iteration caps, and 600-second cutoff are matched.

The checkpoint curve is effectively flat: step 20 is about `+0.0006351` above
step 60, and step 40 is about `-0.0000776` below it. These differences are far
smaller than the granularity/noise of one 25-row stochastic evaluation. The
measured baseline therefore does not show a reliable improvement from update
20 to update 60. The training-rollout reward is also noisy and its 20-update
block mean declines from `0.672135` to `0.664445` to `0.637850`; it should not
be confused with the checkpoint evaluation curve.
