# Parameterization-transfer scaling-ladder harness

This is the candidate workspace for a single parameterization mechanism developed
on visible E0-E4. E5 is held out. Follow `/app/task/instruction.md` and use the
declared `/task-tools` entrypoints; never submit a nested LSF job.

Candidate logic belongs in `/app/project/parameterization_transfer.py`. The trusted
runtime adapter loads that recipe for visible training and for post-freeze trusted
evaluation. The checked-in identity implementation is infrastructure smoke only and
is not an eligible research submission.

## Locked scaling ladder

| Rung | Model | Parameters | Tokens | GPUs | Batch |
|---|---|---:|---:|---:|---|
| E0 | d1152-L12 | 550.3M | 2.904B | 8 | GBS16 / MBS2 |
| E1 | d1408-L15 | 837.0M | 3.613B | 8 | GBS16 / MBS2 |
| E2 | d1536-L16 | 998.0M | 4.983B | 8 | GBS32 / MBS4 |
| E3 | d1792-L18 | 1.385B | 10.560B | 32 | GBS64 / MBS2 |
| E4 | d2048-L21 | 1.935B | 14.805B | 32 | GBS64 / MBS2 |
All visible rungs use TP1 / PP1, context length 4096, the fixed production
batch/data trajectory, and seed 0. A valid freeze requires E0 and at least one
larger completed visible run from the same source inventory. Sanitized E5 shape
metadata is supplied only so the recipe can be materialized before freeze.

## Evaluation

Paloma runs every 10,000 parameterization updates and at the final update. The
trusted entrypoint records synchronized active-step
GPU time and token-normalized training loss in
`parameterization_cost_trace.jsonl`. Matched-cost scoring combines those traces with
the Paloma checkpoints and the offline AdamH reference manifest.

## Runtime layout

- `/app/project/parameterization_transfer.py`: candidate-owned scale-general recipe.
- `runtime/parameterization_candidate.py`: trusted adapter from the recipe to Megatron.
- `runtime/pretrain_gpt_marin_adamh.py`: trusted entrypoint.
- `runtime/lm_eval_runner.py`: trusted Paloma evaluator.
- `data_generation/validate_adamh_v6_blend.py`: trusted staged-data validator.
- `run_marin_adamh_bluevela_lsf.sh`: trusted multi-node RDMA launcher.
