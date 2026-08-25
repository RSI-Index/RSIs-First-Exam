# Positional-encoding scaling-ladder experiment harness

This is the candidate workspace for a single positional mechanism evaluated
from 550M through 2.545B parameters. Follow `/app/task/instruction.md` and use
`/task-tools/run_scaling_ladder.sh`; never submit a nested LSF job.

Candidate logic belongs in `runtime/positional_candidate.py`. The trusted
pretraining entrypoint invokes its model-configuration and runtime hooks during
both training and verifier checkpoint reload.

## Locked scaling ladder

| Rung | Model | Parameters | Tokens | GPUs | Batch |
|---|---|---:|---:|---:|---|
| E0 | d1152-L12 | 550.3M | 2.904B | 8 | GBS16 / MBS2 |
| E1 | d1408-L15 | 837.0M | 3.613B | 8 | GBS16 / MBS2 |
| E2 | d1536-L16 | 998.0M | 4.983B | 8 | GBS32 / MBS4 |
| E3 | d1792-L18 | 1.385B | 10.560B | 32 | GBS64 / MBS2 |
| E4 | d2048-L21 | 1.935B | 14.805B | 32 | GBS64 / MBS2 |
| E5 | d2304-L23 | 2.545B | 18.617B | 128 | GBS128 / MBS1 |

All rungs use TP1 / PP1, context length 4096, the fixed scale-specific Marin
AdamH recipe, and seed 0. E0-E5 may run concurrently, but a valid submission
must pass every rung with the same exact source inventory.

## Evaluation

Paloma runs every 10,000 optimizer updates and at the final update. Official
offline LongPPL runs at the final update over 50 GovReport documents at
16k-32k source tokens, causing the candidate model's position IDs to extend
beyond 4095. Final results are written below each attempt's `eval_harness/`
and `longppl/` directories.

## Runtime layout

- `runtime/positional_candidate.py`: candidate-owned scale-general hooks.
- `runtime/pretrain_gpt_marin_adamh.py`: trusted entrypoint.
- `runtime/lm_eval_runner.py` and `runtime/longppl_runner.py`: trusted evaluators.
- `data_generation/validate_adamh_v6_blend.py`: trusted staged-data validator.
- `run_marin_adamh_bluevela_lsf.sh`: trusted multi-node RDMA launcher.
