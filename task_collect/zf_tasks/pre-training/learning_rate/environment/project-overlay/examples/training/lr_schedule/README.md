# Learning-rate schedule discovery workspace

Edit only `runtime/lr_schedule_candidate.py`. Its `build_schedule()` must
return one deterministic schedule used with identical equations and constants
at E0-E5.

At each update the trusted runtime supplies only normalized progress in
`[0, 1]`. The schedule returns one finite multiplier in `[0, 1]`, shared by
both locked AdamH learning-rate branches. It is applied directly to each
branch's locked `max_lr`, replacing the WSD curve rather than multiplying an
already scheduled LR. Values never compound, and neither peak LR nor the
parameter-group ratio can be retuned. The starter schedule exactly reproduces
the locked 10%-warmup / 70%-stable / 20%-decay WSD control.

Use `/task-tools/run_lr_schedule_candidate_ladder.sh` for a full or selected
rung set. Each attempt records `lr_schedule_trace.jsonl`, paired training loss
and active-training cost, Paloma results, checkpoints, and the frozen source
hash. The schedule is stateless, so checkpoint resume requires matching source
provenance but no participant state sidecar.
