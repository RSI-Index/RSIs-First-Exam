# Persistent task-design memory

Effective 2026-07-17, new and revised AutoResearch tasks are repository-first.

1. Pin the author's runnable public repository to an immutable commit.
2. Derive the baseline model, config, launch recipe, data format, optimizer,
   budget, and evaluator from files in that repository only.
3. Do not use a paper to fill a repository gap. Mark unpublished data,
   checkpoints, settings, or baseline scores as `missing` or `unmeasured`.
4. Do not substitute a third-party implementation or similarly named model.
   If such a source is intentionally studied, create a separately named task.
5. Keep infrastructure additions separate and explicit: container/dependency
   pins, offline staging, scheduler/launcher translation, artifact conversion,
   integrity checks, and scalar reward.
6. Prefer allocating the repository's released topology exactly. Never call an
   infrastructure mapping exact when it changes world size, data order,
   numerics, or another observable run property.
   Compute availability is not a reason to shrink a released topology.
7. Papers and retired task versions may remain only as clearly labeled audit
   history; they are not inputs or baselines for active task design.

For Gated DeltaNet, the active package is `gated_deltanet`, sourced from
`NVlabs/GatedDeltaNet@b53d6d3a161267432a79c1c04af69fa52bddc921` and the
repository's `GatedDeltaNet_H1_0.4B` / `tsz512x4k_15B` release path. The former
FLA/Flame/FineWeb/PG19 design is retained only in `gated_deltanet_old`.
The active H100 task-pristine checkout has exactly three locked infrastructure
patches (combined diff SHA256
`63c451bf94a954445c89850ae4fd7efd3ae8188483f4974050e4088685e5eba4`):
`chunk.py` and `wy_fast.py` remove Triton 2.3.0 `num_warps=8/16/32` autotune
candidates that abort on Hopper backward, while `pretrain.py` fixes 32-rank
resume/output-directory/multiprocessing startup races. These are disclosed
environment compatibility changes, never task-design inputs or agent
optimization variables. The author repository README estimates its public
4-node/15B run at approximately four hours; six hours is only the local SOP
review threshold, not a timeout or scientific parameter.
H100 32-rank qualification job 234716 reached iter 10 / optimizer step 5 after
120 seconds and measured a steady-state rate near 430 ms/iteration and 77.4K
tokens/s/GPU. Its local stop did not propagate through remote `blaunch`, so it
continued to logged iter 8060 and then hit the one-hour LSF runlimit; it is an
incomplete infrastructure qualification, never a baseline. The measured full
training estimate was about 1.8 hours. The subsequent fresh reference, job
238329 / `20260718T153000Z-pristine-h100-v1`, completed all 14,305 iterations,
7,152 optimizer updates, and 14,999,879,680 tokens from random initialization.
Training took 6,082.90 seconds and the complete LSF job took 6,445 seconds.
Its raw metrics were `val_ppl@1x=15.084142519039398` and
`val_ppl@2x=14.536125056457275`; the task-authored reward was `0.269277`.
This sealed single-seed current-environment result is the active matched
baseline for agent optimization.

For RLM, the active package is `rlm`, sourced from
`alexzhang13/rlm@72d6940142ddfb84ee6be573dc999a37e633e671` and its runnable
`training/configs/rlm-qwen3-30b-example.toml`. The compatible RL dependency is
`PrimeIntellect-ai/prime-rl@84f072bc99a2cd0b740ff9e665e1accf55fb6cf4`
(v0.5.0): its config parser accepts the RLM TOML unchanged. The previously
pinned prime-rl v0.6 commit rejects the published `eval_base_model` and
`filters` fields. The active primary metric is raw OOLONG reward; the former
environment-defined subcall-cost scalar is retained only in `rlm_old`.
The active zero-update 25-example verifier reference was measured on 2026-07-18
as raw OOLONG `0.36000535799246636` (job 233113, 25 valid rows, 0 errors,
3,747 seconds end to end). This is neither the active 60-update matched training
baseline nor the repository's full 200-update reproduction, and it is not
comparable to the archived temperature-0 cost-aware score.
User review selected RLM Candidate B, on-policy training using the repository's
runnable setup. Runtime measurement then established the active task protocol:
one experiment/single run is 60 training steps with 32 rollouts per step (1,920
total), checkpoints every 20 steps, and one 25-row final evaluation. Each outer
eval rollout has a 600-second wall-clock cap; timeout rows remain in the fixed
denominator and score zero. Trying another hypothesis is another run; it is not
part of the first run's wall-time estimate. The unchanged upstream config is
still a 200-step schedule with base/20-step evals and must be labeled separately.
Initial LSF job 233873 exited in 61 seconds
before model load or any rollout because the environment omitted prime-rl's
locked `flash-attn` extra. Rebuild 233922 and trainer smoke 233937 passed, but
replacement 234111 exposed missing `orjson` in the separately spawned
orchestrator and exited in 91 seconds, still before model load/rollouts. These
are infrastructure startup failures, not scientific runs or optimization
attempts. The environment must follow the pinned README's complete
`uv sync --frozen --all-extras`; rebuild 234148 and three-heavy-import smoke
234149 passed. Job 234338 next exposed prime-rl's automatic write of a converted
`prime/` DCP cache inside the HF snapshot. Because the shared asset was mounted
read-only it exited before rollouts and was terminated after 364 seconds, still
0 scientific runs. The launcher now uses a run-private writable HF namespace
while nested-mounting the 16 source weights, shared blobs, and OOLONG repository
read-only. Overlay smoke passed. Clean 8-H100 job 234447 completed the private
13-shard conversion, brought official TP2×DP2 inference online, scored the
scheduled step-0 and step-20 evals, and produced stable checkpoints 20 and 40.
It was terminated during the long-tailed step-40 eval after the shorter task
protocol was selected. It is an upstream-path qualification, not a complete
200-step reproduction and not the matched 60-step task baseline. Across 34
ordinary steps it measured 128.10 s/step: 60 steps projected to 2.14 h training,
or about 2.5–3 h including startup, checkpoints, and capped final evaluation;
the matched run below subsequently measured 2:17:28 end to end.
The refreshed task images are environment SHA256 `01ed6e1d...` and tests
SHA256 `873c51b8...`; 1-H100 heavy-import/config/data smoke job 238562 passed.
The first clean matched 60-step baseline is LSF job 238576, frozen run
`20260718T193200Z-task-60-baseline-r1`, with zero optimization attempts.
It completed all 60 updates and 1,920 train rollouts, produced stable
checkpoints 20/40/60, and evaluated all 25 final rows in 602.73 seconds. Total
LSF wall time was 8,248 seconds (2:17:28); raw OOLONG reward was
`0.360077631820896`, with 0 errors and two 600-second timeout rows retained as
zeros. The RL process exited 0. LSF nevertheless reported EXIT 1 after
completion because the wrapper's source manifest counted generated
`__pycache__/*.pyc` files. Filtering those caches makes the before/after
manifests byte-identical; both RLM launchers now exclude them. This is a
wrapper-audit false positive, not a failed scientific run. The score is only
about `0.0000723` above the single zero-update reference, so no improvement
claim is supported without matched repeats and uncertainty.
