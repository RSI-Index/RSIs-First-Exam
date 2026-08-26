# Gated DeltaNet fixed-time autoresearch

You are running a cumulative, empirical research loop over the pinned public
`NVlabs/GatedDeltaNet` codebase. Find the model that achieves the lowest frozen
validation perplexity after **exactly 1,200 measured training seconds on 32
H100s**, then explain which mechanism-level evidence led to it.

This is deliberately a hardware-local autoresearch benchmark. It does **not**
claim matched tokens, matched parameters, release-recipe fidelity, or portable
ranking across machines. Model size, batch shape, kernels and throughput are
part of the design under a fixed wall-clock training budget.

Do not use papers, the web, remote documentation, pretrained checkpoints or
external data. The pinned repository, staged SlimPajama and your own measured
experiments are the entire scientific world of this task.

## The fixed setup

| Item | Contract |
|---|---|
| Writable research tree | `/app/project`; clean task-baseline git commit derived from `NVlabs/GatedDeltaNet@b53d6d3a161267432a79c1c04af69fa52bddc921` |
| Editable source | `pretrain.py` and `lit_gpt/**` only |
| Starting model | `GatedDeltaNet_H1_0.4B`; the first experiment must run this live, unchanged |
| Training data | staged packed SlimPajama `train_slim*` only |
| Validation data | staged packed SlimPajama `validation*`; never train on it |
| Hardware | exactly 4 nodes × 8 H100s = 32 ranks for every experiment |
| Training clock | 1,200 measured seconds; ten identical warm-up iterations are excluded to absorb cold Triton compilation; final evaluation and checkpoint save are outside the clock |
| Token ceiling | 15B nominal tokens is only a safety ceiling; actual tokens are measured and may vary with throughput/batching |
| Architecture invariants | sequence length 4096, vocabulary and padded vocabulary 32000, same tokenizer/data |
| Size constraint | no iso-parameter cap; a larger model pays through speed and memory, and OOM is a crash |
| Decision metric | geometric mean of `val_ppl@1x` (2048) and `val_ppl@2x` (4096), 15 frozen validation batches; lower is better |
| Ledger | `/app/output/results.tsv`, written only through the frozen task tool |
| Research argument | `/app/output/RESEARCH.md` |

The task baseline commit includes the three locked H100/32-rank compatibility
fixes and the fixed-time measurement loop. You may refactor `pretrain.py` as a
research variable, but you must preserve honest measurement, the 1,200-second
limit, final frozen-protocol validation and checkpoint evidence.

The 20-minute budget is tied to measured cluster cost, not chosen arbitrarily:
the pristine 15B H1 run trained for 6,082.90 seconds. Five 1,200-second runs use
53.33 training GPU-hours, approximately the same 54.07 training GPU-hours as
one old full-budget candidate, while yielding a baseline, two diagnostic probes
and two evidence-led synthesis tests. Final validation adds overhead in both
lanes and is reported separately through end-to-end wall time.

## What counts as scientific progress

The final metric selects the artifact, but the experiment sequence must test a
causal story rather than merely sample unrelated hyperparameters.

The first valid experiment is `000-baseline`. Before any synthesis experiment,
run at least **two controlled mechanism probes**. A probe changes one
interpretable architectural mechanism and predicts both a quality effect and,
where relevant, a throughput/memory effect. Examples include counterfactuals
about the delta update, gating/decay parameterization, hybrid GDN/attention
interleave, local window, head structure, normalization or MLP allocation. A
learning-rate-only change is not a mechanism probe.

Then run at least **two synthesis experiments** whose designs explicitly use
the probe evidence. There must be at least five valid fixed-time experiments
in total (baseline + probes + synthesis); crashes do not count. You may and
should continue while a decision-relevant hypothesis remains.

For every valid attempt, write in `RESEARCH.md`:

1. a falsifiable hypothesis and its mechanism;
2. the predicted direction for perplexity, throughput/tokens and memory;
3. the controlled change and what remained fixed;
4. the observed result, including the attempt ID;
5. what the result rules out, what it does not distinguish, and the next test.

Negative probes are valuable when they discriminate between explanations.
Prefer the smaller or simpler implementation when evidence is effectively
tied, but mark `keep` only for a strict improvement in the unrounded decision
metric.

## Editable action space

Architecture and recipe are both open inside the two editable locations. You
may change or replace token mixers, gating/delta rules, hybrid layouts,
attention windows, positional schemes, norms, MLPs, head layouts and kernels;
add a new named `Config`; change optimizer, schedule, batching, seed or
objective shaping; or delete machinery that is unnecessary.

The submitted model must remain constructible as
`Config.from_name(<model_config>)` plus `GPT`, with `Block` retained as an FSDP
wrap unit. `block_size`, vocabulary, data, topology, clock and evaluator are
frozen. Do not optimize against validation examples individually.

## Cumulative commit-and-test loop

Keep research artifacts out of `/app/project`; it should contain only source.
The initial tree is a clean git baseline, so commit every tested source state.
Use one commit per experiment and never reuse a tested commit.

For each attempt:

1. Record the hypothesis in `RESEARCH.md`, edit one interpretable variable
   group, inspect the diff, and commit it. For `000-baseline`, use the unchanged
   task-baseline commit.
2. Submit the experiment asynchronously and end the turn; do not sleep or
   poll:

   ```bash
   attempt=000-baseline  # then 001-..., 002-..., ...
   python /task-tools/gdn_async_run.py submit \
     --attempt-id "$attempt" --poll-seconds 1800 -- pretrain.py \
     --train_data_dir /datasets/gated_deltanet-official/slimpajama/packed/slim \
     --val_data_dir /datasets/gated_deltanet-official/slimpajama/packed/slim \
     --output_root /app/output \
     --exp_name "$attempt" \
     --model_name GatedDeltaNet_H1_0.4B \
     --train_config tsz512x4k_15B \
     --train_time_seconds 1200 \
     --eval_iters 15 \
     --eval_step_interval 1000000 \
     --save_step_interval 500 \
     --micro_batch_size 8
   ```

   Change model/recipe flags when the hypothesis requires it. The frozen async
   adapter resumes your session only after the request is terminal.
3. On resume, inspect real status and evidence, then record the decision:

   ```bash
   python /task-tools/gdn_autoresearch_task_tool.py summarize \
     --attempt "/app/output/attempts/$attempt"
   python /task-tools/gdn_autoresearch_task_tool.py record \
     --attempt "/app/output/attempts/$attempt" \
     --attempt-id "$attempt" \
     --phase baseline \
     --commit "$(git -C /app/project rev-parse HEAD)" \
     --status keep \
     --description "live H1 fixed-time baseline"
   ```

   Later phases are `probe` or `synthesis`; statuses are `keep`, `discard`, or
   `crash`. The tool extracts metric, measured training seconds, end-to-end
   wall seconds, GPU count, GPU-hours, tokens, parameters and peak memory from
   the run. Do not hand-edit `results.tsv`.
4. If the result is a strict metric improvement, keep its commit as the new
   incumbent. Otherwise record it, then return exactly to the incumbent with
   `git reset --hard <incumbent-commit>` before designing the next change.
   This makes every retained improvement cumulative.

The run can resume after an outer scheduler interruption. Trust the attempt's
status, checkpoint counters and logs; the measured budget resumes from the
checkpoint.

## Selection and required outputs

When the research program is complete, leave `/app/project` at the best keep
commit. Stage that attempt and run the audit:

```bash
python /task-tools/gdn_autoresearch_task_tool.py stage \
  --attempt-id <best-attempt> \
  --checkpoint /app/output/attempts/<best-attempt>/outputs/<run>/final-model-ckpt.pth
# Write provenance.json with the fields below.
python /task-tools/gdn_autoresearch_task_tool.py audit --output-root /app/output
python /task-tools/gdn_async_run.py complete-control
```

Required outputs are:

- `results.tsv` with one frozen-tool-generated row per tested commit;
- `RESEARCH.md`, discussing every valid attempt and the evidence chain;
- exactly one `submission/final-model-ckpt.pth` and `model_state.pt`;
- `submission-selection.json` from the staging tool;
- `provenance.json` with `base_model`, `model_config`, `training_data`,
  `time_budget_seconds`, `training_command`, `selected_commit`,
  `upstream_commits`, `downloads`, `evaluation_commands`,
  `optimization_attempts`, `topology_mapping`, and `web_search: "disabled"`.

The final verifier reruns the same frozen 1x/2x validation protocol from your
selected source commit and model state. Reward is `1 / (1 + ln(P))`, where
`P` is the unrounded validation-perplexity geometric mean.
