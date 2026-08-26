# Understand and beat the Molmo2 video pointing recipe

This is a data/recipe/architecture-science task on video pointing. Under the
pinned public `allenai/molmo2` repository's staged data, update budget, and
topology, your goal is to **fine-tune a model from the pinned released
Molmo2-4B pretrain-stage checkpoint that beats the pristine baseline lane on
the frozen video pointing evaluation** — by changing the fine-tuning data
mixture, the temporal/visual preprocessing, the supervision format, the
training recipe, or the pointing architecture itself.

A guiding question to shape your hypotheses (encouraged in your notes, but
not a separately verified deliverable): **what carries video pointing
quality?** Which mechanisms — the density and cap of frame sampling
(`max_frames`, `max_fps`, `frame_sample_mode`), the time encoding
(`time_mode`), token pooling (`pooling_h`/`pooling_w`), the point/count
supervision mix and its min/max-count splits, the pointing output format
(coordinates-as-text vs the in-repo MolmoPoint grounding-token head in
`olmo/models/molmo_point/`), loss token weighting — actually determine
pointing and counting accuracy on video? Controlled ablations that answer
this are usually the fastest route to a principled improvement, and negative
results are worth recording, but your score comes only from the best
submitted model.

Do not use a paper or any external source: the pinned repository, the staged
data and checkpoints, and your own measurements are the only allowed inputs.

## Setup

| Item | Path / value |
|---|---|
| Writable workspace | `/app/project`, pinned `allenai/molmo2@f3cb1085fbb97c4a4d7fdcadd77cf871bf37a88a` |
| Starting checkpoint (frozen init) | staged released Molmo2-4B pretrain-stage checkpoint at `/datasets/molmo2-official/checkpoints/Molmo2-4B-Pretrain` (Qwen3-4B-Instruct LLM + SigLIP2 ViT, trained on image captioning, NLP, and image pointing only — no video pointing yet) |
| Released recipe (starting point, editable) | `launch_scripts/sft.py` fine-tuning defaults: AdamW, connector/vit lr `5e-6`, llm lr `1e-5`, multimodal schedule with 200-step warmup, `alpha_f=0.1`, grad clip 1.0, global batch 128, `seq_len` 16384, seed 6198, and the released `video_pointing` sub-mixture (`vixmo_points_minmax_0_5` 0.4, `vixmo_points_minmax_6_25` 0.3, `vixmo_points_minmax_26_60` 0.1, `academic_points_clip_63s_2fps` 0.2) |
| Update budget | at most 8,000 optimizer updates per candidate at global batch ≤ 128 and `seq_len` ≤ 16,384 (≤ 16,777,216,000 packed tokens); never exceed it |
| Training entrypoint | `/app/project/launch_scripts/sft.py` (editable; new launch scripts allowed) |
| Training data | staged corpora under `MOLMO_DATA_DIR=/datasets/molmo2-official/molmo_data` and `HF_HOME=/datasets/molmo2-official/huggingface`: `allenai/Molmo2-VideoPoint` (train) and the academic point-annotation clips used by `academic_points_clip_63s_2fps`, plus the staged pretrain-stage corpora (`pixmo_cap`, `pixmo_points_train`, `pixmo_multi_points`, `pixmo_count_train`, `cosyn_point`, `tulu4`) for replay/regularization |
| Validation data | `vixmo_points_point_eval:val` (`allenai/Molmo2-VideoPointEval`) and `vixmo_points_count_clip_63s:val`; never train on any eval split |
| Topology | 1 node × 8 H100s, exactly; context parallelism within the node is free |
| Parameter budget | whole-model parameters ≤ 1.005 × the pristine starting checkpoint's whole-model count; smaller is allowed |
| Architecture invariants | tokenizer and text vocabulary unchanged; `mm_preprocessor.video.max_frames` ≤ 128 and `llm.max_sequence_length` ≤ 16,384 in the submitted config; submitted checkpoint must load from your workspace via `MolmoConfig.load(<ckpt>/config.yaml, key="model")` and the repo's standard checkpoint loading |
| Final metric | mean of official-protocol video count accuracy (`correct` on `vixmo_points_count_clip_63s:val`) and video pointing F1 (`f1` on `vixmo_points_point_eval:val`); higher is better |
| Submission | one selected within-budget checkpoint staged as `submission/` under `/app/output`, provenance, ledger |
| Network | external research and web search disabled; staged data read-only; `HF_DATASETS_OFFLINE=1` |

There are no locked source patches for this task: the pinned commit runs
as released. The staged offline data mirrors, the staged checkpoints, the
container, the launch wrapper, the async adapter, the verifier, and the
reward are task infrastructure, not optimization variables.

## Action space

The fine-tuning lane from the frozen starting checkpoint is yours to change.
Free variables include:

- **Mixture**: which staged datasets, their sampling rates and
  `root_size_factor`s, the point/count `mode` mix, min/max point-count
  splits, `oversample` variants, `p_multi_turn`,
  `multi_message_short_clips`, replay of staged pretrain-stage data, and
  curriculum/data ordering within the staged corpora;
- **Temporal and visual preprocessing** (`VideoPreprocessorConfig`,
  `Molmo2PreprocessorConfig`): `max_frames` (≤ 128), `max_fps`,
  `frame_sample_mode`, `time_mode`, `time_sampling`,
  `pooling_h`/`pooling_w`, crop settings — these live in the model config
  and travel with your checkpoint into evaluation;
- **Supervision format**: `data_formatter.pointing_format`, prompt
  templates, system prompt, message format, and new formats you implement —
  but note that the frozen evaluator decodes your generations with the
  in-repo extractor in `olmo/preprocessing/point_formatter.py`
  (`UnifiedPointFormatter`, then `LegacyPointFormatting`). A format that
  extractor cannot parse yields zero extracted points and therefore a zero
  pointing F1, however good the underlying model is. Verify decodability
  before spending a full-budget run on a new format;
- **Loss shaping**: `loss_token_weighting`, `MessageWeight`s, residual
  dropouts, auxiliary losses;
- **Recipe**: optimizer, per-component learning rates, schedule, warmup,
  weight decay, batching, `seq_len` (≤ 16,384), seed, `cp_degree`, and
  `launch_scripts/sft.py` itself (or a new launch script);
- **Architecture**: pointing-head changes implemented in your workspace —
  including adapting the in-repo MolmoPoint grounding-token mechanism
  (`olmo/models/molmo_point/`) to this lane, or entirely new pointing
  mechanisms — as long as LLM and ViT weights initialize from the frozen
  starting checkpoint, newly added modules initialize from scratch inside
  the update budget, and the submitted checkpoint remains loadable from
  your workspace code by the repo's standard config/checkpoint machinery.

Frozen are only: the starting checkpoint, the staged corpora (no external,
scraped, or synthetic-from-external data; model-generated synthetic data
derived from staged corpora is allowed and must be declared), the 8,000
update / 128 batch / 16,384 seq-len budget, the 1×8 topology, the tokenizer
and vocabulary, the parameter budget, the eval-path code
(`olmo/eval/**`, `launch_scripts/eval.py`), and the final evaluator.

Be deliberate about attribution: when you change the mixture and the
preprocessing at the same time you cannot tell which one helped, so prefer
controlled comparisons (one variable group per attempt, fixed seed and data
order when you want comparability with the baseline anchors).

## Measured pristine baseline

> **Status: pending (design_sop step 4).** The pristine reproduction run of
> the released lane has not been executed yet. The table below is filled by
> the task-construction reproduction before this task is finalized; agents
> never see this note, only measured values.

The sealed reference is the unchanged lane: `sft.py` released defaults, the
released `video_pointing` sub-mixture over the staged corpora, 8,000
updates from the staged Molmo2-4B pretrain checkpoint on 1×8 H100s:

| Quantity | Measured value |
|---|---:|
| Whole-model parameters | TBD |
| `count_correct` (`vixmo_points_count_clip_63s:val`) | TBD |
| `count_close` | TBD |
| `point_precision` / `point_recall` / `point_f1` (`vixmo_points_point_eval:val`) | TBD |
| `video_point_score` (mean of `count_correct`, `point_f1`) | TBD |
| Task reward | TBD |
| Training time | TBD on 8 H100s |

A second sealed anchor is the staged released `Molmo2-VideoPoint-4B`
checkpoint (the authors' video-pointing specialist, fine-tuned from the
same pretrain stage) evaluated under this exact frozen protocol:
`count_correct` TBD, `point_f1` TBD, `video_point_score` TBD. Matching or
beating this anchor at the matched budget is the strong version of the task.

Fixed-seed in-training anchors for screening (update → `vixmo_points_count:val`
`correct` under the trainer's built-in eval): TBD after the reproduction
run. When a candidate keeps the released seed, batch, and data order, its
anchors at the same update count are directly comparable to these.

## Agent experiment loop

One **scientific single run** is one independently initialized candidate
fine-tuned to the full 8,000-update budget on 1×8 ranks. Screening prefix
runs are allowed for triage (label them `screening`, record real counters,
keep the full-run data order and LR horizon); only full-budget runs support
attribution claims and final selection.

For each attempt:

1. State one falsifiable hypothesis (mechanism or recipe change → expected
   effect on count accuracy and/or point F1, and why).
2. Change one interpretable variable group; record an exact patch or source
   hash.
3. Submit asynchronously and end your turn (do not sleep or poll):

   ```bash
   attempt=vpt-001-<short-name>
   python /task-tools/mvp_async_run.py submit \
     --attempt-id "$attempt" --poll-seconds 1800 -- \
     launch_scripts/sft.py \
     /datasets/molmo2-official/checkpoints/Molmo2-4B-Pretrain \
     <your-mixture-name> \
     --save_folder /app/output/attempts/$attempt/save \
     --max_duration 8000 \
     --global_train_batch_size 128 \
     --seq_len 16384 \
     --wandb=null
   ```

   This is the baseline-shaped example; mixture names, model overrides
   (e.g. `--model.mm_preprocessor.video.max_frames=...`), recipe flags,
   and any hyperparameters you expose may differ per your hypothesis, as
   long as the run stays within the update budget on the staged corpora.

   The async adapter keeps the environment alive, checks `status.json`
   every 30 minutes without an LLM call, and resumes this session when the
   attempt reaches a terminal state. On resume, summarize the real
   evidence:

   ```bash
   python /task-tools/mvp_task_tool.py summarize \
     --attempt "/app/output/attempts/$attempt"
   ```

4. Append one truthful JSON object to `/app/output/experiments.jsonl` per
   decision cycle. Required keys: `attempt_id`, `hypothesis`, `change`,
   `command`, `source_diff_sha256`, `run_kind`, `gpu_count`,
   `optimizer_updates`, `packed_tokens`, `status`, `metrics`. Use
   `run_kind: "full_budget"` only for the exact 8,000-update lane; statuses
   include `rejected`, `failed`, and exactly one final `selected`.
5. Keeping `/app/output/VIDEO_POINTING_REPORT.md` up to date is encouraged
   (what mechanisms mattered and why, citing `attempt_id`s; what didn't
   work) — it is your research notebook, not a scored deliverable. The
   ledger remains the source of truth.
6. Select exactly one completed within-budget checkpoint, then stage and
   audit:

   ```bash
   python /task-tools/mvp_task_tool.py stage \
     --checkpoint /app/output/attempts/<selected>/save/<step-dir>
   # Write /app/output/provenance.json; model_config must describe the
   # exact submitted config.yaml.
   python /task-tools/mvp_task_tool.py audit --output-root /app/output
   python /task-tools/mvp_async_run.py complete-control
   ```

If the outer scheduler job is ever interrupted, the run resumes in place:
in-flight training continues from the latest saved step with identical data
order (`allow_resume=True` is the repo default), and your session and
ledger carry over. Judge resumed attempts by their real
status/checkpoint/log evidence.

## Final reward

The Reward-Integrity Gate runs first; a confirmed hard violation or
unresolved audit receives reward 0. Otherwise the frozen verifier loads
your submitted checkpoint **from your workspace code** via the recorded
config, checks the invariants and the parameter budget, and evaluates it
under the frozen release protocol (`launch_scripts/eval.py`, val splits in
full, `device_batch_size=1`, `max_new_tokens=2048`, seed 6198, world size
8, frozen evaluator code). With `A = correct` on
`vixmo_points_count_clip_63s:val` and `F = f1` on
`vixmo_points_point_eval:val`:

```text
video_point_score = (A + F) / 2
reward = video_point_score
```

Baseline reward is TBD (pristine reproduction pending). Scientific
reporting must include the raw per-task metrics (`correct`, `close`,
`precision`, `recall`, `f1`); do not optimize a rounded reward.

## Required outputs

- exactly one staged `submission/` checkpoint directory below
  `/app/output` (config.yaml plus model weights, staged by the task tool);
- `/app/output/submission-selection.json` from the frozen task tool;
- `/app/output/provenance.json` with `base_model`, `model_config`,
  `training_data`, `optimizer_updates`, `packed_tokens`,
  `training_command`, `upstream_commits`, `downloads`,
  `evaluation_commands`, `optimization_attempts`, `topology_mapping`,
  `web_search: "disabled"`;
- `/app/output/experiments.jsonl`, one row per decision cycle;
- useful logs under `/app/output/logs`.
