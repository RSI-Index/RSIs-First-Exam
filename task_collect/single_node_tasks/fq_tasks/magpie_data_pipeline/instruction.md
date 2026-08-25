# Improve the Magpie alignment data pipeline

Your goal is to produce a better supervised fine-tuning dataset for a frozen
Llama-3-8B student than the official Magpie SFT baseline. This is a **data
pipeline task**: the student, generator, tokenizer, training recipe,
evaluation prompts, response-generation settings, judge, and reward code are
not research variables.

## Your workspace

You may edit only `/app/project/data_research/`, including its
`pipeline.py`, `pipeline_config.toml`, and support modules. The allowed data
sources are the local official Magpie baseline snapshot and the local pinned
70B Magpie generator. You may select, transform, regenerate, critique, rank,
filter, deduplicate, and balance that data.

Do not change or write `/models`, `/task-assets`, `/task-tools`,
`/opt/project`, or `/tests`. Internet access, external data, external APIs,
alternate checkpoints, evaluator access, and precomputed submissions are
forbidden. Source files may not exceed 10 MiB; do not use symlinks,
`sitecustomize.py`, path escapes, or embedded corpora.

## Fixed scientific contract

The candidate dataset must contain 1–300,000 ShareGPT conversations and no
more canonical Llama-3 chat-template tokens than the complete official
baseline. The root-owned `/task-assets/baseline_budget.json` gives the exact
token cap. Each record must have a unique ID and alternating non-empty
`human`/`gpt` messages ending in `gpt`.

The root verifier canonicalizes your artifact, then performs full SFT with the
following frozen recipe: Llama-3-8B base model, sequence length 8192, sample
packing, two epochs, validation fraction 0.001, micro-batch 1 on eight ranks,
accumulation 4 (global batch 32), paged AdamW 8-bit, learning rate `2e-5`,
cosine schedule, 100 warmup steps, weight decay 0, bf16, Flash Attention,
non-reentrant gradient checkpointing, response-only loss, and seed 42.

Evaluation uses root-only deterministic benchmark subsets and greedy
2048-token Llama-3 responses. A local Qwen judge samples with temperature
1.0, top-p 0.95, top-k 20, min-p 0.0, presence penalty 1.5, repetition penalty
1.0, and a 32768-token limit. Every candidate/baseline pair is judged in both
orders. Hidden prompt contents and identities are not available to you.

## Same-session experiment loop

Start a falsifiable attempt with a unique lowercase slug ID:

```bash
attempt=filtered-prefix-001
/task-tools/magpie_async_run submit \
  --attempt-id "$attempt" \
  --hypothesis "a specific data mechanism and expected result"
/task-tools/magpie_async_run status --attempt-id "$attempt"
```

Keep the session alive until the selected attempt completes. Every attempt is
immutable and must be recorded, including failures. After selecting one
completed attempt, materialize it and close the loop:

```bash
/task-tools/magpie_async_run select --attempt-id "$attempt"
/task-tools/magpie_async_run complete-control
```

The final regular files in `/app/output` must be:

- `dataset.jsonl`
- `dataset_manifest.json`
- `pipeline_config.toml`
- `provenance.json`
- `experiments.jsonl`
- `submission-selection.json`
- `control-complete.json`

The selected `pipeline_config.toml` is preserved byte-for-byte. Provenance
binds it and every ordinary file below the current `data_research/` tree to a
canonical manifest and SHA-256. Every attempt directory must have exactly one
matching `completed` or `failed` experiment event; selection is recorded as a
separate event for the chosen completed attempt.

The verifier runs as root after you exit. Missing artifacts, malformed or
over-budget data, policy violations, training failures, invalid judging, or a
non-finite reward fail closed to zero. Read `policy.yaml` before experiments.
