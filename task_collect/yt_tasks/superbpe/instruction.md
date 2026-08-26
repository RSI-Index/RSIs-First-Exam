# Improve SuperBPE encoding efficiency at fixed data and vocabulary

Your goal is to produce a **deterministic, lossless 200K BPE tokenizer with
higher held-out bytes per token than the released SuperBPE t80 tokenizer**.
Use the pinned repository and your own measurements to understand which
boundary-relaxation and merge-allocation choices matter.

SuperBPE is the reference method, not a binary method gate. A different merge
objective, transition schedule, or tokenizer-training implementation is valid
if it obeys the fixed scientific contract and emits the required BPE artifact.
Do not use papers, web search, remote documentation, external code, or external
data during the run.

## Setup

| Item | Path / value |
|---|---|
| Writable workspace | `/app/project`, pinned `PythonNut/superbpe@bbd09768fc28a875cef48e6bdd66e3a17454628e` |
| Custom tokenizer fork | `tokenizers_superbpe`, pinned submodule `757f2a55c0820ed47064e1fe473deea39b7b611b` |
| Released baseline | `/app/project/tokenizer_json/olmo2_p99_truncate_10G_80K_extend_200K_mw4_colon/tokenizer.json` |
| Training data | `$SUPERBPE_TRAIN_DIR`; the exact repository `meta.json` order, with the final fixed prefix available under `$SUPERBPE_DERIVED_DIR` |
| Full data budget | at most 10,000,000,000 training bytes per candidate |
| Base vocabulary | exactly 200,000 BPE model entries; the released five added tokens and IDs are fixed |
| Held-out evaluation | pinned public `eval/135.txt`, evaluator-only; never inspect, encode, train on, or derive rules from it |
| Raw metric | file bytes / token count; higher is better |
| Submission | `/app/output/submission/tokenizer.json`, provenance, and experiment ledger |
| Compute | CPU only; up to 64 cores and 512 GiB RAM for a full candidate |
| Network | disabled; staged data and repository are the complete scientific inputs |

The official 10GB order is `48, 61, 66, 25, 6, 58, 51, 75, 80, 32`, then
the first 557,641,266 bytes of `70.txt`. The task tool verifies and writes
this metadata without modifying read-only staged data.

## Fidelity boundary

The public repository releases tokenizer artifacts, training code, example
shell scripts, configs, and notebook outputs. It does **not** release an exact
command for the 200K/t80 artifact, the post-training tokenizer transformation
that produced its serialized pretokenizer, or the private 1GB `eval/135`
variant used by the notebook. Therefore:

- the scored baseline is the released t80 artifact, measured afresh on the
  pinned current public `eval/135.txt`;
- `/task-tools/superbpe_task_tool.py prepare-reference-shape` is a convenient
  repository-derived starting shape, not a claim that the missing author
  command has been reconstructed;
- downstream language-model claims are out of scope. This task supports a
  claim about tokenizer compression/encoding efficiency only.

## Action space

You may change the tokenizer-learning method and its implementation broadly:
merge scoring/objective, the amount or schedule of inherited BPE merges,
single- or multi-stage boundary relaxation, pretokenization, deterministic
corpus ordering or weighting within the fixed corpus, pruning/reallocation,
and Python or Rust code in the workspace. You may start from any released
in-repository tokenizer artifact.

Frozen constraints are deliberately narrower: no external or synthetic data;
no held-out use; no more than 10GB of training text per full candidate; exactly
200,000 base BPE vocabulary entries; the released five added tokens at IDs
200000–200004; deterministic encoding (no dropout); and byte-exact round-trip
decoding. The final artifact must be a Hugging Face `tokenizers` BPE JSON and
must be at most 64 MiB. These constraints make candidates comparable and keep
the reward path honest; they do not require a particular SuperBPE mechanism.

## Measured baseline

The immutable current-environment reference on the pinned 944,259,457-byte
public eval file is 141,911,558 tokens, or **6.653858715299285 bytes/token**.
The matched baseline reward is 0.5. Repository notebook values are historical
upstream outputs from a different private 1GB eval file, not this task's
baseline. Always compare candidates with the verifier's full-precision raw
`bytes_per_token`, not a rounded reward.

## Experiment loop

One scientific full run is one independently produced tokenizer that processes
at most the fixed 10GB corpus. Smaller prefixes are screening runs unless the
hypothesis explicitly studies data efficiency; a selected sub-budget result
must be labeled with its exact byte count and may not be described as a 10GB
reproduction.

For each attempt:

1. State one falsifiable hypothesis and the expected bytes/token effect.
2. Change one interpretable variable group and record the exact source diff.
3. Train or construct the tokenizer. The repository entrypoint remains:

   ```bash
   python /task-tools/superbpe_task_tool.py prepare-reference-shape \
     --output-dir /app/output/attempts/tok-001/work --transition-merges 80000
   cd /app/project
   python train_tokenizer.py \
     --output_dir /app/output/attempts/tok-001/work \
     --vocab_size 200000 \
     --regex_string '\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)'
   ```

   This is a starting point, not a locked recipe. Inspect the repository and
   change it when your hypothesis requires doing so.
4. Append one truthful JSON object to `/app/output/experiments.jsonl` with:
   `attempt_id`, `hypothesis`, `change`, `command`, `source_diff_sha256`,
   `run_kind`, `training_bytes`, `status`, and `metrics`. Use `run_kind:
   "full_budget"` only when the accounting is exactly 10GB. Record failed and
   rejected attempts too. Exactly one completed row must end as `selected`.
5. Use only training-side diagnostics for selection. Never inspect or call the
   held-out evaluator yourself.
6. Stage exactly one selected tokenizer:

   ```bash
   python /task-tools/superbpe_task_tool.py stage \
     --attempt-id <selected> \
     --tokenizer /app/output/attempts/<selected>/work/tokenizer.json
   # Write /app/output/provenance.json, then:
   python /task-tools/superbpe_task_tool.py audit --output-root /app/output
   ```

## Final evaluation and reward

The Reward-Integrity Gate runs first. A confirmed hard violation or unresolved
audit receives reward 0. The frozen evaluator then checks the artifact schema,
fixed vocabulary and added tokens, determinism, size, and byte-exact decoding;
it counts tokens with the repository's paragraph-chunk protocol on the pinned
held-out file. It evaluates the released t80 artifact in the same process.

Let `C` be candidate bytes/token and `B` the matched released baseline. The
bounded scalar reward is `C / (C + B)`, so the baseline is exactly 0.5 and
higher raw compression receives higher reward. Scientific reporting must lead
with `C`, token count, file bytes, and the `C/B` ratio.

## Required outputs

- `/app/output/submission/tokenizer.json` (at most 64 MiB);
- `/app/output/submission-selection.json` from the task tool;
- `/app/output/provenance.json` with `base_tokenizer`, `training_data`,
  `training_bytes`, `training_command`, `upstream_commits`,
  `dataset_revision`, `downloads`, `evaluation_commands`,
  `optimization_attempts`, and `web_search`;
- `/app/output/experiments.jsonl`, one row per decision cycle and exactly one
  selected attempt;
- useful training logs under `/app/output/logs`.
