# SuperBPE SOP Step 1--5 audit

Audit date: 2026-07-23. Step 6 (the Codex/Harbor optimization trajectory) has
not been authorized or started. This document separates what the pinned public
repository actually releases from task infrastructure and from historical
outputs embedded in notebooks.

## Decision summary

The proposed task is to improve deterministic, lossless held-out bytes per
token over the repository's released 200K SuperBPE t80 tokenizer while fixing
the repository's exact 10GB tokenizer-training corpus and the 200K vocabulary
budget. This is a scientifically meaningful tokenizer-learning problem, not a
downstream language-model pretraining claim.

The Harbor package, pinned data, role-separated views, two SIFs, evaluator,
policy gate, and CPU training/evaluation launchers are built. A 64MiB
two-stage trainer smoke passed. The immutable full released-artifact baseline
job 324917 completed successfully; its result is recorded below.

Do not start Step 6 automatically. The exact official 200K/t80 training command
and final serialized-pretokenizer transformation are absent from the public
repository, and a full 10GB/200K candidate training run has not yet been timed.
That resource uncertainty requires the requested user review.

## Prior Codex session applied

The requested session was resolved from `/u/yuetai/.codex/session_index.jsonl`
as `019f87d8-ead2-7970-a5f4-eeba1397d8d1`, titled
“查找Codex环境配置session”. Its design guidance was applied here: preserve a
broad, scientifically reasonable action space; treat the repository method as
a preference/baseline rather than a binary gate; and freeze only the data,
budget, evaluator, comparability, and reward-integrity constraints that are
actually necessary. The Gated DeltaNet task was used only as a Harbor packaging
and audit-structure reference, not as a template for SuperBPE's scientific
variables, compute topology, or reward.

## Step 1: repository-only source audit

Scientific source of truth:

- `PythonNut/superbpe@bbd09768fc28a875cef48e6bdd66e3a17454628e`
  (`v1.0.0`), vendored cleanly under `environment/project`;
- its declared submodule
  `alisawuffles/tokenizers-superbpe@757f2a55c0820ed47064e1fe473deea39b7b611b`;
- main repository MIT license and tokenizer fork Apache-2.0 license.

No paper result was used to choose the setup, baseline, data, objective, or
budget. The repository contains tokenizer training/encoding code, example
scripts, tokenizer artifacts, configs, and notebook outputs. The selected
released artifact is:

```text
tokenizer_json/olmo2_p99_truncate_10G_80K_extend_200K_mw4_colon/tokenizer.json
SHA256 0a9b5dda03c25cd1dbc2c44b48695694548f4a921b13b3a092c0e1be355cd608
```

It has 200,000 BPE model entries, 199,757 serialized merge entries, and five
released added tokens at IDs 200000--200004.

The repository's `meta.json` files identify
`UW/olmo-mix-1124-subset-p99`. The task pins public dataset revision
`64b9a7c502482035602810c4e9acda1c1dd21908` and the exact repository order:

```text
train/{48,61,66,25,6,58,51,75,80,32}.txt in full,
then the first 557,641,266 bytes of train/70.txt
```

The total is exactly 10,000,000,000 bytes. The task manifest hashes all source
files and the materialized final prefix:

```text
/proj/datasets/interns/yuetai/agent_envs/scale_autoresearch_tasks/
  asset_manifests/superbpe-official-64b9a7c5.json
```

The current public `eval/135.txt` is 944,259,457 bytes with SHA256
`81e2f413b434b5a89090f9cfeacd3ea01289bbba7439d2eabc0c6f20434b6784`.
It is evaluator-only. Hard-linked views prevent the agent from seeing it: the
agent view has exactly the eleven 10GB training objects and the verifier view
has only `eval/135.txt`.

## Step 2: autoresearch task and scientific value

The raw metric is held-out file bytes divided by token count, higher is better.
For a fixed 200K vocabulary and fixed 10GB corpus, this directly tests how
merge allocation, boundary relaxation, and pretokenization change the amount
of text represented per token. Better compression is a concrete, falsifiable
tokenizer result; it does not by itself establish better downstream language
model quality.

The agent may change merge objectives/scoring, transition schedules, inherited
merge allocation, pretokenization, deterministic corpus ordering/weighting
within the fixed corpus, pruning/reallocation, and Python or Rust
implementations. The SuperBPE mechanism is a reference preference, not a hard
method gate. Frozen constraints are only those needed for comparison and reward
integrity: the exact data and 10GB cap, 200K BPE model entries, released added
tokens/IDs, deterministic lossless encoding, hidden evaluator, and a 64MiB
artifact limit.

The matched evaluator validates the JSON schema and fixed contract, performs
byte-exact round-trip checks on every held-out chunk, probes determinism, and
evaluates candidate and released t80 artifact in the same process. It reports
full-precision bytes/token and token counts. Harbor reward is the bounded,
monotone `C / (C + B)`; the released baseline is exactly 0.5. Raw bytes/token,
not the scalar mapping, remains the primary scientific result.

The policy makes held-out use, evaluator tampering, external solutions/data,
sample-specific behavior, budget/vocabulary bypass, external research, and
false provenance hard-zero violations. Candidate networking and web search are
disabled. The verifier has public control-plane networking only for the policy
judge.

## Step 3: resource estimate and review gate

This task needs no GPU. The Harbor allocation is up to 64 CPU cores, 512GiB
memory, and 128GiB writable storage for candidate training; the verifier uses
32 CPU cores and 128GiB. These are safety envelopes, not claims that the
trainer scales linearly to all cores.

Measured 64MiB/8K two-stage pilot (job 324882):

| Measurement | Result |
|---|---:|
| Stage 1 training | 7.9205 s |
| Stage 2 training | 86.1554 s |
| LSF wall | 104 s |
| LSF maximum memory | 5,782 MB |
| Maximum threads | 175 |

That pilot proves the compiled custom Rust trainer and both stages execute. It
does not measure a 10GB/200K full candidate. Simple extrapolations disagree by
more than an order of magnitude depending on whether data volume, merge count,
or their interaction dominates. Therefore the task is not declared to satisfy
the SOP single-run six-hour threshold. Step 6 needs user review or a separately
authorized full-training qualification first.

The repository's downstream 8B/11B pretraining path is not selected: it refers
to missing `scripts/train.py`, private `.npy` inputs, inconsistent config names,
and does not publish the topology/checkpoints/logs needed for exact
reproduction. Its approximately 8.115B-parameter, 331.69B-token recipe is also
far beyond this task's tokenizer-training resource envelope.

## Step 4: reproduction and infrastructure qualification

### Released-artifact reference

The reproducible repository baseline is the released t80 tokenizer artifact,
evaluated afresh on the pinned current public `eval/135.txt`. The immutable
run freezes its launcher and a clean copy of the pinned project in `control/`,
records their `SHA256SUMS`, and records the SHA256 of the tests SIF that already
contains the evaluator, scorer, and policy implementation before evaluation:

```text
LSF job 324917
/proj/datasets/interns/yuetai/agent_envs/scale_autoresearch_tasks/
  reference_runs/superbpe/20260723T110500Z-released-t80-frozen
```

Final status: **DONE**, exit 0. On 944,259,457 bytes and 21 upstream-style
paragraph chunks, both candidate and reference produced 141,911,558 tokens:

| Full reference metric | Result |
|---|---:|
| bytes/token | 6.653858715299285 |
| candidate/reference ratio | 1.0 |
| worst chunk token ratio | 1.0 |
| candidate encoding | 512.4190 s |
| reference encoding | 505.5102 s |
| verifier elapsed | 1,115 s |
| LSF maximum memory | 13.4 GB |
| effective CPU peak | 1.00 core |
| reward | 0.5 |

All chunks passed byte-exact decoding and the runtime determinism probes. The
artifact and reference SHA256 were identical (`0a9b5dda...`). The author-side
policy run used the static integrity checks only and found no violations; the
LLM policy judge is reserved for untrusted Step-6 candidate changes.

Compared with the repository notebook's historical t80 value of 6.634021, the
current public-file result is higher by 0.0198377153 bytes/token, or 0.29903%.
This is an evaluation-file difference, not a tokenizer improvement: the
artifact is byte-identical and the repository notebook points to a private
1,000,000,000-byte path.

A prior full run, job 324862, was deliberately killed and marked invalid after
the live launcher was edited while the process was running. Its partial chunks
cannot become baseline evidence. A 64MiB author smoke evaluated the released
artifact against itself and obtained 10,111,958 tokens,
6.636584526953138 bytes/token, ratio 1.0, and reward 0.5; it only validated the
evaluator path and is not the full reference.

### Two-stage trainer path

Job 324882 completed both custom trainer stages on 64MiB. Stage 2 produced
7,784 merge entries. Of 4,000 requested inherited merges, 3,997 occurred
somewhere but only 1,292 formed a literal leading prefix. The initial wrapper
incorrectly required exact prefix preservation and exited 1 after successful
training. Validation-only recovery job 324889 passed after the assertion was
changed to report the trainer's actual semantics.

This is not merely a wrapper curiosity. The pinned Rust trainer loads
`merges.txt`, skips tokens containing `:Ġ`, and rejects new tokens spanning more
than four nonempty whitespace-marker words. Those checks can skip inherited
merges and change later merge availability. Exact-prefix preservation is not a
valid upstream invariant.

## Step 5: complete task setup and fidelity gaps

Harbor package:

- `task.toml`: task identity, resources, images, role-separated data roots;
- `instruction.md`: full action space, fixed contract, experiment ledger, and
  output protocol visible to every agent iteration;
- `policy.yaml`: Reward-Integrity rules;
- `environment/`: clean pinned source, custom task tool, and build definition;
- `tests/`: independent policy check, schema/losslessness evaluator, scorer,
  and verifier image;
- `lsf_validation/`: Apptainer definitions and immutable LSF reference/trainer
  qualification launchers.

Environment SIF:
`9bd0a77643c6549668d6c4736b5ebf4078ed40da716aa6fdfb925ea10e275a2f`.
Tests SIF:
`39f5750ad83022c43046967559bd060c43e1d30266d870029ce3d4b6b09b56c1`.
Both use Python 3.12.9 and tokenizer encoding version 0.20.1. The environment
uses Rust 1.84.1 and the pinned custom tokenizer fork. `ai2-olmo` is omitted
because the selected tokenizer-only entrypoints do not import it. Top-level
dependencies are pinned, but not every transitive dependency has an upstream
lockfile; the SIF hashes are the actual environment identities.

Known repository/current-environment discrepancies:

1. Repository notebook values use a private
   `/gscratch/.../eval/135_truncated_1000000000.txt`, not the current public
   944,259,457-byte file. Historical embedded t80 output is 6.634021
   bytes/token; it is not the current baseline.
2. The example extension script is a 128K/100K example, not the exact
   200K/t80 command. The released t80 serialized pretokenizer matches neither
   the committed Stage 2 script nor `construct_hf_tokenizer`; the final
   author-side transformation is missing.
3. `prepare-reference-shape` hashes all 10GB inputs and creates the repository
   order plus an 80K merge starting shape, but it is explicitly infrastructure,
   not a reconstructed official command.
4. The verifier discards token-ID lists after each repository-style paragraph
   chunk to bound memory. This preserves token counts and bytes/token while
   adapting the evaluation to the available host.
5. The tests image uses standard `tokenizers==0.20.1` for encoding. The custom
   fork's changes are in the trainer, not the runtime encoding path; the
   submitted JSON and reference JSON are both evaluated by the same library.

## Step 6 gate

No agent trajectory or optimization result exists yet. Before proceeding,
review these choices:

1. accept the tokenizer-only scientific claim and the released-artifact
   baseline despite the missing exact author training command;
2. accept the fixed 10GB/200K contract and broad method action space;
3. decide whether to first authorize a full 10GB/200K training qualification,
   or enter the async Harbor trajectory with an explicit conservative runtime
   envelope;
4. accept that this is CPU-only and that candidate single-run time is currently
   unmeasured rather than presumed under six hours.

The shared `submit_full_run.sh` intentionally does not yet list `superbpe`, so
Step 6 cannot be started accidentally. If approved, its CPU branch must use the
same asynchronous contract as requested: the agent submits a candidate train,
exits its turn, the host-side adapter monitors terminal state, and Harbor
resumes the same session. Adding a synchronous branch merely to make the task
name launchable would violate that operational contract and is not part of
Steps 1--5.
