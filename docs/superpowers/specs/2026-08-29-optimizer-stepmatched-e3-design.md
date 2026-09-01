# E3-Only Step-Matched Optimizer Harbor Task Design

## Objective

Create a new RSI-Harness-compatible Harbor AutoResearch task that preserves the
optimizer update-geometry question from
`optimizer_update_geometry_stepmatched_v2` while evaluating only the E3 rung:
the 1,384,584,448-parameter d1792-L18 model trained for 40,283 updates on
10,559,946,752 tokens.

## Package and compatibility boundary

The new package is
`task_collect/zf_tasks/pre-training/optimizer_stepmatched_e3`. The existing
six-rung source package remains byte-for-byte unchanged. The task contains no
LSF, `blaunch`, hostname, Blue Vela, or Apptainer orchestration. Operators select
Blue Vela with `rsi-harness run ... --cluster bluevela`; RSI-Harness reads the
standard resource declarations and chooses its existing multi-node path.

Work and Judge are independently preallocated. Work declares 32 GPUs, which is
four eight-GPU Blue Vela nodes. Judge declares 8 GPUs, which is one node. E3
training uses one 32-rank, four-node torchrun. Judge reloads that distributed
checkpoint on eight ranks. Paloma is model-only; optimizer-state reload and the
fixed-window loss replay must prove that the checkpoint can be resharded from
32 ranks to 8 ranks.

## Scientific contract

The candidate supplies one scale-independent optimizer update rule through the
existing typed `build_megatron_optimizer` interface. E3 keeps context 4096,
TP=PP=CP=1, pure data parallelism, seed 0, GBS64/MBS2, the locked WSD schedule,
model construction, initialization, objective, tokenizer, data order, and
40,283-update endpoint.

One scored candidate consists of a full mechanism run and a reduced-mechanism
control run. Both use E3, exact source snapshots, complete distributed
checkpoints, and the same declared changed-path inventory. Judge performs six
requests: full/reduced Paloma reloads, full/reduced optimizer-state reloads,
and full/reduced fixed-window loss replays.

The fixed-window metric is the token-normalized mean over 16,777,216 tokens
ending at update 40,283. Judge starts from the last durable checkpoint at or
before the window and performs the authoritative replay on its eight-GPU
topology. A baseline certificate is valid only when it was produced by this
same 32-GPU Work to 8-GPU Judge protocol; the old 32-GPU loss must not be
silently reused as an eight-GPU replay result.

For baseline Paloma BPB Bp and fixed-window loss Bt, candidate values Cp and Ct,
the single-rung reward is `sqrt((Bp / Cp) * (Bt / Ct))`. Policy, submission,
checkpoint reload, source identity, novelty, and mechanism-ablation gates must
pass. Candidate-caused invalid submissions receive 0.0. Infrastructure,
scheduler, container, timeout, evaluator, and ambiguous incomplete failures
write no reward.

## Data and environment

E3 uses the prepared 12.413B-token `adamh-v6-pretokenized-12b` tree, the pinned
Llama 3.1 tokenizer revision, and Judge-only Paloma revision already evidenced
by the source package. E4/E5 training views and their task asset declarations
are excluded. The image keeps the pinned Megatron-Bridge commit
`9e5b537db01db83e94264e3f6f35e9ba89101e41`, Megatron-LM commit
`58bf14e9e68915a5e0c7d70451620e6e713c07d5`, and the rank-symmetric lm-eval
metric-gather compatibility patch.

## Validation and live completion

Generation follows test-first contract changes. Static task tests must prove
that only E3 is accepted, E3 needs 32 Work GPUs, Judge needs 8 GPUs, full and
reduced E3 evidence is required, 32-to-8 reload is declared, and E0/E1/E2/E4/E5
cannot be staged or scored. The package must pass shell/Python syntax, source
locks, the bundled task suite, the Harbor static validator, RSI-Harness
compilation, and an independent read-only review.

Live validation then uses the Blue Vela adapter. The one-time temporary runtime
profile may use the `priority` queue and non-exclusive hosts as explicitly
authorized for this debug run, but the shared profile is restored to its normal
queue and exclusive-host policy immediately after submission. Completion
requires an actual four-node Work allocation, one-node Judge allocation,
successful 32-rank E3 training/checkpoint creation, successful 8-rank Judge
reload/replay, a finite terminal reward, native RSI-Harness final artifacts,
and no regression in the single-node Harness suite.
