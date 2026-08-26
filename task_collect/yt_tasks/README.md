# Scale AutoResearch tasks

This directory contains the current Harbor 0.18 task format. It is intentionally
independent of `../tasks_oldformat`.

Every task has exactly these public parts:

```text
<task>/
├── task.toml
├── instruction.md
├── policy.yaml
├── environment/
│   ├── Dockerfile
│   └── project/        # pinned upstream source, used as the agent workspace
└── tests/
    ├── Dockerfile      # clean, separate verifier image
    ├── test.sh
    ├── policy_check.py
    └── ...             # public official-eval launcher and reward mapping
```

There is no `solution/`, private benchmark, copied baseline tree, or hidden
finished checkpoint. The writable training/decode workspace starts from pinned
upstream source. The verifier harness is nevertheless task-authored: it chooses
the public qualification workload, wraps official evaluator paths with trusted
timing/correctness/token checks, and maps raw metrics to a custom scalar reward.
It must not be described as an unmodified upstream or paper evaluator.
The final verifier runs separately, derives a diff against a clean upstream
checkout, applies deterministic checks, and then calls an LLM policy judge
before running the metric. An `UNCERTAIN` primary audit triggers an independent
second audit (optionally with `POLICY_JUDGE_MODEL_SECONDARY`). Only `PASS` enters
metric evaluation; a confirmed `FAIL` or unresolved `UNCERTAIN` receives reward
zero.

The five current tasks are `gated_deltanet`, `gated_deltanet_autoresearch`,
`deepspec`, `deepocr`, and `rlm`.
Task design follows the pinned author repository whenever it publishes a
runnable setup; papers are not used to fill missing repository settings. Read
[`SETUP_FIDELITY.md`](./SETUP_FIDELITY.md) for upstream/current-environment
differences and every disclosed patch, and [`RUN_CONTRACT.md`](./RUN_CONTRACT.md)
for assets, GPUs, trajectories, scientific-attempt counting, and validation
status. The persistent repository-first decision is in
[`DESIGN_MEMORY.md`](./DESIGN_MEMORY.md); its instruction-writing consequences
are in [`INSTRUCTION_DESIGN.md`](./INSTRUCTION_DESIGN.md).

The retired FLA/Flame Gated DeltaNet package is preserved as
`gated_deltanet_old`; it is audit history, not an active task or baseline. The
active `gated_deltanet` package uses the public NVlabs repository and its
released H1 0.4B / 15B SlimPajama recipe.

`gated_deltanet_autoresearch` is an independent fixed-time research task; it
does not alter or supersede the release-lane task. It establishes H1 live,
runs every candidate for 1,200 measured seconds on the same 4×8 H100
allocation, and requires controlled mechanism probes before evidence-led
synthesis. Its claims are hardware-local rather than matched-token or
matched-parameter claims.

The former cost-aware RLM package is preserved as `rlm_old`. The active `rlm`
package uses the repository-published OOLONG training config unchanged with a
compatible prime-rl v0.5.0 commit and ranks checkpoints by raw OOLONG reward;
subcall and iteration counts are diagnostics only.

The verifier needs `POLICY_JUDGE_MODEL` and the corresponding LiteLLM provider
credentials. This is deliberate: the reward-integrity gate fails closed when a
judge is unavailable. `POLICY_JUDGE_SKIP_LLM=1` exists only for task-author
static smoke tests and must not be set in scored runs.

Large checkpoints and public datasets are staged before a full run under
`/proj/datasets/interns/yuetai/agent_envs/scale_autoresearch_tasks`. The agent
and evaluator run with Hugging Face offline mode and may not replace missing
assets by downloading alternatives. `run_contract.py preflight` fails before
GPU submission when a required staged asset is absent.
