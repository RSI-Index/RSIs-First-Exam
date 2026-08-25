# Open-ended RL discovery for deep research agent

## Research Question

Starting from the immutable `osunlp/QUEST-35B-MT-Plus-SFT` checkpoint,
discover an RL-only post-training method whose fixed-hidden-subset accuracy
strictly exceeds the official `osunlp/QUEST-35B-RL` reference on both
BrowseComp and HLE-Text under the locked QUEST deep-research evaluation
harness.

The candidate may choose the RL algorithm, training reward, reward model or
verifier, data, data-generation method, search environment, environment
mixture, curriculum, rollout strategy, credit assignment, optimizer,
regularization, and hyperparameters. The training reward is candidate-owned
and may change during research.

All submitted policy-weight changes must result from reinforcement-learning
updates beginning at the frozen MT+SFT checkpoint. Supervised
token-likelihood training, DPO, rejection-sampling fine-tuning, model merging,
and importing policy weights from another checkpoint are not valid
interventions. Auxiliary models may generate tasks or compute rewards but may
not contribute weights to the submitted policy.

## Reference Baselines

The immutable starting policy is `osunlp/QUEST-35B-MT-Plus-SFT` at revision
`a15c5e088bac3410f5918480d2dec47fb4b28789`. The score to beat is the official
`osunlp/QUEST-35B-RL` checkpoint at revision
`cfe1bd34132816e76bbbd338cddfcef8ba289c1c`. The task source is derived from
`OSU-NLP-Group/QUEST` commit
`962e10b6b99f53efc8fd45229ed80728b9ee9a05`.

The QUEST reference uses a fully asynchronous VERL pipeline, GRPO-style
outcome optimization, eight responses per prompt, session-level advantage
propagation, and no KL penalty. Its training data contain 864 objective tasks
and 269 open-ended tasks. The paper uses 32 H100 GPUs split into 16 training
and 16 rollout GPUs, prompt length 24,000, response length 12,288, eight
rollouts per prompt, at most 100 tool turns, and 80 asynchronous steps. These
choices are context rather than constraints.

### Baseline metrics

The published model-card values below are contextual only. They use the model
card's stated `avg@3` protocol and are not trusted thresholds for the hidden
fixed-subset evaluation. Missing values may not be fabricated, projected,
interpolated, or silently excluded.

| Baseline | Metric | Value | Unit | Direction | Evaluation unit or scale | Status |
|---|---|---:|---|---|---|---|
| `osunlp/QUEST-35B-RL` | BrowseComp `avg@3` accuracy | 45.5 | percent | maximize | published model-card protocol | contextual; differently scored |
| `osunlp/QUEST-35B-RL` | HLE `avg@3` accuracy | 37.9 | percent | maximize | published model-card protocol | contextual; differently scored |

No baseline training curve was supplied, so the baseline manifest records an
empty `training_curves` list.

## Evaluation and success

### Primary metric

The final report contains exactly two primary benchmark metrics:

- `avg@3` accuracy on one fixed hidden 130-example BrowseComp subset; and
- `avg@3` accuracy on one fixed hidden 130-example HLE-Text subset.

Each example receives exactly three runs. Missing, timed-out, malformed, or
unscored runs count as incorrect, and incomplete benchmark runs fail validity.
The candidate succeeds only if both final accuracies strictly exceed the
corresponding accuracies of the frozen official QUEST-35B-RL reference under
the same pinned harness. A gain on one benchmark cannot compensate for a tie
or regression on the other.

### Secondary metrics and validity gates

There is no benchmark-based research screen. Select and freeze the candidate
using candidate-owned, benchmark-isolated development tasks and diagnostics.
At handoff, evaluate the frozen candidate once on both fixed hidden subsets and
rerun the official RL reference on those same examples.

Reference and candidate must use the same hidden-subset manifest, pinned
inference code, system prompt, tool schema, search and visit providers, context
and cache policies, judge, sampling, turn limit, token budget, retry policy,
and answer extraction. Validity gates cover exact model lineage, RL-only policy
updates, compute and deadline limits, benchmark isolation, complete
evaluations, unchanged final harness, reproducible reward and data provenance,
and independent checkpoint reload.

Operational metrics such as malformed calls, timeouts, search and visit calls,
generated tokens, latency, API cost, and training compute must be recorded for
audit, but they do not change the two benchmark scores or prescribe the
candidate's training reward.

BrowseComp and HLE questions, answers, paraphrases, translations,
answer-derived prompts, benchmark-specific facts, and hidden outcomes may not
be used for training, reward design, retrieval-corpus construction, data
generation, candidate selection, or debugging.

### Reward

When every validity gate passes and the candidate strictly improves on both
benchmarks, the trusted scientific reward is the arithmetic mean of the two
candidate accuracies. Otherwise it is zero. The Layer 1 reward only reports
whether the portable environment and artifact contract pass; it is never a
scientific benchmark score.

## Workspace

The candidate receives the complete pinned QUEST source at `/app/project`
except that the entire upstream `evaluation/` directory is deliberately
absent. This prevents evaluator code and benchmark assets from entering the
candidate environment. `/app/project` is editable; Task Tools and verifier
files are protected by `policy.yaml`.

| Runtime input | Path or immutable identity | Owner/access | Layer 1 status |
|---|---|---|---|
| Redacted codebase | `/app/project`; `OSU-NLP-Group/QUEST@962e10b6b99f53efc8fd45229ed80728b9ee9a05` | candidate-readable/editable | materialized and hash-locked |
| Task Tools | `/task-tools` | candidate-readable, read-only | available |
| Durable output | `/app/output` | candidate-readable/writable | available |
| Training data | `osunlp/QUEST-RL-Data`, full-run mount `/task-data/train` | candidate-readable | external-unverified; no immutable dataset revision published |
| Starting checkpoint | `osunlp/QUEST-35B-MT-Plus-SFT@a15c5e088bac3410f5918480d2dec47fb4b28789`; full-run mount `/task-data/checkpoints/starting` | candidate read-only | external-unverified |
| Baseline checkpoint | `osunlp/QUEST-35B-RL@cfe1bd34132816e76bbbd338cddfcef8ba289c1c`; full-run mount `/task-data/checkpoints/baseline` | verifier-only | external-unverified |
| Hidden evaluation data | fixed 130-example BrowseComp and HLE-Text subsets; full-run mount `/task-data/eval` | verifier-only | unresolved outside Layer 1 |
| Trusted evaluator | upstream `evaluation/browsecomp/eval.py` and `evaluation/hle/judge.py` at the pinned commit, wrapped by a verifier-private fixed-subset adapter | verifier-only | interface identified; full assets unresolved |
| Verifier evidence | `/logs/verifier/reward.json` and `/logs/verifier/reward.txt` | verifier-owned | available in Layer 1 |

The pinned full-run training entrypoint is:

```bash
cd /app/project/training_scripts/rl
bash recipe/deepresearch/run_deepresearch_fully_async_megatron.sh
```

This command requires the separately provisioned 32-GPU backend, model and
data mounts, search services, and runtime dependencies; it is not executed by
the portable Layer 1 task. The candidate-visible Layer 1 evaluator is:

```bash
python3 /task-tools/evaluate.py --run-dir /app/output/attempts/<attempt-id>
```

The scientific evaluator is verifier-private. It invokes the two pinned
upstream evaluator interfaces against the hidden fixed-subset manifest and
does not return questions, answers, per-example outcomes, or judge traces to
the candidate.

## Persistent experiment loop

For every attempt, first write a unique JSON hypothesis under
`/app/output/hypotheses/<attempt-id>.json` with matching `attempt_id`. Then use:

```bash
python3 /task-tools/preflight.py --attempt-id <attempt-id> --hypothesis-file /app/output/hypotheses/<attempt-id>.json
python3 /task-tools/run_candidate.py --attempt-id <attempt-id> --hypothesis-file /app/output/hypotheses/<attempt-id>.json --dry-run
python3 /task-tools/evaluate.py --run-dir /app/output/attempts/<attempt-id>
python3 /task-tools/task_state.py record --run-dir /app/output/attempts/<attempt-id>
python3 /task-tools/task_state.py stage --run-dir /app/output/attempts/<attempt-id>
```

The `--dry-run` path verifies portable plumbing only. A non-dry invocation
fails explicitly with `FULL-RUN-UNVERIFIED` until a separately validated
backend adapter is supplied. Use `task_state.py summarize`, `restore`, and
`prune` to manage durable attempts; staged attempts cannot be pruned.

## Research deadline and handoff

The full research budget is 24 hours with at most 256 concurrent H100 GPUs and
6,144 aggregate H100-hours. A reference-scale experiment uses 32 H100 GPUs.
These are scientific full-run limits recorded as metadata, not resources
requested by Layer 1. Freeze one candidate before final evaluation. Handoff
must identify the exact checkpoint, source revision, reward and data
provenance, experiment ledger, resource accounting, and independently
reloadable inference configuration. Full scientific validation remains
`external-unverified` until the private evaluator and required mounts are
provisioned and run.

## Required artifacts

Layer 1 requires `/app/output/submission.json`,
`/app/output/staged_candidate.json`, `/app/output/experiments.jsonl`, and the
staged attempt's `preflight.json`, `candidate_result.json`, and
`evaluation.json`. Every Layer 1 artifact must state
`validation_mode: layer1-smoke` and `scientific_validation: not-run` where the
schema provides those fields.

A full-run submission additionally requires the frozen policy checkpoint,
checkpoint and source digests, RL-only lineage evidence, complete training and
evaluation configuration, reward implementation, data manifest and licenses,
per-attempt ledger, compute and deadline accounting, operational metrics, and
the verifier-private candidate/reference evaluation reports. Layer 1 does not
manufacture or substitute any of those scientific artifacts.
