# `nanovlm-finevision-self-selection` — RSI benchmark task, CHEAP TIER

> **CONFIGURATION ONLY.** This package contains no data, no weights and no vendored upstream
> source. Every dataset and checkpoint is declared by URL in `environment/assets.yaml` with
> `state: NOT_STAGED`. `environment/fetch_assets.sh` is the fetcher and it ships unexecuted.

> **CHEAP TIER, not an independent result.** This asks the same question as
> `vlmr1-rec-self-curriculum` at roughly a tenth of the cost. Reporting it as a third data point
> would be inflating one question into three. Its value is answering whether a chain effect at
> 8 GPU-hours per round survives at a fraction of that — which is the question anyone building on
> the result will actually have. See `contract.yaml` `relation_to_sibling_packages`.

## The question

Under a frozen SFT recipe, a frozen evaluator and a fixed 3000-optimizer-step budget, does an
agent that may rewrite its own **data-selection** improver — and that inherits its own trained
checkpoints — produce better selections round over round?

Six rounds. Each round the agent's selector picks which rows of a 1.5M-row FineVision prefix the
trainer sees; from round 2 on, it may score those rows with the checkpoint it produced in round
k−1. Headline is MMStar accuracy from a pinned lmms-eval, anchored on our own measured
no-selection baseline.

## Why this repo has a real experiment in it

Everything the task rests on is upstream code, not our construction:

| what | where |
|---|---|
| a data-selection baseline that the trainer already applies | `data/datasets.py` `BaseDataset._get_messages` — drops a conversation **turn** whose relevance / image-correspondence / visual-dependency / formatting rating is below a threshold |
| the four thresholds already wired to the CLI | `train.py:648-651` |
| the step budget enforced by upstream, twice | `train.py:364` loop condition, `train.py:590` break |
| a defined samples-seen quantity | `train.py:620` |
| one cost datapoint about itself | `README.md`: the 222M configuration reached 35.3% MMStar in ~6h on one H100 over ~1.7M samples |

The default thresholds keep everything, so "no selection" is the shipped baseline and an improver
has something concrete to beat.

## `/opt/project` is NOT bare upstream

It is `4e0c096` **plus `environment/rsi_adapter.patch`** — 7 change sites in 2 files, because the
selection channel does not exist upstream: `train.py` has flags for the four thresholds but none
for a row selection and none for the step budget. The patch adds an `rsi_idx` column, applies the
keep list to train **after** the val split, and adds the two flags.

Verified, not asserted: `git apply --check` against the pinned commit passes, the patched tree
byte-compiles, and `tests/upstream_check.py` re-checks the commit, the touched-file set, the
change-site counts and the val-split ordering at **both** image builds.

## Where the resources come from

| asset | URL | mount |
|---|---|---|
| vision backbone | `huggingface.co/google/siglip2-base-patch16-512` | `/models/siglip2-base-patch16-512` |
| language backbone | `huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct` | `/models/SmolLM2-360M-Instruct` |
| selection pool | `huggingface.co/datasets/HuggingFaceM4/FineVision_concat_shuffled_2` | `/datasets/finevision_pool` (56 shards) |
| evaluator | `github.com/EvolvingLMMs-Lab/lmms-eval` | `/opt/lmms-eval` — **commit `PIN_REQUIRED`** |
| eval sets | resolved by the pinned lmms-eval | `/eval-data/lmms_eval_home` (verifier only) |
| source | `github.com/huggingface/nanoVLM` @ `4e0c096` | `/opt/project` |

Total is a **guess** (~250 GB) and labelled as one — nothing upstream states FineVision's size.
`fetch_assets.sh` measures what it wrote and records it in `staged.lock.yaml`; the gates read the
lockfile, never the guesses.

## Two things must be pinned before any number exists

`lmms-eval` is installed from an **untagged branch** in upstream's README, and nanoVLM's
`evaluation.py` is a copy of that project's `__main__.py` importing its internals. Unpinned, it is
a moving metric: an upstream change to a prompt template or answer parser shifts the headline
mid-chain and the shift gets credited to the improver.

So `contract.yaml` carries `evaluator_commit: PIN_REQUIRED` **and**
`metric.headline.source_metric: PIN_REQUIRED` — the metric key inside lmms-eval's results for the
mmstar task is a property of that repository, and this package will not guess it. Three independent
mechanisms stop a placeholder from reaching a score:

- `tests/Dockerfile` declares `ARG LMMS_EVAL_COMMIT` with **no default**, so the build fails
- `tests/contract_check.py` prints `EVALUATOR NOT PINNED`
- `tests/test.sh` gate 0 `infra_fail`s (exit **2**, not a zero)

## Validate it

```bash
source /proj/long-multi/hhua/.venv/bin/activate
pip install -r tests/requirements-checks.txt          # pyyaml, nothing else
../../common/sync_common.sh .                         # shared verifier files + tests/common.lock
python tests/contract_check.py --package .
python tests/upstream_check.py --project /path/to/patched/nanoVLM --contract contract.yaml
```

Expected while nothing is staged:

```
PACKAGE VALID, ASSETS NOT STAGED, EVALUATOR NOT PINNED, ANCHORS UNMEASURED
```

| checked | by |
|---|---|
| the five copies of the contract agree | `tests/contract_check.py` |
| `effective_batch` and `samples_seen` are the identity they claim | `tests/contract_check.py` |
| the patch sets every frozen-recipe override | `tests/contract_check.py` against the `.patch` text |
| upstream still has the cited symbols; the patch is the declared shape; the keep list is applied after the val split | `tests/upstream_check.py`, at both image builds |
| assets staged, by **count** not existence | `task-tools/asset_check.py`, `tests/evaluate.py`, `test.sh` gate 0 |
| the shared verifier files are neither edited nor stale | `tests/common.lock` + `common/` comparison |
| `reference/` cannot reach an image | `.dockerignore` + two Dockerfile checks + no whole-context `COPY` |
| the null chain's selection was constant **and** its scores varied | `reference/assert_null_chain.py` |

## Order of operations

```
1. fetch assets, record the lockfile, flip assets.yaml to STAGED
2. pin lmms-eval; fill evaluator_commit and source_metric from ITS output
3. launch gate: pool_rows >= 3 * 384000                       (staging-time, no GPU)
4. anchors: pristine x3 seeds, reference improver x3 seeds; sigma = max(3-seed std, binomial floor)
5. launch gate: anchors_separated, then signal_resolvable      <-- either may fail; a failure is a result
6. null chain x3 seeds, then reference/assert_null_chain.py
7. only then open it to agents
```

## Known risks, each encoded as a gate rather than prose

| | risk | gate |
|---|---|---|
| NR-1 | 3000 steps may be too few for the headline to see data selection at all | `launch_gates.signal_resolvable` — compares the best selection upstream can express against none |
| NR-2 | the evaluator is external and unpinned | `launch_gates.evaluator_pinned` + the no-default ARG + gate 0 |
| NR-3 | `/opt/project` is not bare upstream and a reader may assume it is | `pristine_base.patch` names every change site; `upstream_check.py` enforces it |
| NR-4 | MMStar is multiple choice, so format learning can look like capability | ChartQA guard, 3% max relative drop, zeroes the round |
| NR-5 | three seeds can cluster and shrink σ until noise gates cannot fail | σ = `max(3-seed std, sqrt(p(1-p)/n))` from the realized item count |
| NR-6 | reported carelessly this looks like an independent replication | `relation_to_sibling_packages`, and `contract_check.py` fails if this file or `instruction.md` drops the cheap-tier framing |

**NR-1 is the one to take seriously.** It is entirely possible that at this budget the difference
between the best expressible selection and no selection is smaller than the noise floor. That gate
is written before any number exists and it fails rather than scaling. The honest response to a
failure is to report the cost finding and either raise the budget in a **new dated contract
section** or drop the tier — not to lower the gate.
