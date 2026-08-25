# `datacomp-small-self-filtering` — RSI benchmark task

> **CONFIGURATION ONLY.** No data, no weights, no vendored upstream source. Everything is declared by
> URL in `environment/assets.yaml` with `state: NOT_STAGED`. `environment/fetch_assets.sh` is the
> fetcher and it ships unexecuted.

## The question

Under a training recipe that **upstream forbids changing**, a frozen 38-dataset evaluator, and a
fixed 12.8M-samples-seen budget that **does not shrink when you select less data**, does an agent
that may rewrite its own pool-filtering improver — and that inherits the CLIP models it previously
trained — produce better filters round over round?

## Why this is the cleanest fixed-compute setting available

Three properties, all upstream's rather than ours:

1. **The recipe is frozen by the benchmark itself.** `README.md:190`: *"You should not modify any
   hyper-parameters for training, including batch size. Any changes may affect accuracy and make
   results incomparable."* `scale_configs.py` fixes batch size, lr, warmup, architecture and
   samples-seen per scale.
2. **Samples seen is independent of subset size.** `train.py:232` reads `train_num_samples` from the
   scale config and `train.py:282` passes `--dataset-resampled`, so a 3M-uid subset trains for the
   same 12.8M samples as a 12.8M-uid one — it just revisits its members. Selecting less data cannot
   buy less compute, and **no enforcement of ours is required for that**; it is how the harness works.
3. **The artifact is a sorted uid array** — `baselines.py`'s own output format (`baselines.py:86`) —
   so a round's submission is integers that can be diffed, hashed and audited.

A benchmark whose fixed-budget half is the repository's own code is worth more than one where we have
to police it.

| what | where |
|---|---|
| seven runnable filters, incl. `no_filter` (our reference arm) | `baselines.py` `BASELINES` |
| uid subset → webdataset | `resharder.py` |
| the frozen recipe | `scale_configs.py` `SCALE_CONFIGS["small"]` |
| the headline mean, with `assert len(df) == 38` | `aggregate_scores.py:42` |
| upstream's own noise statement | `README.md:194` — ~0.2pp on ImageNet, up to 0.006 on the average |
| **no accuracy numbers at all** | `README.md:117` defers every baseline to the paper's Table 3 |

That last row is why **every anchor in `contract.yaml` is `MEASURE`**.

## The two risks that cost this package the top slot, as gates

### 1. The pool must be crawled, so it is site-local and time-local

`download_upstream.py` fetches parquets of **URLs**; `img2dataset` crawls the images. Link rot is
monotone, so re-crawling is **guaranteed** to give a different pool, and our 12.8M is not anyone
else's.

| encoded as | what it does |
|---|---|
| `data.leaderboard_comparable: false` | `contract_check.py` **fails the build if this is ever true** |
| `environment/pool.lock.yaml` | realized uid count, uid-set sha256, download rate, crawl date |
| `tests/pool_check.py` | re-verifies the fingerprint at every round start; **exit 2**, never a zero |
| `launch_gates.pool_is_large_enough` | refuses a crawl below 70% of nominal — set *before* the download |
| all anchors `MEASURE` | no published number can be the denominator of a reward measured on our pool |

A changed pool is an infrastructure failure, not a round the agent earned nothing on. That
distinction is why `test.sh` has an `infra_fail` path at all: a 0.0 fed into `chain_score.py` as
though it were a measurement is how a chain gets reported as saturated.

### 2. A CLIP trained at `small` may be a poor data scorer

The recursive claim is that round k can rank the pool with the model round k−1 produced. That model
was trained on 12.8M samples. It may simply not be competent, in which case the rational agent move
is to use the **precomputed OpenAI similarities that ship in the parquet metadata** (zero GPU cost),
the chain reduces to a filter search, and any slope is unattributable to recursion.

Encoded as `launch_gates.recursion_channel_has_signal`: a filter that ranks the pool with the round-1
CLIP must beat a **size-matched random** subset by ≥ 2σ. Size-matched random rather than `no_filter`,
because subset size alone changes the number of passes under the fixed budget and would otherwise
confound the comparison.

**This gate may fail, and a failure is a result** — it says the recursive claim needs `medium` scale.
The response is a new dated contract section, not a lower gate.

## The 38-vs-40 trap

`tasklist.yml` holds **40** tasks; the headline averages **38**. The two that drop out are
`fairness/fairface` and `fairness/utkface`, and they drop out *implicitly*: neither has a
`main_metric` key, so `evaluate.py:389` falls back to `"acc1"`, the fairness evaluators do not produce
it, `main_metric` is `None`, and `aggregate_scores.py`'s `dropna()` removes them.
`fairness/dollar_street` and `fairness/geode` **are** in the 38 — so two worst-group fairness metrics
sit inside the headline average.

`tests/evaluate.py` therefore **imports** `aggregate_scores.get_aggregate_scores` instead of computing
its own mean, and `contract_check.py` fails if that import disappears: reimplementing the mean would
discard upstream's `assert len(df) == 38` along with the code, and the result would still be called
`average_38`. `upstream_check.py` re-derives the 40 → 38 arithmetic from `tasklist.yml` at build time.

## Validate it

```bash
source /proj/long-multi/hhua/.venv/bin/activate
pip install -r tests/requirements-checks.txt          # pyyaml, nothing else
../../common/sync_common.sh .                         # shared verifier files + tests/common.lock
python tests/contract_check.py --package .
python tests/upstream_check.py --project /path/to/datacomp --contract contract.yaml
```

Expected while nothing is staged:

```
PACKAGE VALID, ASSETS NOT STAGED, POOL NOT FINGERPRINTED, OPEN_CLIP NOT PINNED, ANCHORS UNMEASURED
```

`open_clip` must be pinned because `train.py:13` does `from training.main import main` — it **imports**
open_clip's internal training entry point rather than shelling out, so an unpinned open_clip is an
unpinned recipe. Both Dockerfiles declare `ARG OPEN_CLIP_VERSION` with **no default**.

## Order of operations

```
1. crawl (days), then fingerprint: pool.lock.yaml + pool_uids.npy
2. launch gate: pool_is_large_enough (>= 70% of nominal) -- fetch_assets.sh fails on it
3. pin open_clip; fill contract.yaml data.* from pool.lock.yaml
4. anchors: pristine (untrained) x3 seeds, no_filter x3 seeds
5. reported baselines: clip_score, basic_filter, text_based at seed 0
6. launch gates: anchors_separated, then recursion_channel_has_signal  <-- either may fail
7. null chain x3 seeds, then reference/assert_null_chain.py
8. only then open it to agents
```

## On what counts as leakage

`image_based` — upstream's own baseline — selects pool images by proximity to ImageNet-1k **training**
images, and ImageNet-1k's **validation** split is one of the 38 scored sets. That is permitted here,
and `environment/assets.yaml` says so explicitly rather than leaving a reader to infer it: forbidding
it would make `reference_baselines` unrunnable and turn this into a different benchmark that merely
resembles DataComp. The line drawn instead is that the evaluation **sets** are never available to the
agent, by any path, in any round. Selecting data that resembles a training distribution is curation;
selecting data by looking at the test set is not.
