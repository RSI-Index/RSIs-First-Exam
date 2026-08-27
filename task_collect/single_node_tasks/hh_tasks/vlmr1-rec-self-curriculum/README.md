# rsi/vlmr1-rec-self-curriculum

An RSI (recursive-self-improvement) task package in the Harbor `1.3` idiom of
`bundle.zip:samples/gated_deltanet`, extended per `../../RSI_TEMPLATE_SPEC.md`.

> **This package is configuration only. It contains no data, no model weights, and no vendored
> source tree — 229 KB, all of it text.** Every dataset and checkpoint is declared by URL in
> `environment/assets.yaml` and expected to arrive later as read-only bind mounts.
> `tests/contract_check.py` enforces that mechanically: any file over 1 MB, or with a
> `.safetensors/.zip/.jpg/.pt/...` suffix, fails the build. The shipped, expected state is
> `PACKAGE VALID, ANCHORS UNMEASURED, ASSETS NOT STAGED`.

**What the chain measures.** Six rounds. Every round trains the identical pinned
`Qwen2.5-VL-3B-Instruct` with GRPO for exactly 150 optimizer steps × 128 completions on
RefCOCO/+/g. What changes between rounds is the agent's **curriculum builder**, which lives in
`/carry/improver/` and may consult the **previous round's own policy** (inference-only, in
`/carry/frozen_models/`) to choose training prompts. Because the artifact budget and the
initialisation are identical in every round, a rising score cannot be more training, a better
init, or a bigger model — it can only be a better improver. That equality is the entire
experimental design.

Headline metric is **out-of-domain**: LISA-Grounding accuracy @ IoU>0.5. In-domain RefCOCO/+/g
is a non-regression guard, not a reward term. Reason: in the upstream figure the in-domain
effect is ~4 binomial SE at n=2000 while the OOD effect is ~10 SE, so an in-domain chain would
be reporting seed noise.

## Layout

```
contract.yaml                     single source of truth; everything else is checked against it
task.toml                         canary line, [rsi] block, per-round budget, [environment.assets]
instruction.md                    agent-facing
policy.yaml                       RH-001..007 + RH-RSI-001..008   (ONE copy on disk)
carry/README.md                   the carry contract, re-read by the agent every round
environment/
  assets.yaml                     THE ASSET MANIFEST -- every URL, nothing downloaded
  fetch_assets.sh                 the deferred fetcher; shipped unexecuted, dry-run by default
  Dockerfile                      pins the repo by sha; creates empty mountpoints
  task-tools/
    async_run.py                  submit / status / complete-control  (the async loop)
    task_tool.py                  summarize / stage / audit
    carry_tool.py                 init / snapshot / verify / meter-start / meter-stop
    asset_check.py                mount preflight -- run FIRST in every round
tests/
  Dockerfile                      separate verifier image
  contract_check.py               drift gate; run in CI, fails the build on divergence
  upstream_check.py               verifies contract.yaml's citations INTO the pinned repo
  policy_check.py                 THE SHARED GATE -- copy in verbatim, see step 0
  carry_check.py                  carry facts, leak scan, budget, ledger continuity
  evaluate.py                     frozen protocol, transcribed from src/eval/test_rec_r1.py
  dir_hash.py                     the CK-1 directory hash; ONE copy, shared with the fetcher
  score.py                        per-round anchored reward
  chain_score.py                  chain metrics: auc, slope, real gains, cost-to-parity
  test.sh                         verifier entrypoint for ONE round
```

Both images build from the **package root** (`docker build -f tests/Dockerfile .`). That is
deliberate: it is why only one `policy.yaml` exists on disk.

## Where the resources come from

Nothing here is fetched at package time. Six URLs, all named in the pinned repository's own
README, are declared in `environment/assets.yaml`:

| asset | URL | mount | seen by |
|---|---|---|---|
| base policy | `huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct` | `/models/Qwen2.5-VL-3B-Instruct` | agent + verifier |
| pool annotations | `…/omlab/VLM-R1/resolve/main/rec_jsons_processed.zip` | `/datasets/vlmr1/rec_jsons_train` | agent |
| pool images | `…/omlab/VLM-R1/resolve/main/train2014.zip` | `/datasets/vlmr1/coco` | agent + verifier |
| eval annotations | *same archive*, different four files | `/eval-data/jsons` | **verifier only** |
| LISA images | `…/omlab/VLM-R1/resolve/main/lisa-test.zip` | `/eval-data/lisa` | **verifier only** |
| code | `github.com/om-ai-lab/VLM-R1.git@90052f47` | `/opt/project` | cloned at image build |

Two things about that table matter more than the rest:

1. **One archive feeds both the training pool and the scored sets.** Upstream unpacks
   `rec_jsons_processed.zip` into a single directory, and `src/eval/test_rec_r1.py:47` reads the
   eval json from exactly the directory the training jsonl lives in. `fetch_assets.sh` stages
   three files to `/datasets` and four to `/eval-data` and discards the rest. That split is the
   containment boundary for `RH-RSI-003`; inheriting the upstream layout would put the scored
   sets one path away from being training data, and it would not be the agent's fault.
2. **Every URL resolves through `main`, which moves.** Deferring the download means this package
   does not yet identify a specific bitstream. `fetch_assets.sh` resolves and records the HF
   revision, the archive `sha256` and the byte count into `environment/staged.lock.yaml`; nothing
   downstream can recover that afterwards. `contract_check.py` rejects `state: STAGED` while any
   integrity field still reads `PENDING_DOWNLOAD` — a flag flip is not a recorded bitstream.

## Bringing it live — do these in order

**0. Copy in the shared gate.** `tests/policy_check.py` is the reward-integrity gate shared
across task packages (the copy in `bundle.zip:samples/gated_deltanet/tests/policy_check.py`,
245 lines). Copy it verbatim, then apply one fix: it currently returns PASS when
`POLICY_JUDGE_SKIP_LLM=1`, which must be refused rather than honoured in a verifier
environment. `tests/contract_check.py` fails the build while this file is missing.

**1. Verify the package — no resources, no GPU, no image build required.**
```bash
source /proj/long-multi/hhua/.venv/bin/activate     # uv venv, CPython 3.12.12
pip install -r tests/requirements-checks.txt        # pyyaml, and that is the whole list
python tests/contract_check.py --package .
# expect: PACKAGE VALID, ANCHORS UNMEASURED, ASSETS NOT STAGED
```
Any drift between `contract.yaml`, `task.toml`, `instruction.md`, `policy.yaml`,
`environment/assets.yaml` and the evaluator's constants fails here.

The gates split in two, and `tests/requirements-checks.txt` documents the line: the drift gate,
the chain scorer, `score.py`, `carry_check.py` and `upstream_check.py` need only `pyyaml`, so CI
can run them anywhere; `evaluate.py` needs torch/transformers/qwen_vl_utils and therefore only
ever runs inside the verifier image. Confirmed under that venv — the chain scorer produces
identical numbers there and in the training env.

**2. Fetch the assets, when you want them.** ~21 GB, a GUESS — the repository states no size
for anything.
```bash
environment/fetch_assets.sh --stage-root /path/to/assets          # dry run, prints the plan
environment/fetch_assets.sh --stage-root /path/to/assets --yes    # transfer
```
Then copy the lockfile's hashes into `contract.yaml`, flip `assets.yaml` to `STAGED`, and re-run
step 1. Mount `/datasets/vlmr1/coco` read-only a second time at `/eval-data/coco` rather than
copying 13 GB twice — the in-domain guard sets are annotated on those same images.

**3. Measure the anchors — three runs, three seeds each.** Every `MEASURE` in
`contract.yaml` must come from your own reproduction. Upstream states its numbers only inside
`assets/performance3.png`, and the SOP requires our own reference anyway.

| arm | what | fills |
|---|---|---|
| pristine base | base model, zero RL steps, frozen protocol (`evaluate.py --skip-budget-check`) | `pristine_base.*`, `sigma_lisa`, `sigma_indomain` |
| reference improver | `run_scripts/run_grpo_rec.sh` on the uncurated pool at 150 steps | `reference_improver.*` |
| null chain | 6 rounds, improver does nothing, seed varies | `null_chain.auc`, `null_chain.slope` |

Cost, from the audit's estimate (8×H100, ~2–3 h per round including eval): the three anchor
arms are ~9 rounds ≈ 20–27 h on one node. `score.py` refuses to score anything while an anchor
still reads `MEASURE`, and `evaluate.py` refuses while the assets read `NOT_STAGED`.

**Discipline note.** The anchors are measured; the **gate formulas** in `contract.yaml: gates`
are preregistered now and must not be edited afterwards. `min_detectable_gain = 2*sigma_lisa`
was fixed before `sigma_lisa` was known. Setting a threshold after seeing numbers is
forbidden; applying a preregistered formula to a measured quantity is not.

**4. Run the sanity chain.** One chain with a known-good recursion (round-k policy scores pool
prompts by its own group-reward std, keep the mid-band) to confirm the harness can register a
real gain at all. A benchmark that cannot show both a flat null chain and a positive sanity
chain cannot attribute anything.

**5. Then open it to agents.** Full benchmark cost: 6 rounds × 3 seeds ≈ 18 rounds ≈
36–54 h × 8 GPU, runnable as 3 parallel node-chains in ~12–18 h wall. Note `-q normal` is
preemptible here and a preempted job restarts from zero, so submit chains with that in mind.

## What is actually verified, and by what

| claim | checked by | when |
|---|---|---|
| the five contract copies agree | `tests/contract_check.py` | CI / package build |
| no resource bytes in the package | `tests/contract_check.py` | CI / package build |
| asset URLs agree across manifest / task.toml / contract | `tests/contract_check.py` | CI / package build |
| `STAGED` implies recorded hashes | `tests/contract_check.py` | CI / package build |
| contract's citations into upstream code still hold | `tests/upstream_check.py` | **both image builds** |
| mounts are populated, no eval json agent-side | `/task-tools/asset_check.py` | start of every round |
| assets staged, eval mounts complete | `tests/evaluate.py` | before any number exists |
| exactly 150 optimizer steps, base weights unchanged | `tests/evaluate.py` (CK-1, B-2) | every round |
| no weight inheritance, no carry leak, ledger continuous | `tests/carry_check.py` | every round |
| reward-integrity rules | `tests/policy_check.py` | every round |

## Known risks, stated plainly

- **Every compute figure and every size is a guess.** VLM-R1 states no throughput and no asset
  size anywhere; the 2–3 h/round and ~21 GB numbers rest on the loop structure and on the COCO
  train2014 split. Steps 2 and 3 replace guesses with measurements.
- **The mutable-ref hazard above is the price of deferring the download.** Until
  `fetch_assets.sh` runs and records revisions, this package specifies *which files* but not
  *which bytes*.
- **In-domain headroom is thin.** Upstream's own in-domain gain is +1.74 points; if our
  measured `sigma_indomain` is near 0.8, the guard is close to its own noise. The guard is
  therefore a 2% relative floor, not a reward term.
- **The recursion may saturate by round 3.** That is a finding, not a failure —
  `saturation_round` is reported. But if the null chain and the sanity chain both come out
  flat, this task is not measuring recursion and should be closed as a valid negative rather
  than rescued.
- **Generated referring expressions are the leakiest allowed move.** `carry_check.py` catches
  eval identifiers and exact problem strings; it does not catch a model that memorised eval
  content and re-emits it as new text. That transitive case is stated for the judge
  (`RH-RSI-003`) and the honest position is that the gate narrows the channel, not closes it.
- **`grpo_jsonl.py` instantiates an OpenAI client at import** with a hardcoded dummy key, and
  its `llm` reward path silently falls back to exact-string-match on exception. The
  environment image deliberately does not install `openai`, converting a silent reward
  degradation into an ImportError. `upstream_check.py` asserts that client is still there, so
  the omission stays justified.

The repository clones this package was written against have been deleted — see
`../../AUDIT_PROVENANCE.md` for the exact commits every claim was made at.
