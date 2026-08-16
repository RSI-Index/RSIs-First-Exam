# Current Task Proposal Re-judge

Date: 2026-08-15

All 25 task instructions were re-judged from scratch after the proposal rubric
was revised. Every judge was explicitly configured as:

- Model: `gpt-5.6-sol`
- Reasoning effort: `xhigh`

Each judge read the full revised rubric, the full task instruction, and relevant
repository, harness, configuration, contract, and evaluator files in the task
directory. These are proposal-stage recommendations; baseline reproduction and
the post-baseline statistical/compute gates have not been run.

## Calibration used

- Any direct model-development work is in scope, including data, tokenizer,
  inference, orchestration, serving, compiler, operator, and kernel research.
- A technically reasonable variance justification is sufficient before baseline
  reproduction. Literal `TODO` placeholders in the central metric/reward remain
  a gate failure.
- Compute above 8 H100-equivalent GPUs or 12 hours is always reported as a
  `Flag`, but cannot by itself cause proposal-stage human review or rejection.
- Official checkpoints and transparent benchmark-owned matched/hardened or
  clean-room baselines are accepted when provenance and comparison are traceable.

## Result summary

| Decision | Count |
|---|---:|
| Strong Accept | 8 |
| Accept | 10 |
| require human review | 1 |
| Reject | 4 |
| Strong Reject | 2 |
| **Total** | **25** |

## Per-task results

| Task | Decision | Compute note | Non-passing gate(s) | Decisive evidence | Judge |
|---|---|---|---|---|---|
| `chenhao_tasks/dinov3_imagenet_semdense/instruction.md` | Strong Accept | **Flag:** 32 H100, ≤14.06h/candidate | None | Pinned config, four matched probes, balanced reward, and mechanism-driven early stops are traceable (`instruction.md:81-203`). | `gpt-5.6-sol/xhigh` |
| `chenhao_tasks/openclip_datacomp_v2_hardened_baseline/instruction.md` | Strong Accept | Within: 8 H100, ≤2.5h/candidate | None | Two upstream commits, `avg_38`, hardened sweep baseline, and several filter/target/architecture hypotheses are explicit (`instruction.md:70-216`). | `gpt-5.6-sol/xhigh` |
| `fq_tasks/magpie_data_pipeline/instruction.md` | Accept | Estimate incomplete: 8 H100; 2-epoch SFT wall time absent | None | Revisions for source, models, data, checkpoint, and judge are pinned; hidden paired-preference evaluation supports repeated data attempts (`task.toml:2-58`, `shared/evaluation_contract.py:70-93`). | `gpt-5.6-sol/xhigh` |
| `fq_tasks/slime_search_r1_algorithm/instruction.md` | Accept | Estimate incomplete: 8 H100; 1,000-round wall time absent | None | Pinned Slime/source artifacts, fixed exact-match reward, and an allowlisted but meaningful RL search space are traceable (`instruction.md:7-78`). | `gpt-5.6-sol/xhigh` |
| `hh_tasks/datacomp-small-self-filtering/instruction.md` | Accept | Estimate incomplete: training GPU type/count and wall time absent | None | Exact DataComp source, frozen recipe, `no_filter` reference, hidden 38-task metric, and six adaptive filtering rounds are defined (`contract.yaml:31-176`). | `gpt-5.6-sol/xhigh` |
| `hh_tasks/nanovlm-finevision-self-selection/instruction.md` | Reject | Within: 8 H100, about 1h artifact run | Traceable Baseline; Scientific Objective and Metric | The instruction makes `pristine` and identity `reference` the same arm, while the canonical contract uses a zero-step pristine arm, so the reward denominator conflicts with the scorer (`instruction.md:97-103`, `contract.yaml:219-256`). | `gpt-5.6-sol/xhigh` |
| `hh_tasks/vlmr1-rec-self-curriculum/instruction.md` | Strong Accept | Within: 8 H100, about 2–3h/round | None | Pinned source, pristine/reference arms, OOD metric, guard, 2σ rule, and six-round carry boundary form a clear recursive design (`instruction.md:50-157`). | `gpt-5.6-sol/xhigh` |
| `rl_tasks/tmax_dppo_mvp/instruction.md` | Accept | **Flag:** 64 H100, about 13.78h/run | None | Official checkpoint baseline, sealed matched evaluator, short-run proxies, trusted diagnostics, and resume support create an adaptive loop (`instruction.md:31-143`). Compute is the only material concern. | `gpt-5.6-sol/xhigh` |
| `yc_tasks/bigvision_lit_coco_alignment/instruction.md` | Strong Reject | Within: 8 H100, planned 2–6h/run | Model-Development AutoResearch Scope | The harness gives no performance feedback between candidates and only evaluates after exit; a prewritten experiment ledger does not create an observe-revise-rerun loop (`run_candidate.py:76-197`). | `gpt-5.6-sol/xhigh` |
| `yc_tasks/datacomp_s_filter_discovery/instruction.md` | Strong Accept | Within: 4 H100, 2–5h/run | None | Selector hypotheses, fixed learner, 38-task metric, sealed evaluation, immutable contract, and experiment ledger are unusually complete (`instruction.md:3-13`, `environment/immutable_contract.md:21-55`). | `gpt-5.6-sol/xhigh` |
| `yt_tasks/datacomp/instruction.md` | require human review | **Flag:** 16 H100, ≤6h/full run | Traceable Baseline | The proposal requires 16 ranks while the verifier hard-requires and launches 8 GPUs, leaving matched baseline/final topology unresolved (`instruction.md:86-159`, `tests/evaluate.py:34-36,181-206`). The compute flag is not the cause of human review. | `gpt-5.6-sol/xhigh` |
| `yt_tasks/deepocr/instruction.md` | Accept | Estimate incomplete: 8 H100; 100-step single-run time absent | None | The matched 100-step loop, mechanisms, checkpoint baseline, public-feedback safeguards, and independently recomputed OCR metrics are defined (`instruction.md:112-171`, `tests/evaluate.py:114-189`). | `gpt-5.6-sol/xhigh` |
| `yt_tasks/deepspec/instruction.md` | Accept | Estimate incomplete: 4 H100; single A/B profile time absent | None | Matched repeated A/B trials, raw timing metrics, policy constraints, target-distribution audit, and trusted goodput are explicit (`instruction.md:72-126`, `tests/evaluate.py:36-318`). | `gpt-5.6-sol/xhigh` |
| `yt_tasks/gated_deltanet_autoresearch_run/instruction.md` | Strong Accept | **Flag:** 32 H100, 1,200s training plus eval/save overhead | None | Live baseline, two controlled probes, two evidence-led synthesis trials, fixed loader/loss, and fixed validation protocol form a strong loop (`instruction.md:17-76`, `tests/evaluate.py:43-263`). | `gpt-5.6-sol/xhigh` |
| `yt_tasks/olmo3-7b-zero-rl-math/instruction.md` | Strong Reject | **Flag:** 72 H100; 2,000-step wall time absent | Source Repository; Scope; Action Space; Evaluation Integrity; Data/Network | It is one fixed pipeline run rather than adaptive research, clones unpinned HEAD, and scores candidate-written JSON without rerunning evaluation (`instruction.md:3-106`, `environment/Dockerfile:45-48`, `tests/test.sh:154-189`). | `gpt-5.6-sol/xhigh` |
| `yt_tasks/rlm/instruction.md` | Strong Accept | Within: 8 H100, 43m21s/evaluation | None | Pinned baseline, measured variance, five repeats, broad orchestration policy space, masked gold, protected code, and recomputed rewards are unusually explicit (`instruction.md:10-153`, `tests/evaluate.py:110-641`). | `gpt-5.6-sol/xhigh` |
| `yt_tasks/superbpe/instruction.md` | Accept | Estimate incomplete: CPU-only/64 cores; full 10GB time absent | None | Released artifact, tokenizer hypotheses, deterministic lossless metric, role-separated hidden evaluation, and in-process baseline comparison are traceable (`instruction.md:14-130`, `tests/evaluate.py:40-159`). | `gpt-5.6-sol/xhigh` |
| `zf_tasks/post-training/search_rl/instruction.md` | Accept | **Flag:** 32 H100/run; single-run time absent, 24h trajectory cap | None | Exact source/checkpoint refs, hidden matched `avg@3`, bounded RL action space, provider isolation, and a persistent experiment loop pass all proposal gates (`instruction.md:11-173`). | `gpt-5.6-sol/xhigh` |
| `zf_tasks/pre-training/learning_rate/instruction.md` | Reject | **Flag:** 216 H100 peak; E5 128 H100, about 23.23h | Scientific Objective and Metric | Candidate-facing Paloma, training-loss, and reward sections remain literal `TODO`s, so the primary success protocol is undefined (`instruction.md:61-71`). | `gpt-5.6-sol/xhigh` |
| `zf_tasks/pre-training/optimizer_update_geometry/instruction.md` | Reject | **Flag:** 216 H100 peak; E5 128 H100, about 23.22h | Scientific Objective and Metric | Candidate-facing evaluation and reward remain literal `TODO`s despite repository metadata elsewhere (`instruction.md:59-68`). | `gpt-5.6-sol/xhigh` |
| `zf_tasks/pre-training/parameterization_transfer/instruction.md` | Reject | **Flag:** E5 128 H100, about 23.22h | Scientific Objective and Metric | The complete candidate-facing “Evaluation and success” section is `TODO`, leaving no fixed success/reward protocol (`instruction.md:75-76`). | `gpt-5.6-sol/xhigh` |
| `zf_tasks/pre-training/positional_encoding/instruction.md` | Accept | **Flag:** 216 H100 peak; E5 128 H100, about 23.22h | None | Source, baseline, metric thresholds, reward, recomputed Paloma/LongPPL, and six-scale mechanism search are concrete (`instruction.md:42-181`, `tests/score.py:34-127`). | `gpt-5.6-sol/xhigh` |
| `zx&qj_tasks/molmo2-video-pointing/instruction.md` | Accept | Estimate incomplete: 8 H100; 8,000-update time TBD | None | Pinned source, pristine recipe/checkpoint, matched metric, frozen evaluator, and explicit data/network/leakage policy are traceable (`task.toml:2-68`, `policy.yaml:35-60`). | `gpt-5.6-sol/xhigh` |
| `zy_tasks/concurrent-agent-serving-optimizer-qwen36-27b/instruction.md` | Strong Accept | Within: 2 H100, about 30–35m/run | None | Pinned runtime/model overlay, fresh hidden Direct/Candidate legs, isolated workloads, calibrated noise, audit gates, and rich feedback make a strong serving research loop (`README.md:8-35,83-100,315-359`). | `gpt-5.6-sol/xhigh` |
| `zy_tasks/gemm-h100-kernel-lab/instruction.md` | Strong Accept | Within: 1 H100; final verifier ≤1h | None | Exact clean-room WMMA baseline/ABI, hidden-seed correctness, median TFLOPS, protected harness, fast iterations, and several kernel families are all traceable (`SOURCES.md:3-26`, `tests/verify_gemm.py:309-385`). | `gpt-5.6-sol/xhigh` |

## What the new distribution shows

- Ten tasks carry an over-reference compute flag. Five of them still receive a
  passing decision because compute is their only or a non-blocking concern:
  DINOv3, TMAX DPPO, Gated DeltaNet, Search-RL, and positional encoding.
- Data, orchestration, serving, tokenizer, and CUDA-kernel tasks are no longer
  rejected merely for being outside a narrow weight-training definition.
- The remaining negative decisions target substantive proposal defects: three
  literal metric/reward `TODO`s, one baseline/reward denominator conflict, one
  missing agent research loop, and one fixed pipeline with source/evaluation
  integrity failures.
- The only `require human review` result is a concrete 16-rank-versus-8-rank
  baseline/evaluator contradiction. Its compute flag is recorded separately and
  did not determine the decision.
