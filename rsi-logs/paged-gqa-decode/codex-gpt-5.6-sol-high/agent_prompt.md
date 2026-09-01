# Single-File Triton BF16 Paged-GQA Decode Attention

Improve a from-scratch Triton paged-GQA/MQA decode-attention kernel on one H100 NVL while preserving the fixed ABI and numerical semantics.

## Goal

- Test hypotheses about page traversal, grouped-head reuse, `S_q`-specific scheduling, context splitting, and stable online-softmax organization.
- Maximize correctness-gated geometric-mean effective throughput across fixed `S_q=1` and `S_q=16` families.
- Use family and case aggregates to learn which mechanisms help, fail, or trade off across context regimes.

## Workspace

| Item | Value |
| --- | --- |
| Working directory | `/workspace` |
| Starting state | Fixed `/workspace/TASK_BRIEF.md`, an editable signature-correct `/workspace/submission.py` stub, and non-scoring `/workspace/smoke_test.py`; pinned FlashAttention/FA3 source is visible outside the candidate tree |
| Deliverable | Exactly one complete closed regular file: `/workspace/submission.py` |
| Public smoke command | `python3 /workspace/smoke_test.py` |

## Reference baseline

| Item | Value |
| --- | --- |
| Baseline | Official source-built FA3 `flash_attn_with_kvcache` |
| Reported result | Not reported for the fixed six-case H100 NVL protocol |
| Status | Repository implementation and methods are documented; fixed-protocol performance was not reproduced during task authoring |
| Comparison | Every ordinary submission runs the immutable candidate and FA3 sequentially on the same Judge GPU under the same fixed protocol; this contributor-confirmed paired pass returns absolute candidate throughput and `baseline_ratio` and is included in the 20-minute cap. A separate baseline submission receives FA3's real score. |

## Required ABI

`/workspace/submission.py` must export:

```python
paged_gqa_decode_bf16(
    q, k_cache, v_cache, page_table, cache_seqlens, softmax_scale, out
) -> None
```

`q` and `out` are CUDA BF16 `[B,S_q,H_q,128]`; paged K/V are CUDA BF16 `[N_pages,P,H_kv,128]`; `page_table` and `cache_seqlens` are contiguous CUDA int32; `H_q % H_kv == 0`; and `softmax_scale` is the inverse square root of 128. The function must return `None`, fully write the supplied `out` on the current CUDA stream, preserve every input bit, and implement bottom-right causal attention.

## Research loop

1. Inspect `/workspace/submission.py`, `/workspace/TASK_BRIEF.md`, `/workspace/smoke_test.py`, and prior safe feedback.
2. Form one falsifiable hypothesis about mapping, tiling, reuse, splitting, or stable softmax.
3. Change only `/workspace/submission.py`; run `python3 /workspace/smoke_test.py` when useful and finish all candidate-producing/JIT work in Work.
4. Flush and close the complete file, submit it, and inspect aggregate evidence with `rsi-submit --list`.
5. Keep, revise, or reject the hypothesis before the next candidate.

## What you may change

- Only `/workspace/submission.py`.
- Triton JIT kernels, reviewed Python launch helpers in that file, dispatch, and compile-time/runtime scheduling parameters.
- Shape-specialized scheduling for the declared shape classes. The scored entrypoint must reach a declared Triton JIT kernel; all tensor data-path computation belongs inside reachable Triton kernels.

## What stays fixed

- Head dimension 128, BF16 inputs/outputs, current-stream execution, bottom-right causality, physical page indirection, per-row ragged lengths, and grouped query-to-KV head mapping.
- Cases `(B,S_q,L_max,H_q,H_kv,P)`: `(64,1,1024,32,8,16)`, `(16,1,8192,32,4,64)`, `(16,1,32768,32,1,128)`, `(64,16,1024,32,8,16)`, `(16,16,8192,32,4,64)`, and `(16,16,32768,32,1,128)`.
- No edits, symlinks, or shadow files outside `/workspace/submission.py`; no additional candidate artifacts or data.
- No FlashAttention, FA3, FA4, CUTLASS, CuTe, SDPA, host-side Torch tensor computation or mutation, matrix multiplication, einsum, vendor attention/GEMM, native/external extensions, dynamic libraries, file reads, network, subprocesses, dependency/environment mutation, monkeypatching, private streams, CUDA graphs, input mutation, replay, hard-coded tensor answers, evaluator discovery/tampering, timing manipulation, or fabricated metrics.

## Evaluation and feedback

| Item | Value |
| --- | --- |
| Fixed workload | Six declared cases with hidden fixed seeds, fresh BF16 tensors and pointers, one-to-one physical-page permutations, and distinct lengths from the greater of `S_q` and half of `L_max` rounded down through `L_max`; one JIT call and three warmups precede seven fresh timed trials per case; the complete paired evaluation is capped at 20 minutes |
| Correctness | Every fresh output is NaN-poisoned, must be fully finite/written, must preserve input hashes, and must satisfy the declared FP32-vs-BF16-reordering relative max/mean error bounds |
| Reward | Effective FLOPs use valid bottom-right-causal pairs; each family is the geometric mean of three case TFLOP/s values; `reward=score_tflops=sqrt(T_q1*T_q16)` in effective TFLOP/s, higher is better |
| Scoreable candidate | The complete Work-produced `/workspace/submission.py` passes scope, source, ABI, integrity, deadline, and correctness gates and is directly imported; Judge performs JIT loading and fixed evaluation, not candidate optimization |
| Visible feedback | On success: `score_tflops`, `baseline_ratio`, both family geometric means, and each declared case's correctness pass, median latency, effective TFLOP/s, and bounded max/mean error ratios; on failure: a stable category, public case ID when applicable, and bounded candidate-owned diagnostic; all Judge output and the Harness footer are visible |
| Hidden | Judge source/internals, fixed seeds, exact ragged lengths, page maps, values, pointers, poison data, oracle/FA3 outputs, per-row/head outcomes, per-trial timings, failing elements, and undeclared evaluator state |

There is no finite candidate-failure scalar. Candidate-invalid, failed-case, timeout, incomplete, dependency, evaluator, and infrastructure outcomes are unscored and produce no reward.

## Submission checklist

- `/workspace/submission.py` is the only changed path and contains the complete candidate.
- The required function returns `None`; run `python3 /workspace/smoke_test.py` when that public proxy is useful.
- All candidate-producing work, JIT probes, and local processes have finished; the file is flushed and Judge needs no Work process, GPU state, socket, cache, or auxiliary file.
- The candidate does not depend on network, added data, system mutations, or another file.
- The next submission tests a stated hypothesis.

## Evaluation

The Judge evaluates the entire current WORKDIR as it exists when you submit.
The best valid primary score wins. You may call `rsi-submit` repeatedly to
receive feedback while improving the same workspace.
`rsi-submit --list` shows previous submissions; `rsi-submit --help` shows local usage.
Each submission stores complete Judge stdout and stderr in
`/run/rsi-harness/feedback/agent-N.log`, where N is the submission number.
Judge submissions are unlimited during this run.

Every Work GPU process must exit before rsi-submit. A rejected preflight does not consume a submission.
