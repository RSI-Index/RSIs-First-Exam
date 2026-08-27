# CUDA fp16 GEMM Kernel Search

Work in:

```text
/app/workdir/gemm_lab
```

Implement hand-written CUDA fp16 GEMM kernels for row-major matrices. Add
candidate versions under:

```text
kernels/
```

The kernel ABI is defined in:

```text
include/gemm_task/api.hpp
```

Each candidate must provide:

```cpp
extern "C" void launch_gemm_f16(
    int m,
    int n,
    int k,
    const half* A,
    const half* B,
    half* C,
    cudaStream_t stream);
```

A and B are row-major device pointers, and C must be overwritten with
`C = A x B`. All benchmark dimensions are positive multiples of 16.

## Optimization Target

This is a CUDA kernel throughput optimization task. The mathematical operation
is plain fp16 matrix multiplication:

```text
C[m, n] = sum_k A[m, k] * B[k, n]
```

Inputs `A` and `B` are fp16 row-major matrices. Output `C` is fp16 row-major.
Correctness is checked with sampled dot products against the same operation
accumulated in float.

A strong solution should push sustained fp16 Tensor Core GEMM throughput as
close as practical to the hardware roofline of the exposed NVIDIA H100 NVL.
The GPU has Hopper compute capability 9.0 and the task builds native `sm_90a`
code with CUDA 12.8. NVIDIA specifies 1,671 TFLOP/s for H100 NVL fp16 Tensor
Cores with structured sparsity, or about 835 TFLOP/s for the dense operation
used here. On this 400 W host, a tuned vendor-library calibration is roughly
550 TFLOP/s on the visible public 2048 shape. That measurement is context for a
practical ceiling, not an allowed implementation path: candidate code must
remain hand-written. Do not treat a 30-100 TFLOP/s kernel as mature. Keep
looking for designs that move materially closer to the GPU's practical
sustained Tensor Core limit.

Improve measured TFLOPS by writing a better kernel, for example with tiling,
coalesced memory access, shared memory staging, warp-level work partitioning,
and Tensor Core paths such as WMMA or inline MMA. Do not be afraid to replace a
working kernel with a substantially different design when the data says the
current family is weak. The task is not to change the harness or call an
external GEMM implementation.

## Long-Run Search Requirement

This task is designed for iterative kernel research, not a one-shot correct
implementation. A correct but slow kernel is only a starting point. Keep making
measured successor candidates until the session budget is genuinely exhausted or
the continuation wrapper hands off to another native Codex session.

Within each native session:

- create several distinct candidate versions, not just one edit;
- run targeted `tools/run_one.sh` probes before expensive public feedback;
- run public feedback more than once when the best version changes;
- set `FINAL_VERSION` to the candidate version you want submitted;
- preserve older working candidates so the wrapper can snapshot the trail;
- after each public measurement, immediately choose and implement or profile the
  next concrete experiment unless the session is genuinely blocked;
- record the best numbers, rejected ideas, and the next concrete experiment in
  `notes.md`.

If progress stalls, switch kernel family instead of only changing block sizes.
At minimum, consider these clean-room implementation families:

- scalar or simple CUDA baseline only for sanity checks;
- WMMA using direct global-memory tile loads;
- WMMA with shared-memory staging, skew/padding, and coalesced cooperative
  loads;
- shape-specialized fast paths for aligned full tiles plus a correctness-safe
  fallback for tails;
- lower-level inline `mma.sync` / `ldmatrix` variants when WMMA overhead appears
  limiting;
- Hopper-native `wgmma.mma_async`, TMA, asynchronous transaction barriers, and
  warp-group schedules compiled for `sm_90a`;
- deeper pipeline or data-movement rewrites such as double buffering,
  warp-specialized load/compute organization, or other clean-room Tensor Core
  schedules;
- occupancy/register/shared-memory tradeoff variants guided by profiler output.

Do not spend the whole run inside one weak family. When two nearby variants tie,
the next useful experiment is usually a different data-movement strategy,
accumulation strategy, or launch geometry.

## Versioning

Create candidate files using three-digit version names:

```text
kernels/gemm_v001_baseline.cu
kernels/gemm_v002_tiled.cu
```

Write the candidate version number you want graded to:

```text
FINAL_VERSION
```

For example:

```text
2
```

or:

```text
002
```

Both forms select `gemm_v002*.cu`.

## Useful Commands

From `/app/workdir/gemm_lab`:

```bash
./tools/build.sh --version 1
./tools/run_one.sh --version 1
bash /workspace/benchmark/run_public_tests.sh
```

Optional profiling:

```bash
./tools/profile.sh --version 1 --m 2048 --n 2048 --k 2048
```

## Public Feedback

The visible development benchmark is:

```bash
bash /workspace/benchmark/run_public_tests.sh
```

By default it reports feedback on this public shape:

```text
2048x2048x2048
```

Use this script as development feedback while you build better candidates. It
compares the newest candidate against the current public baseline, writes
diagnostic artifacts, and helps you decide the next experiment. Public feedback
is diagnostic only; it is ignored by the final verifier.

The public workload is deliberately visible so you can iterate quickly. The
final verifier uses hidden seeds and a separate hidden workload.

## Final Verifier Workload

The final verifier rebuilds and remeasures the candidate named in
`FINAL_VERSION` from scratch on hidden inputs. Optimize for correctness and
throughput beyond the public feedback signal. Hidden verifier details are not
provided in the task instruction.

The final gates are:

- protected harness/public benchmark files were not modified;
- candidate source does not call banned GEMM libraries or frameworks;
- hidden sampled correctness passes;
- hidden performance measurements complete successfully.

## Constraints

- Do not call or include GEMM libraries or frameworks, including cuBLAS,
  cuBLASLt, CUTLASS, CuTe, cuDNN, Triton, PyTorch, FlashInfer, or other
  library GEMM paths. Do not use dynamic loading to call those libraries.
- Do not modify the harness, task metadata, verifier, public benchmark scripts,
  or files outside `/app/workdir/gemm_lab/kernels`, `FINAL_VERSION`, and
  `notes.md`.
- Preserve prior working candidate versions. Create a new version for each
  distinct experiment after a candidate has built and run.
- Keep `FINAL_VERSION` pointing to the candidate you want submitted.

## Feedback

The public feedback script is a development signal only. It uses the visible
2048x2048x2048 shape and writes:

```text
/workspace/benchmark/public_feedback.md
/workspace/benchmark/public_results.json
/workspace/benchmark/public_history.jsonl
/workspace/benchmark/optimization_state.json
/workspace/benchmark/candidate_evals/
```

Continuation runs also expose:

```text
/workspace/benchmark/round_controller.py
/workspace/benchmark/training/latest_summary.md
/logs/agent/continuation_rounds/
```

The round controller records public feedback, code snapshots, `FINAL_VERSION`,
and a public curve for native-session continuation. You may call:

```bash
python3 /workspace/benchmark/round_controller.py status
python3 /workspace/benchmark/round_controller.py eval --native-session ${CODEX_NATIVE_SESSION:-0} --tier public
```

Each public run also writes a per-evaluation artifact directory containing the
exact public result and current code snapshot. In continuation runs these are
mirrored to:

```text
/logs/agent/continuation_rounds/candidate_evals/
```

The final verifier uses hidden seeds and a separate hidden workload. Do not
overfit to a single public timing sample.

## Completion

Before ending, leave:

```text
FINAL_VERSION
notes.md
```

`notes.md` should summarize the candidate trail, best public measurements, and
the `FINAL_VERSION` you want graded.
