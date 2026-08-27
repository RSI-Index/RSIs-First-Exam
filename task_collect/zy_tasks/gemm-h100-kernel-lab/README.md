# ai-infra/gemm-h100-kernel-lab

Clean-room Harbor task for optimizing a hand-written CUDA fp16 GEMM kernel on a
local NVIDIA H100 NVL GPU.

The agent works in `/app/workdir/gemm_lab`, creates versioned candidate kernels
under `kernels/`, uses the public feedback script for development measurements,
and writes the candidate version to `FINAL_VERSION`.

## Task Objective

The optimization target is a hand-written CUDA fp16 GEMM kernel:

```text
C = A x B
```

All matrices are row-major. The candidate must expose the ABI in
`include/gemm_task/api.hpp`:

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

The oracle installs an original WMMA Tensor Core baseline. It is much better
than the previous naive CUDA sanity check, but it is still intentionally simpler
than a fully tuned pipelined MMA kernel. The intended improvement path is
standard GEMM kernel optimization: tiling, coalesced loads, shared memory,
warp-level decomposition, and lower-level Tensor Core implementation. External
GEMM libraries and frameworks are banned, including dynamic loading paths to
those libraries.

The performance target is not "pass correctness" or "beat the baseline by a
little"; it is to push sustained fp16 Tensor Core GEMM throughput as close as
practical to the H100 NVL dense fp16 hardware roofline. NVIDIA's sparse fp16
figure is 1,671 TFLOP/s, corresponding to about 835 TFLOP/s for this task's
dense GEMM. Local 400 W H100 NVL calibration reached roughly 550 TFLOP/s on the
public 2048 shape and 460 TFLOP/s on the final-size 4096 shape. The oracle's
30-36 TFLOP/s is therefore intentionally a weak seed rather than a mature
solution. Candidate code may not call the calibrated vendor library.

## Public Feedback And Final Verification

| Stage | Workload | Purpose |
| --- | --- | --- |
| Public feedback | Visible public GEMM workload | Development signal for candidate iteration. |
| Final verifier | Hidden GEMM workload | Hidden-seed scoring workload. |

The public script is:

```bash
bash /workspace/benchmark/run_public_tests.sh
```

It runs public candidate feedback and writes `/workspace/benchmark/public_feedback.md`,
`/workspace/benchmark/public_results.json`,
`/workspace/benchmark/optimization_state.json`, and
`/workspace/benchmark/public_history.jsonl`. These files are ignored by the
final verifier; they are only a diagnostic trail for the agent.

Continuation runs use `/workspace/benchmark/round_controller.py` to preserve the
same public signal across native Codex sessions. Each round records public
metrics, code snapshots, `FINAL_VERSION`, `notes.md`, and a public curve under
`/logs/agent/continuation_rounds`; a short in-container summary is mirrored to
`/workspace/benchmark/training/latest_summary.md`.

The final verifier recomputes correctness and performance from scratch using
hidden seeds. Reward is the hidden performance score after all gates pass.

The implementation is clean-room: the task does not reuse the previous
playground, build scripts, feedback state machine, or reference kernel.

## Environment

- Base image: NVIDIA CUDA 12.8.1 runtime on Ubuntu 22.04
- Packages: CUDA 12.8 compiler/Nsight Compute, CMake, Ninja, build tools, Python 3, ripgrep
- Hardware: one NVIDIA H100 (validated here on a 94 GB H100 NVL), native `sm_90a`
- Agent timeout: 17200 seconds
- Verifier timeout: 3600 seconds
- Public timing: 10 warmups, 100 measured launches, 5 paired repeats
- Final timing: 20 warmups, 200 measured launches, 5 repeats

## Verifier

| Gate | Type | What it checks |
| --- | --- | --- |
| Protected files | Programmatic | Clean-room harness and public benchmark files were not modified. |
| Source scan | Programmatic | Candidate code does not reference library GEMM/framework implementations. |
| Correctness | CUDA runner | Hidden sampled dot-product checks pass on hidden inputs. |
| Performance | CUDA runner | Reward is the hidden performance score after gates pass. |

The verifier writes:

```text
/logs/verifier/reward.txt
/logs/verifier/reward.json
/logs/verifier/gemm_results.json
```

## Layout

```text
instruction.md
task.toml
SOURCES.md
environment/
  Dockerfile
  benchmark/
    run_public_tests.sh
    public_feedback.py
    round_controller.py
  workdir/gemm_lab/
    CMakeLists.txt
    README_AGENT.md
    include/gemm_task/
    src/
    kernels/
    tools/
solution/
  solve.sh
  original_solution/
tests/
  test.sh
  verify_gemm.py
  protected_hashes.txt
scripts/
  gpu_docker_env.py
```

## Running

Oracle:

```bash
harbor run -p 0608-task-search-design-v2/harbor_tasks/gemm-h100-kernel-lab -a oracle
```

On the local Docker setup used for this repo, GPU support is exposed through the
task's helper environment class:

```bash
HARBOR_GPU_DEVICE=3 \
PYTHONPATH=0608-task-search-design-v2/harbor_tasks/gemm-h100-kernel-lab/scripts \
  harbor run -p 0608-task-search-design-v2/harbor_tasks/gemm-h100-kernel-lab \
  -a oracle \
  --environment-import-path gpu_docker_env:GpuEnabledDockerEnvironment \
  --n-concurrent 1 \
  --yes
```

Direct public feedback inside the container:

```bash
bash /workspace/benchmark/run_public_tests.sh
```

Direct verifier inside the container:

```bash
bash /tests/test.sh
```

Three-session Codex run on this host's physical GPU 3:

```bash
bash 0608-task-search-design-v2/harbor_run_scripts/gemm-h100-kernel-lab/launch-gpt-5.6-luna-xhigh-3round-gpu3-tmux.sh
```

This launcher uses `openai/gpt-5.6-luna` with `reasoning_effort=xhigh`, validates
the pinned H100 NVL UUID, and maps host GPU 3 to container CUDA device 0.

## Provenance

See `SOURCES.md`. This task is not a port of VibeKernel or `playground-base`.
