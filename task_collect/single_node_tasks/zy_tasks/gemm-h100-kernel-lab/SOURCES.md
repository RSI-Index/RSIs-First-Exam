# Sources and Provenance

This task is a clean-room Harbor task for CUDA fp16 GEMM optimization.

No source files, documentation, task harness files, or solution kernels were
copied from VibeKernel, `playground-base`, or any other GEMM benchmark
repository.

The task uses standard CUDA APIs, CUDA headers, CMake, Ninja, and the CUDA base
image listed in `environment/Dockerfile`. Agents may consult public CUDA
documentation when network access is available, but the task package itself
does not vendor third-party benchmark code or third-party solution kernels.

The previous internal task `vibekernel-gemm-a6000` is intentionally not used as
source material for this task. This replacement keeps only the generic benchmark
idea: implement and optimize hand-written fp16 GEMM on a CUDA GPU. This H100
variant was derived from the clean-room task structure and recalibrated for
Hopper rather than porting any external kernel.

The oracle solution is an original WMMA Tensor Core baseline written for this
task's C ABI. It does not port or rewrite the old VibeKernel `v17` kernel.

The continuation wrapper and public-feedback round controller are original
task infrastructure for this clean-room GEMM lab. They reuse the generic Harbor
idea of recording public feedback across native sessions, but they do not copy
the old playground files, VibeKernel state machine, or VibeKernel kernels.
