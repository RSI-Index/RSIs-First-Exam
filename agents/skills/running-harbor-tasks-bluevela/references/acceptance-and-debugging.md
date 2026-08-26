# Acceptance and debugging

## Evidence-first failure loop

Classify the failing boundary before editing anything:

| Boundary | Primary evidence |
| --- | --- |
| task/compiler | CLI exception and resolved dry-run |
| LSF scheduling | exact `bjobs -l JOB_ID` output |
| image build/cache | build stdout/stderr, SIF sidecar, `RUN_INFO.json` |
| Apptainer/GPU binding | run stdout/stderr, selectors, `nvidia-smi` |
| Work/Agent | `agent_output.txt`, `run_agent.log` |
| submission/Judge | `feedback/agent-N.log`, `report.json` |
| finalization | `final_result.json`, `RUN_INFO.json`, controller exit |

For a failure:

1. Read the complete exact-job stdout/stderr and the relevant standard log.
2. State one root-cause hypothesis tied to that evidence.
3. Change one source-controlled behavior under `/u` and add a focused test.
4. Run only lightweight tests on the login node; use compute for CUDA imports,
   compilation, image work, high-memory preprocessing, training, and evaluation.
5. Submit the same production-shaped command with a fresh run ID.

After three failed hypotheses, reassess the architecture instead of stacking a
fourth patch.

## Hardware adaptation rules

- Treat actual allocation evidence as authoritative. Do not assume GPU 0,
  eight GPUs, H100, a fixed compute capability, or identical selector ordering.
- Under `--containall`, explicitly propagate GPU selectors. A container seeing
  no GPU usually indicates a missing NVIDIA bind or lost selector, not a reason
  to weaken the task.
- With LSF `exclusive_process`, extra parent/helper CUDA contexts can conflict
  with worker ownership. Inspect process and framework logs before changing
  cleanup or collective settings. Using NCCL instead of a framework's custom
  all-reduce is an evidence-driven compatibility option, not a universal rule.
- Ray/vLLM must derive a valid Judge-container IPv4 at runtime when their
  distributed path requires it. Never hardcode an address or use `0.0.0.0` as a
  fallback.
- A build-node CUDA import is not a valid GPU check. Perform real CUDA imports,
  extension compilation, and model loading in the final GPU allocation.
- Do not turn a portability fix into a scientific change. GPU counts, dataset,
  epochs, model, baseline, evaluator formula, and feedback boundary remain the
  task contract unless the user explicitly changes them.

## Success contract

The final production command succeeds only when all applicable evidence agrees:

- login-side command exits zero and prints `Status: completed`;
- final LSF run job is `DONE` with exit code zero; a cache-miss build job is
  also `DONE` with exit code zero;
- `RUN_INFO.json` is `completed`, records exact owned job IDs and resolved
  resources, and names the verified SIF SHA256;
- standard log leaf is
  `<profile logs_root>/runs/<run-id>/<task-id>/`;
- `final_result.json` is `completed` and records at least one completed
  submission;
- `agent_output.txt`, `run_agent.log`, `feedback/agent-N.log`, and
  `submissions/agent-N/report.json` exist and are nonempty;
- Work/Judge selectors are unique and match the compiled counts; and
- expected artifacts are produced directly by `RunArtifactWriter`, not
  converted from Harbor logs.

`PEND`, `RUN`, a successful SIF build, `nvidia-smi`, Agent startup, or one Judge
line alone does not prove end-to-end success.

The production allocation is also the compatibility proof. While it is healthy
and active, leave it running. When it writes a valid terminal
`final_result.json`, that run is immutable and complete; `max_submissions`
omitted only removes the Harness submission cap and does not resume a completed
Agent session.

## Retry and blocking boundaries

Ordinary runtime defects do not require a user checkpoint: fix, test, create a
new run, and continue. Ask for input only when proceeding needs new credentials,
permissions, quota, unavailable external state, an authorized GPU-model/profile
change, or a decision that changes the task's scientific contract.

Never delete another run, shared cache, checkpoint, SIF, or log. Preserve every
failed attempt's manifest and exact job output as audit evidence.
