# Deployment-supplied task tools

Scheduler control, input snapshotting, counters, and artifact staging live
under immutable `/task-tools`. Candidate code cannot replace them.

## Adaptive attempts

`tmax_async_run.py submit --attempt-id ID --max-updates N` accepts every
integer `N` from 1 through 200. It freezes candidate source, training-data,
training-prompt, training-reward, and RL-recipe hashes before launching one
64-GPU 16+48 attempt. It polls without consuming research decisions and
returns control to the same agent session at terminal state.

`tmax_async_run.py resume` extends a checkpoint only when all five scientific
input hashes match. `stop` records an early-stop reason and terminates only the
named attempt.

Short attempts terminate at their requested update count but retain a
200-update scheduler horizon.

## Inspection and staging

`tmax_task_tool.py summarize` reports requested/completed updates, public
training reward, effective trajectories/tokens, variance groups, entropy,
drift and importance diagnostics, parser/tool validity, generated tokens,
sandbox steps/failures, latency, GPU seconds, checkpoints, and status.

`tmax_task_tool.py stage` accepts only a completed 200-update,
51,200-trajectory, 64-H100 run and creates `/app/output/submission/model`, a
hashed model manifest, and `submission-selection.json`.

`tmax_task_tool.py audit` cross-checks the append-only experiment ledger,
provenance, attempt control records, topology, resource counters, scientific
input hashes, checkpoint ownership, staged hashes, network isolation, and
cleanup evidence.

The final verifier separately evaluates the immutable official `step_200`
baseline and staged candidate with one frozen protocol. Candidate training
code, prompt, reward, and data artifacts are not imported by that evaluator.
