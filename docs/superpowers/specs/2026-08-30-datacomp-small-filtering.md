# DataComp-small Filtering RSI Task Specification

## Status

Approved by the user on 2026-08-30. This is a new task proposal, not a repair
of an existing Harbor task.

## Deliverable

Create `examples/harbor-task-agent/datacomp-small-filtering` as a Harbor 1.4
task that runs through the ordinary RSI-Harness command with
`--cluster bluevela`. The task must complete one coherent Work-to-`rsi-submit`
to-independent-Judge run and produce a finite absolute reward.

## Immutable scientific evidence

- DataComp source: `mlfoundations/datacomp` commit
  `4a8df1992566ef8334773f7152e1855b1f716162`.
- OpenCLIP runtime: `mlfoundations/open_clip` commit
  `395851ee9c55fdddb63f5579955cdd6d754d261c`.
- CommonPool-small metadata: revision
  `b7dd54c19199236fe33f367ce7f88f412244fab8`.
- Fixed protocol: ViT-B/32, 12.8M nominal samples, global batch 4096,
  learning rate `5e-4`, warmup 500, seed 0.
- Evaluation: exactly 40 task rows and the official 38-score aggregate named
  `Average`.
- Approved baseline: official L/14 CLIP-score top-30% selector intersected
  with the build-realized successful crawl.
- The baseline has source-reported variance but no locally reproduced
  protocol-matched score. Do not invent a confidence interval.

## Harness resource contract

- Work GPUs: 16. On the current eight-GPU-per-node profile this resolves to
  two Work nodes.
- Judge GPUs: 16. They are a disjoint two-node pool owned by Harness.
- CPU slots: 32 per physical node.
- Memory: 262,144 MiB per physical node.
- Shared storage planning value: 716,800 MiB.
- Work timeout: 21,600 seconds.
- Judge timeout: 21,600 seconds.
- Build timeout: 259,200 seconds because the pinned ML image is large.
- Work and Judge runtime network mode: no-network.
- Blue Vela default submission policy remains `normal` plus exclusive `-x`.
  Priority plus nonexclusive is permitted only as a full-scale fallback after
  concrete scheduler evidence; a successful fallback is the final accepted
  run and is not repeated.

## Architecture boundary

The task contains no `cluster/` directory and no LSF, `blaunch`, host
discovery, rendezvous-IP discovery, Apptainer, scheduler queue, group, or site
filesystem paths. It invokes only fixed framework-native `torchrun` commands.
RSI-Harness reads `task.toml`, allocates Work and Judge, provides topology, and
owns the snapshot/submission lifecycle.

The Environment contains pinned sources and portable asset preparation and
validation tools. Large data remains outside the image under the logical
container root `/rsi-data/datacomp-small`; the Blue Vela profile maps a
site-owned source there. Preparation is idempotent, validates complete outputs,
and atomically publishes a `READY.json`. Normal Work and Judge mounts are
read-only.

The mutable candidate is only `/workspace/research/**` plus the declared
candidate UID/provenance files. Work may use its assigned 16-GPU pool for
research. Before submission it must close all GPU processes and call ordinary
`rsi-submit`.

Judge treats candidate-authored checkpoints and scores as diagnostics. It
validates and snapshots the candidate UID set, starts trusted fixed-protocol
training from immutable sources, evaluates all 40 rows, verifies all 38 reward
operands are finite, recomputes `Average`, and writes exactly one finite
`reward` to `/logs/verifier/reward.json`. A candidate-contract failure writes
reward 0. Infrastructure, assets, scheduler, container, timeout, crash, and
incomplete-evaluation failures write no reward.

## Data readiness

The proposed Blue Vela data root was inspected on 2026-08-30. Its directory
structure existed but contained no files, and the proposal's legacy base SIF
and sidecars were absent. Therefore asset preparation and sealing must finish
before GPU allocation. The current filesystem had more than 800 GiB free.

