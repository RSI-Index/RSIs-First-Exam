# Environment design

Environment defines the clean Base image and the persistent Work state. Judge shares this image family; private tests are injected later and never belong here.

## Required output contract

Freeze these values before Verifier design:

```text
WORKDIR: existing absolute POSIX path; /workspace is the default convention
starting source: official URL + exact 40-character SHA
starting assets: exact paths, revisions, hashes, and visibility
candidate-owned paths: absolute paths
prohibited paths/state: absolute paths or named components
Verifier entry interface: command/module/artifact to evaluate
Work resources: GPUs, CPUs, memory, storage estimate, shm_size
build budget: build_timeout_sec
common/Judge env: [environment.env] plus [verifier.env] overrides
Agent network: public, no-network, or exact allowlist
runtime provider route: explicit API base/proxy requirement when non-public
```

Use a non-root `/workspace` WORKDIR by default, but follow repository or image evidence when the real code directory is `/testbed`, `/repo`, `/app`, or another absolute path. RSI-Harness gives no special meaning to `/workspace`. A non-root directory must already exist in the clean image; RSI-Harness places it in an Engine-owned volume, keeps it across rounds, and mounts its Judge snapshot read-only. The WORKDIR must not overlap `/tests`, `/logs/verifier`, or `/run/rsi-harness/staging`.

Use `/` only when full-filesystem visibility is necessary. It selects full-rootfs snapshots, so large changes anywhere in the container increase submission snapshot time and Docker storage. Record that operational cost in README and the contributor review.

## Source state

A common build pattern is:

1. clone the official repository during image build;
2. checkout and verify the exact 40-character SHA;
3. initialize submodules/LFS at pinned revisions when required;
4. remove remote history that could reveal later answers;
5. optionally initialize a fresh local Git repository at the approved starting state so scope checks and Agent diffs remain useful.

Do not rely on a default branch, shallow `HEAD`, mutable release URL, or unverified archive. Do not copy the contributor's current working tree unless that local state is an explicitly confirmed starting artifact.

Everything under `environment/` enters the Work build context and may be visible to the Agent in image layers. Never place private tests, hidden cases/seeds, expected outputs, answer patches, optimal checkpoints, private baseline comparators, `solution/`, or credentials there.

## Dependencies and assets

- Pin base images by digest when available.
- Pin external Python packages exactly. Requirements files, local paths, and VCS URLs with immutable revisions are valid exceptions to inline `==` syntax.
- Pin external source repositories and submodules by immutable revision.
- Pin public datasets/checkpoints by immutable revision and checksum. Record license and provenance.
- Do not pin apt package versions. Run apt update in the install layer, use `--no-install-recommends`, and remove `/var/lib/apt/lists/*` in that layer.
- Bake all Verifier tooling into the shared Environment. `tests/test.sh` must not install it later.
- Do not use bare `nproc`; set task-bounded parallelism explicitly.

If a public asset is too large for `tests/`, it may be baked into Environment only when it is intentionally visible to Work and contains no hidden evaluation information. Hidden `tests/` injection is limited to 100,000 entries and 1 GiB of regular-file bytes.

A Dockerfile is optional when `[environment].docker_image` or Compose `image` names a reviewed prebuilt image. Pin that image by digest and confirm its effective WORKDIR, source/assets, dependencies, user, license, provenance, and empty Docker `Config.Volumes`. RSI-Harness rejects an image that declares any volume because its contents fall outside snapshot ownership. Checking image metadata is an image-preflight execution check, not something the static validator can prove. Do not add an empty Dockerfile merely to satisfy a file-tree example.

## RSI-Harness shape

Use Harbor schema 1.4, Linux, one continuous task, one `main` service, and one shared Environment. Do not author:

- sidecars or multiple services;
- task-authored volumes, host binds, or Docker `VOLUME`;
- privileged mode, capabilities, arbitrary devices, host networking, IPC/PID sharing, or extra hosts;
- multi-step tasks or reward strategies;
- MCP servers, Harbor skill directories, healthchecks, TPU, artifacts, or collect hooks;
- a separate Verifier environment/image;
- `allow_internet` or `gpu_types`.

Use `environment/docker-compose.yaml` only when a supported field such as `shm_size` must be explicit. Do not use the `.yml` spelling; RSI-Harness does not discover it. Its root contains only `services`, with exactly one service named `main`; build context is exactly `.`. Never use Compose volumes. Keep `[environment].workdir` and Compose `working_dir` identical if both exist.

Declare Work GPU count in `[environment].gpus`. Omit GPU model restrictions. Declare Judge GPU count separately later; Work and Judge may receive disjoint GPUs or time-share the caller's pool through release-all mode.

`cpus` and `memory_mb` are enforced Docker limits. `storage_mb` is recorded for planning but is not currently an enforced disk quota. State that limitation in the task README and budget host storage for image layers, Work data, checkpoints, and Judge snapshots.

Default Docker shared memory is only 1 GiB. Add supported Compose `shm_size` when NCCL, multiprocessing DataLoaders, or the repository workload needs more.

CPU, memory, storage, shared memory, build timeout, Compose user, and common service environment describe the shared `main` service used by both Work and Judge. Work and Judge GPU counts, phase users, network policies, and timeouts are separate. Put shared variables in `[environment.env]`; put only Judge-specific overrides in `[verifier.env]`. `[agent].env` and `[verifier.environment]` are not supported interfaces.

## Snapshot and reload boundary

Docker commit and managed WORKDIR snapshots capture filesystem state, not running processes, open buffers, GPU memory, Unix sockets, or in-memory model servers. The Agent must fully write and close candidate-owned code, prompts, configs, indexes, and checkpoints before submission. Judge starts clean from the captured filesystem and must reload the candidate and, when needed, launch local serving inside its single `main` container.

For a non-root WORKDIR, Judge sees that directory read-only. Confirm before Verifier design that candidate loading and evaluation do not attempt to compile extensions, populate caches, update checkpoints, create databases, or write scratch there. Redirect unavoidable disposable library/runtime scratch outside the WORKDIR; never use it as a feedback, result, cross-round state, or hidden-data channel. If evaluation genuinely needs to mutate the candidate tree, choose `/` deliberately or redesign the interface and confirm the trade-off.

## Network boundary

Never use deprecated `allow_internet`.

- `public`: only when the confirmed research requires general web/external access and its data/leakage boundary is documented.
- `allowlist`: use exact hostnames needed for research and for the configured Agent provider/proxy. Avoid wildcards; current runtime behavior is safest with exact endpoints.
- `no-network`: default for offline research, but RSI-Harness still requires the operator to provide an explicit Agent API base URL or HTTP/HTTPS proxy so model-provider access can be resolved and pinned.

Verifier is always `no-network`. Build-time downloads do not imply Agent- or Verifier-runtime network permission; bake their results into the image with immutable provenance.

Do not place provider credentials in `task.toml`, Compose, Dockerfile, README, or assets. RSI-Harness supplies them at run time.

## Environment review before handoff

Confirm:

- image creation produces the exact approved starting state;
- the effective WORKDIR exists and contains only public starting artifacts;
- the Agent can run the public proxy/development commands without changing task definition;
- candidate-owned paths are inside the Judge-visible WORKDIR;
- the candidate is fully materialized on disk before submission and Judge reloads it without Work process state;
- non-root Judge evaluation works with the WORKDIR mounted read-only;
- Verifier dependencies are present before runtime;
- a prebuilt image declares no Docker volumes;
- common resources/env and Judge-only overrides match the confirmed execution plan;
- network and provider routing are operationally possible;
- no hidden or solution material enters the build context or image history.

Then freeze the interface and proceed to Verifier design.
