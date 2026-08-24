# RSI Harness

RSI Harness runs supported local Harbor GPU tasks. It creates one
persistent Work container for the Agent and a fresh Judge container whenever
the Agent asks for feedback. The Agent can submit multiple times while keeping
the same Work state.

## Requirements

- Linux with Docker and the NVIDIA Container Toolkit
- working `docker` and `nvidia-smi` commands
- Python 3.12 or 3.13
- permission to manage Docker and the host `DOCKER-USER` and `INPUT` iptables
  chains (usually by running the command with `sudo -E`)
- enough Docker storage for the task image and temporary Judge snapshots
- a supported local Harbor task directory

## Install

Install from a checkout:

```bash
uv tool install /path/to/RSI-Harness
```

Or build and install the wheel:

```bash
uv build
uv tool install dist/rsi_harness-0.1.0-py3-none-any.whl
```

The package includes the run engine and RSI Loop integration. No custom
`PYTHONPATH` is required.

## Configure the Agent

Provide an API key:

```bash
export RSI_AGENT_API_KEY="..."
```

Or reuse an existing Codex or Claude Code login:

```bash
sudo -E "$(command -v rsi-harness)" run /absolute/path/to/task --agent codex --agent-auth local --gpus 4,5
sudo -E "$(command -v rsi-harness)" run /absolute/path/to/task --agent claude-code --agent-auth local --gpus 4,5
```

When using a compatible API proxy, also set its base URL and optionally a
model:

```bash
export RSI_AGENT_API_BASE_URL="https://your-api.example/v1"
export RSI_AGENT_MODEL="your-model"
```

These values are passed only to the Agent process and are redacted from Engine
artifacts. Do not put credentials in the Harbor task directory.

## Run a Harbor task

Pass the task directory and the GPUs that this run is allowed to use:

```bash
TASK=/absolute/path/to/my-harbor-task

sudo -E "$(command -v rsi-harness)" run "$TASK" \
  --agent codex \
  --model gpt-5.5 \
  --reasoning-effort xhigh \
  --gpus 4,5 \
  --max-submissions 2 \
  --verbose
```

`--gpus` accepts host indexes, NVIDIA UUIDs, or a mixture. It is an ordered
pool authorized by the scheduler for this run; the Engine resolves it to
physical UUIDs, while Work sees only its task-declared count. Every UUID in
`--gpus` remains scheduler-owned for the run.

The task's GPU declaration is still enforced:

- `count: 2` gives Work the first two GPUs in the authorized pool.
- `count: all` uses every GPU listed in `--gpus`, not every GPU on the host.

Declare Judge GPUs separately when verification needs them:

```toml
[environment]
gpus = 2

[metadata.rsi_harness.verifier]
gpus = 1
```

Without the verifier declaration, Judge is CPU-only and Work is frozen while
it runs. When the pool has enough spare GPUs, Judge receives a disjoint slice.
When it does not, RSI Harness uses release-all mode: every Work GPU process
must release its GPU before Judge runs. That rejection is safe to retry and
does not consume `--max-submissions`.

The Engine reads the task's image or Dockerfile, timeouts, environment,
network mode, users, verifier, and GPU requirements automatically. Run
`rsi-harness run --help` for all optional controls.

`--reasoning-effort` is a generic per-run Agent option. Codex currently
supports `minimal`, `low`, `medium`, `high`, and `xhigh`; an Agent or value
that is not supported is rejected before Docker resources are created. If the
option is omitted, the Agent keeps its configured default.

Use `--disable-stop-hook` when the Agent should exit normally without the
RSI Loop stop hook.

A real run may be quiet for a long time while the image builds or the Agent is
working. The command prints the run ID, final status, Judge rounds, best score,
workspace location, and log location when it finishes.

## Run on Blue Vela

Blue Vela is available as a cluster profile on the same `run` command. First
inspect the fully resolved submissions without creating a run directory or
calling `bsub`:

```bash
uv run rsi-harness run "$TASK" \
  --cluster bluevela \
  --dry-run \
  --agent codex \
  --model gpt-5.6-sol \
  --reasoning-effort xhigh \
  --agent-auth local
```

Remove `--dry-run` to execute. The command waits synchronously for every stage.
It first reuses a SHA256-verified cached SIF or submits an LSF CPU build job
that performs Dockerfile → Podman/Docker archive → Apptainer SIF conversion in
node-local `/tmp`. It then submits one exclusive-process GPU job and returns
success only after the native RSI-Harness Engine and its expected artifacts
both validate.

GPU counts are read from the task, not from a cluster default. For example, a
task declaring two Work GPUs and two
`[metadata.rsi_harness.verifier]` GPUs requests exactly four GPUs on one node.
`--gpus` remains a local-device selector and is rejected with `--cluster`. A
task declaring `gpus = "all"` needs an explicit numeric override in its cluster
profile.

The packaged profile owns Blue Vela-specific queue, fair-share group,
Apptainer, cache, and GPFS paths. Each run gets a unique directory and exact LSF
job IDs; interruption never performs name-based or bulk cancellation. Cluster
artifacts retain the existing
`logs/runs/<run-id>/<task-id>/` leaf schema below the profile's GPFS logs root.
The Agent, each Judge round, and `RUN_INFO.json` remain in the unique run
directory. Logs are written live by `RunArtifactWriter`; they are not converted
from another harness after the run.

To migrate the same adapter, copy
`src/rsi_harness/cluster/bluevela/profile.toml`, change its paths and LSF policy,
then pass the new file:

```bash
uv run rsi-harness run "$TASK" --cluster /absolute/path/to/profile.toml ...
```

Cluster mode uses the same `RunCoordinator`, embedded `SubmissionService`,
`RSILoopAgentAdapter`, and multi-submission feedback loop as no-cluster mode.
The cluster layer owns only scheduling, image materialization, and Apptainer
Work/Judge process isolation. Harbor is used only to compile the task format;
the adapter never invokes `harbor run`.

## Feedback and rewards

The Agent calls `rsi-submit` when it wants feedback. RSI Harness then:

1. pauses Work;
2. creates a fresh Judge from the current Work state;
3. runs the task's `tests/test.sh`;
4. returns the test output and reward to the Agent; and
5. removes the Judge and resumes the same Work container.

No human action is needed between submissions. The number of submissions is
limited by `--max-submissions`.

The verifier should write a Harbor reward to
`/logs/verifier/reward.json`. A scalar `reward.txt` is also accepted. Use
`--primary-reward NAME` when the reward JSON contains multiple keys and one of
them should determine the best round.

Detailed task-authoring references: [`task.toml`
fields](docs/harbor-task-authoring/task-toml.md) and [Compose, Dockerfile, and
WORKDIR fields](docs/harbor-task-authoring/docker-compose.md).

## View results

The default data and log roots are `.rsi-harness` and `logs`; run artifacts are
written under `logs/runs/RUN_ID`. From the repository root, start the stock
RSI Loop visualizer with:

```bash
rsi-harness visualize
```

Then open `http://127.0.0.1:8000`.

## Recovery and cleanup

Normal multi-round runs manage their own resources. `recover` is only needed
after interruption, a host restart, or an infrastructure failure:

```bash
sudo -E "$(command -v rsi-harness)" recover RUN_ID
```

Omit `RUN_ID` to recover every unfinished recorded run.

A completed run keeps its final Work state as a Docker image, plus a managed
WORKDIR volume when split mode is used. Ordinary cleanup removes leftover
runtime resources but keeps that final state:

```bash
sudo -E "$(command -v rsi-harness)" cleanup RUN_ID
```

Delete the retained final state explicitly when it is no longer needed:

```bash
sudo -E "$(command -v rsi-harness)" cleanup RUN_ID \
  --delete-workspace --yes
```

## How it works

The Agent runs as root in Work and can modify project files, Python packages,
shells, `PATH`, system libraries, and other container files, matching the
expected Harbor workflow. Judge sees the Work filesystem at submission time,
but Judge writes never flow back into Work.

RSI Harness chooses the snapshot method from the effective WORKDIR:

- A non-root WORKDIR uses a managed Docker volume. Work mounts it read-write
  and Judge mounts it read-only, avoiding a new copy of large files stored
  there for each submission.
- A `/` WORKDIR uses a complete Docker root-filesystem snapshot. This is more
  compatible, but large Agent changes anywhere in the container can increase
  snapshot time and Docker storage use.

While Judge runs, Work processes are paused rather than destroyed. A CPU-only
Judge uses freeze-only mode. A GPU Judge uses a disjoint spare allocation when
available; otherwise release-all mode checks that every Work GPU process has
released its allocation before Judge starts. A release-all rejection can be
retried without consuming a submission.

Network access is taken from the Harbor task. Public, no-network, and supported
allowlist policies are compiled and enforced for Work and Judge.

## Supported task shape

The current Engine supports the common single-service Harbor GPU shape:

- Harbor schema 1.4 on Linux
- one continuous step and one `main` service
- NVIDIA GPU reservations
- one shared task environment
- the standard `tests/test.sh` verifier
- a local Docker build context or prebuilt image

Unsupported or unsafe features are rejected before the Agent starts. Examples
include sidecars, multiple task steps, task-authored Docker volumes, host
networking, privileged containers, arbitrary devices or capabilities, and an
independent verifier image.

The task source directory is never modified. Engine state and logs are written
only below the selected data and log roots.

## Important limitations

- Docker commit captures filesystem changes, not running process state.
- Task-authored Docker or Compose volumes are not supported because their
  contents are outside the committed container root filesystem.
- A `/` WORKDIR can produce large Judge snapshots when the Agent writes large
  files outside reusable image layers.
- Standard Harbor compatibility runs `test.sh` beside the code under test in
  Judge. It isolates Judge from Work, but it is not a separate secret-verifier
  protocol.
- The Engine fails closed when required Docker or firewall authority is
  unavailable; it does not silently run with weaker isolation.
