# Blue Vela RSI-Harness Multi-Node and Step-Matched Task Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task with review checkpoints.

**Goal:** Extend the existing RSI-Harness Blue Vela adapter so an unchanged Harbor schema-1.4 `rsi-task` can run multi-node Work and multi-node Judge phases in one LSF allocation, then migrate `task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2` to that standard format.

**Architecture:** Keep the compiler, local engine, other cluster adapters, and the current Blue Vela single-node branch unchanged. Add a Blue Vela-only multi-node resource plan, one multi-host LSF request, immutable Work/Judge host partitions, phase-scoped `blaunch` brokers, and a transparent Harness-owned `torchrun` shim. The user invokes only `rsi-harness run <task> --cluster bluevela`; the adapter reads standard GPU fields and dispatches automatically. The native coordinator remains the owner of Agent, synchronous `rsi-submit`, split-WORKDIR snapshot, Judge, feedback, reward, and artifacts.

**Tech Stack:** Python 3.12, Pydantic 2, pytest, Bash, Harbor schema 1.4, RSI-Harness native coordinator, IBM Spectrum LSF, Apptainer, GPFS, `blaunch`, CUDA/NCCL/InfiniBand.

**Spec:** [Blue Vela native RSI-Harness multi-node design](../specs/2026-08-28-bluevela-rsi-harness-multinode-design.md)

## Global Constraints

- Treat the current dirty worktree as contributor-owned. Before every commit, inspect `git status --short` and stage only files named by that task. Never reset, checkout, delete, or rewrite unrelated changes.
- Preserve the current single-node dispatch exactly: `max(work_gpus, verifier_gpus) <= profile.resources.gpus_per_node` uses the existing `ClusterResources`, `RunGPUPlan`, `span[hosts=1]`, and `RELEASE_ALL` behavior. In particular, Work 8 + Judge 4/8 remains single-node.
- Enter the new branch only when either phase itself exceeds one node. Multi-node Work and Judge pools are allocated together, disjoint for the entire run, and never loan nodes to each other.
- Use only standard task GPU fields: `[environment].gpus` and `[metadata.rsi_harness.verifier].gpus`. Do not add task-local Blue Vela configuration.
- Do not add `cluster/`, `bluevela.toml`, task-owned `bsub`, task-owned `blaunch`, nested `harbor run`, or a prebuilt SIF-only Environment to the migrated task.
- The top-level allocation is the only scheduler mutation. All remote commands run inside it with `blaunch` owned by RSI-Harness.
- A setup, allocation, broker, snapshot, Judge, or aggregation failure must not create a candidate reward.
- Do not submit the 59-node live job until the user explicitly authorizes that resource-consuming run. Static tests and `--dry-run` are mandatory and non-mutating.

## File Map

### RSI-Harness files to create

- `RSI-Harness/src/rsi_harness/cluster/bluevela/resources.py` — compatible single/multi resource planning.
- `RSI-Harness/src/rsi_harness/cluster/bluevela/allocation.py` — LSF inventory parsing, node probes, immutable pool partition, and allocation digest.
- `RSI-Harness/src/rsi_harness/cluster/bluevela/multinode.py` — phase-scoped host broker and subpool lease authority.
- `RSI-Harness/src/rsi_harness/cluster/bluevela/torchrun_shim.py` — transparent parser/client for standard fixed-size `torchrun` commands.
- `RSI-Harness/src/rsi_harness/cluster/bluevela/remote_worker.py` — fixed-shape per-host Apptainer launcher and run-scoped cleanup.
- `RSI-Harness/tests/cluster/bluevela/test_allocation.py` — inventory, partition, IPv4, CUDA, and digest tests.
- `RSI-Harness/tests/cluster/bluevela/test_multinode.py` — broker isolation, leasing, cancellation, and cleanup tests.
- `RSI-Harness/tests/cluster/bluevela/test_torchrun_shim.py` — standard torchrun parsing, request, stream, and cancellation tests.
- `RSI-Harness/tests/fixtures/tasks/minimal-bluevela-multinode/` — small standard schema-1.4 compile/dry-run fixture with no `cluster/` directory.
- `RSI-Harness/docs/bluevela-multinode.md` — operator and task-author contract.

### RSI-Harness files to modify

- `RSI-Harness/src/rsi_harness/cluster/config.py`
- `RSI-Harness/src/rsi_harness/cluster/bluevela/profile.toml`
- `RSI-Harness/src/rsi_harness/cluster/schedulers/lsf.py`
- `RSI-Harness/src/rsi_harness/cluster/bluevela/adapter.py`
- `RSI-Harness/src/rsi_harness/cluster/bluevela/engine.py`
- `RSI-Harness/src/rsi_harness/cluster/bluevela/runtime.py`
- `RSI-Harness/src/rsi_harness/cli.py`
- `RSI-Harness/tests/cluster/bluevela/test_resources.py`
- `RSI-Harness/tests/cluster/schedulers/test_lsf.py`
- `RSI-Harness/tests/cluster/bluevela/test_adapter.py`
- `RSI-Harness/tests/cluster/bluevela/test_engine.py`
- `RSI-Harness/tests/cluster/bluevela/test_runtime.py`
- `RSI-Harness/tests/test_cli.py`

### Step-matched task files to create or rewrite

- `task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2/task.toml`
- `task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2/README.md`
- `task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2/environment/Dockerfile`
- `task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2/environment/source.lock.json`
- `task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2/environment/task-data/optimizer_adamh_baselines.json`
- `task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2/environment/task-tools/run_optimizer_node.sh`
- `task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2/environment/task-tools/run_optimizer_scaling_ladder.sh`
- `task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2/environment/task-tools/run_optimizer_candidate_ladder.sh`
- `task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2/solution/solve.sh`
- `task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2/tests/baselines/optimizer_adamh_baselines.json`
- `task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2/tests/baselines/PROVENANCE.md`
- `task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2/tests/test.sh`
- `task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2/tests/evaluate.py`
- `task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2/tests/test_task_contract.py`

### Obsolete step-matched task files to remove with `apply_patch`

- `cluster/allocate.sh`
- `cluster/blaunch_proxy.py`
- `cluster/blaunch_proxy_client.py`
- `cluster/bluevela.toml`
- `cluster/check_storage.py`
- `cluster/cleanup.sh`
- `cluster/cluster_ready.sh`
- `cluster/dry_run.sh`
- `cluster/nccl_smoke.py`
- `cluster/node_ready.py`
- `cluster/publish_sif.sh`
- `cluster/resolve_config.py`
- `cluster/run_current_allocation.sh`
- `cluster/run_experiment.py`
- `environment/Apptainer.def`
- `environment/build.sh`
- `environment/container.env`
- `environment/materialize_project.sh`
- `environment/task-tools/host_lease.py`
- `baselines/PROVENANCE.md`
- `baselines/optimizer_adamh_baselines.json`

---

## Task 1: Freeze the current single-node contract and add resource-plan types

**Files:**

- Create: `RSI-Harness/src/rsi_harness/cluster/bluevela/resources.py`
- Modify: `RSI-Harness/src/rsi_harness/cluster/bluevela/adapter.py`
- Modify: `RSI-Harness/src/rsi_harness/cluster/config.py`
- Modify: `RSI-Harness/src/rsi_harness/cluster/bluevela/profile.toml`
- Test: `RSI-Harness/tests/cluster/bluevela/test_resources.py`
- Test: `RSI-Harness/tests/cluster/bluevela/test_adapter.py`

### Step 1: Add failing dispatch and arithmetic tests

Add tests that assert all of the following:

```python
def test_legacy_eight_plus_eight_stays_single_node() -> None:
    plan = derive_resource_plan(definition(work=8, judge=8), profile())
    assert plan.single_node is not None
    assert plan.multi_node is None
    assert plan.single_node.total_gpus == 8


def test_thirty_two_plus_sixteen_becomes_four_plus_two_nodes() -> None:
    plan = derive_resource_plan(definition(work=32, judge=16), profile())
    assert plan.single_node is None
    assert plan.multi_node.work.node_count == 4
    assert plan.multi_node.verifier.node_count == 2
    assert plan.multi_node.total_nodes == 6


@pytest.mark.parametrize(("work", "judge"), ((9, 8), (16, 9)))
def test_multinode_rejects_partial_gpu_nodes(work: int, judge: int) -> None:
    with pytest.raises(SetupError, match="whole 8-GPU nodes"):
        derive_resource_plan(definition(work=work, judge=judge), profile())
```

Retain the existing exact dictionary assertion for the 2 + 2 single-node dry-run. This is the regression lock that prevents new optional fields from appearing in old output.

Run:

```bash
cd RSI-Harness
uv run pytest tests/cluster/bluevela/test_resources.py tests/cluster/bluevela/test_adapter.py -q
```

Expected: the new tests fail because `derive_resource_plan` and multi-node models do not exist; all pre-existing tests still pass.

### Step 2: Move resource logic behind a compatible public facade

Implement these frozen Pydantic models in `resources.py`:

```python
class ClusterResources(PersistedModel):
    work_gpus: int = Field(ge=0)
    verifier_gpus: int = Field(ge=0)
    total_gpus: int = Field(ge=0)
    cpu_slots: int = Field(gt=0)
    memory_mb: int = Field(gt=0)
    local_tmp_mb: int = Field(ge=0)
    build_walltime: str
    run_walltime: str


class PhaseResources(PersistedModel):
    gpu_count: int = Field(ge=0)
    node_count: int = Field(ge=0)


class MultiNodeResources(PersistedModel):
    work: PhaseResources
    verifier: PhaseResources
    total_nodes: int = Field(gt=1)
    gpus_per_node: int = Field(gt=0)
    cpu_slots_per_node: int = Field(gt=0)
    memory_mb_per_node: int = Field(gt=0)
    shared_workspace_mb: int = Field(ge=0)
    node_tmp_mb: int = Field(ge=0)
    build_walltime: str
    run_walltime: str


class BlueVelaResourcePlan(PersistedModel):
    single_node: ClusterResources | None = None
    multi_node: MultiNodeResources | None = None

    @model_validator(mode="after")
    def _exactly_one_branch(self) -> Self:
        if (self.single_node is None) == (self.multi_node is None):
            raise ValueError("exactly one Blue Vela resource branch is required")
        return self
```

Keep `derive_resources()` byte-for-byte equivalent to the present single-node arithmetic. Add `derive_resource_plan()` that resolves numeric phase GPU counts once, chooses single-node by `max(work, verifier) <= gpus_per_node`, and otherwise requires every non-zero phase count to be divisible by `gpus_per_node`.

Add `min_runtime_tmp_mb: int = Field(default=0, ge=0)` to `ResourceProfile`; set the packaged Blue Vela value to `71680`. In the multi-node branch, map standard `storage_mb` to `shared_workspace_mb` and the profile floor to `node_tmp_mb`. Do not reinterpret storage in the legacy branch.

Re-export `ClusterResources`, `MultiNodeResources`, `BlueVelaResourcePlan`, `derive_resources`, and `derive_resource_plan` from `adapter.py` so current imports remain valid.

### Step 3: Make resource tests pass

Run:

```bash
cd RSI-Harness
uv run pytest tests/cluster/bluevela/test_resources.py tests/cluster/bluevela/test_adapter.py -q
```

Expected: all tests pass, including the unchanged exact single-node dry-run payload.

### Step 4: Commit only the resource slice

```bash
git status --short
git add RSI-Harness/src/rsi_harness/cluster/bluevela/resources.py \
  RSI-Harness/src/rsi_harness/cluster/bluevela/adapter.py \
  RSI-Harness/src/rsi_harness/cluster/config.py \
  RSI-Harness/src/rsi_harness/cluster/bluevela/profile.toml \
  RSI-Harness/tests/cluster/bluevela/test_resources.py \
  RSI-Harness/tests/cluster/bluevela/test_adapter.py
git commit -m "feat: plan Blue Vela multinode resources"
```

---

## Task 2: Extend LSF submission without changing single-host argv

**Files:**

- Modify: `RSI-Harness/src/rsi_harness/cluster/schedulers/lsf.py`
- Test: `RSI-Harness/tests/cluster/schedulers/test_lsf.py`

### Step 1: Add an exact multi-host golden test

Construct a 59-node spec with 8 slots and 8 GPUs per host. Assert the exact relevant argv:

```python
spec = LSFJobSpec(
    name="mt-rsi-optimizer-run-alice",
    queue="priority",
    group="grp_models",
    cpu_slots=472,
    memory_mb=262144,
    walltime="72:15",
    stdout_path=(tmp_path / "lsf.%J.out").resolve(),
    stderr_path=(tmp_path / "lsf.%J.err").resolve(),
    script_path=script.resolve(),
    gpu_count=8,
    local_tmp_mb=71680,
    one_host=False,
    hosts=59,
    slots_per_host=8,
    memory_per_host=True,
    exclusive=True,
)
argv = LSFScheduler().render_submit(spec)
assert "select[tmp>=71680] span[ptile=8] rusage[mem=262144]" in argv
assert argv[argv.index("-gpu") + 1] == "num=8:mode=exclusive_process"
assert "-x" in argv
```

Keep `test_gpu_submit_argv_is_explicit_and_single_host` unchanged and exact.

Run:

```bash
cd RSI-Harness
uv run pytest tests/cluster/schedulers/test_lsf.py -q
```

Expected: the multi-host construction fails validation because the new fields do not exist.

### Step 2: Add validated multi-host scheduler fields

Extend `LSFJobSpec` with backwards-compatible defaults:

```python
hosts: int = Field(default=1, gt=0)
slots_per_host: int | None = Field(default=None, gt=0)
memory_per_host: bool = False
exclusive: bool = False
```

Add a model validator with these rules:

- `one_host=True` requires `hosts == 1` and `slots_per_host is None`.
- `one_host=False` requires `hosts > 1`, a non-null `slots_per_host`, and `cpu_slots == hosts * slots_per_host`.
- `memory_per_host=True` is legal only on the multi-host form.

Render the old path exactly as today. On the multi-host path render `span[ptile=N]`; append `rusage[mem=M]` when requested; append `-x` when requested. Continue using Blue Vela's already evidenced `-gpu num=8:mode=exclusive_process`, which LSF applies per selected host with `span[ptile=8]`.

### Step 3: Run the scheduler regressions

```bash
cd RSI-Harness
uv run pytest tests/cluster/schedulers/test_lsf.py -q
```

Expected: both exact single-host and multi-host argv tests pass.

### Step 4: Commit

```bash
git add RSI-Harness/src/rsi_harness/cluster/schedulers/lsf.py \
  RSI-Harness/tests/cluster/schedulers/test_lsf.py
git commit -m "feat: render Blue Vela multi-host LSF jobs"
```

---

## Task 3: Parse and freeze the allocation before starting the Agent

**Files:**

- Create: `RSI-Harness/src/rsi_harness/cluster/bluevela/allocation.py`
- Create: `RSI-Harness/tests/cluster/bluevela/test_allocation.py`

### Step 1: Write fail-closed inventory tests

Cover valid ordering and every rejected shape:

```python
raw = "work-a 8 work-b 8 judge-a 8"
inventory = parse_lsb_mcpu_hosts(raw, expected_hosts=3, expected_slots=8)
assert tuple(item.host for item in inventory) == ("work-a", "work-b", "judge-a")

@pytest.mark.parametrize(
    "raw,match",
    (
        ("", "missing"),
        ("node-a", "HOST SLOTS pairs"),
        ("node-a x", "integer slots"),
        ("node-a 4 node-b 8", "expected 8 slots"),
        ("node-a 8 node-a 8", "duplicate host"),
        ("node-a 8", "expected 3 hosts"),
    ),
)
def test_invalid_inventory_is_rejected(raw: str, match: str) -> None:
    with pytest.raises(InfrastructureError, match=match):
        parse_lsb_mcpu_hosts(raw, expected_hosts=3, expected_slots=8)
```

Also test that Work gets the first `work.node_count` entries, Judge gets the remainder, neither pool overlaps, one unique non-loopback IPv4 is required per host, every host exposes exactly eight unique CUDA selectors, and changing any host/probe value changes the SHA-256 allocation digest.

### Step 2: Implement immutable allocation models

Use these interfaces:

```python
class InventoryHost(PersistedModel):
    host: str
    slots: int = Field(gt=0)


class AllocatedNode(PersistedModel):
    host: str
    slots: int = Field(gt=0)
    ipv4: IPvAnyAddress
    cuda_devices: tuple[str, ...]


class AllocatedPools(PersistedModel):
    run_id: str
    work: tuple[AllocatedNode, ...]
    verifier: tuple[AllocatedNode, ...]
    gpus_per_node: int = Field(gt=0)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
```

Implement the concrete public interfaces
`parse_lsb_mcpu_hosts(raw: str, *, expected_hosts: int, expected_slots: int) -> tuple[InventoryHost, ...]`
and
`probe_and_partition(*, run_id: str, resources: MultiNodeResources, inventory: tuple[InventoryHost, ...], runner: CommandRunner) -> AllocatedPools`.

The runner executes fixed `blaunch -z HOST` probes for `getent ahostsv4`, `CUDA_VISIBLE_DEVICES`, `/dev/infiniband`, `/sys/class/infiniband`, GPFS visibility, SIF SHA-256, and node-local free space. Normalize duplicate DNS rows to one unique address, then reject zero or multiple unique safe IPv4 values. Reject loopback, unspecified, link-local, multicast, and non-IPv4 results.

Partition only after every node passes. Serialize the ordered pools with sorted JSON keys and compute the digest from that serialization. Write no manifest in this module; the adapter owns durable control state.

### Step 3: Run tests

```bash
cd RSI-Harness
uv run pytest tests/cluster/bluevela/test_allocation.py -q
```

Expected: all parser, probe, partition, and digest tests pass with a fake command runner; no LSF command is executed on the development host.

### Step 4: Commit

```bash
git add RSI-Harness/src/rsi_harness/cluster/bluevela/allocation.py \
  RSI-Harness/tests/cluster/bluevela/test_allocation.py
git commit -m "feat: freeze Blue Vela host pools"
```

---

## Task 4: Build the phase-scoped broker and transparent `torchrun` shim

**Files:**

- Create: `RSI-Harness/src/rsi_harness/cluster/bluevela/multinode.py`
- Create: `RSI-Harness/src/rsi_harness/cluster/bluevela/torchrun_shim.py`
- Create: `RSI-Harness/src/rsi_harness/cluster/bluevela/remote_worker.py`
- Create: `RSI-Harness/tests/cluster/bluevela/test_multinode.py`
- Create: `RSI-Harness/tests/cluster/bluevela/test_torchrun_shim.py`

### Step 1: Test transparent standard `torchrun` behavior first

Test these accepted task commands:

```bash
torchrun --nnodes 4 --nproc-per-node 8 train.py --arg value
torchrun --nnodes=4 --nproc_per_node=8 --no-python /task-tools/train.sh
```

`--nnodes 1` must exec local `python -m torch.distributed.run` without contacting the broker. For `--nnodes > 1`, the shim must derive the subpool size and contact only the already-injected current-phase endpoint. Reject elastic `MIN:MAX` nodes, zero/negative nodes, an oversized fixed node count, explicit `--node-rank`, task-provided rendezvous endpoint, hostname, LSF variable, pool, or phase selection.

Add broker tests proving:

- `torchrun --nnodes 4` atomically leases four currently free nodes.
- Concurrent requests receive disjoint subpools.
- A fifth-node request while only four are free fails without starting `blaunch`.
- A Work broker cannot target any Judge host and vice versa.
- Shim-provided `LSB_*`, `LSF_*`, `APPTAINER_*`, `SINGULARITY_*`, Agent credentials, submit token, and proxy secrets are removed.
- Every selected node receives relative `RSI_NODE_RANK`, absolute `RSI_POOL_RANK`, `RSI_NUM_NODES`, `RSI_POOL_SIZE`, `RSI_LOCAL_WORLD_SIZE=8`, a pool-local master IPv4, and a run/request-derived port in 20000-49999.
- Cancellation terminates all exact request processes, runs remote cleanup for those nodes only, and releases the subpool.
- `require_idle()` reports active Work writers and succeeds only after outputs and status files are fsynced.

### Step 2: Implement a closed protocol

Define frozen request/result models:

```python
class MultiNodeRequest(PersistedModel):
    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    run_id: str
    pool_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    node_count: int | None = Field(default=None, gt=0)
    argv: tuple[str, ...] = Field(min_length=1)
    environment: dict[str, str]


class RankResult(PersistedModel):
    pool_rank: int = Field(ge=0)
    node_rank: int = Field(ge=0)
    returncode: int


class MultiNodeResult(PersistedModel):
    request_id: str
    ranks: tuple[RankResult, ...]
    cancelled: bool = False
    timed_out: bool = False
```

Use a run-owned GPFS directory with owner-only permissions and `requests`, `results`, `heartbeats`, and `cancel` subdirectories. Publish requests and terminal status by write-fsync-rename. The shim streams bounded per-rank output in deterministic rank order and exits zero only when all ranks return zero.

The shim sends only normalized `torch.distributed.run` argv, the fixed `--nnodes` count, safe working-directory identity, and a filtered environment. It cannot send hosts or scheduler commands. The broker removes task-provided rendezvous/rank fields, selects nodes under a lock, and creates only this host command shape:

```python
(
    "blaunch", "-z", node.host,
    sys.executable, "-m", "rsi_harness.cluster.bluevela.remote_worker",
    "run", "--control", str(frozen_control),
    "--request", request.request_id,
    "--pool-rank", str(pool_rank),
    "--node-rank", str(node_rank),
)
```

`remote_worker.py` loads the frozen broker-owned JSON rather than trusting arbitrary host argv. It creates `/tmp/rsi-harness/<run-hash>/<phase>/<request>/<rank>` with mode 0700, runs the fixed Apptainer command with the broker-owned SIF, binds, local CUDA selectors, and phase environment, records the exact child process group, and removes only that request's node-local directory. Its `stop` subcommand accepts the same run/request identity and kills only the recorded process group.

### Step 3: Run protocol tests

```bash
cd RSI-Harness
uv run pytest \
  tests/cluster/bluevela/test_multinode.py \
  tests/cluster/bluevela/test_torchrun_shim.py -q
```

Expected: all tests pass using fake `Popen` and fake rank output; no real `blaunch` or Apptainer command runs.

### Step 4: Commit

```bash
git add RSI-Harness/src/rsi_harness/cluster/bluevela/multinode.py \
  RSI-Harness/src/rsi_harness/cluster/bluevela/torchrun_shim.py \
  RSI-Harness/src/rsi_harness/cluster/bluevela/remote_worker.py \
  RSI-Harness/tests/cluster/bluevela/test_multinode.py \
  RSI-Harness/tests/cluster/bluevela/test_torchrun_shim.py
git commit -m "feat: add transparent Blue Vela torchrun launcher"
```

---

## Task 5: Connect resource planning, LSF, allocation, and Engine payloads

**Files:**

- Modify: `RSI-Harness/src/rsi_harness/cluster/bluevela/adapter.py`
- Modify: `RSI-Harness/src/rsi_harness/cluster/bluevela/engine.py`
- Modify: `RSI-Harness/src/rsi_harness/cli.py`
- Modify: `RSI-Harness/tests/cluster/bluevela/test_adapter.py`
- Modify: `RSI-Harness/tests/cluster/bluevela/test_engine.py`
- Modify: `RSI-Harness/tests/test_cli.py`

### Step 1: Add adapter and Engine failures

Add a 32 + 16 dry-run test asserting:

```python
assert dry_run["multi_node"] == {
    "work": {"gpu_count": 32, "node_count": 4},
    "verifier": {"gpu_count": 16, "node_count": 2},
    "total_nodes": 6,
    "gpus_per_node": 8,
    "cpu_slots_per_node": 8,
    "memory_mb_per_node": 65536,
    "shared_workspace_mb": 15360,
    "node_tmp_mb": 71680,
    "build_walltime": "02:00",
    "run_walltime": "02:15",
}
assert "span[ptile=8]" in dry_run["run_argv"]
assert "span[hosts=1]" not in dry_run["run_argv"]
assert dry_run["pool_policy"] == "ordered Work prefix; ordered Judge suffix"
```

Add Engine tests for a multi-node payload with host-qualified devices such as `work-a:GPU-0` and `judge-a:GPU-0`. Assert the resulting `RunGPUPlan` is disjoint and `JudgeGPUMode.DISJOINT`. Keep every current single-node Engine test unchanged.

### Step 2: Add an optional multi-node payload field

Extend `EnginePayload` only with:

```python
multi_node: MultiNodeResources | None = None
```

For single-node runs, serialize `None`, use `bind_lsf_devices()`, and render the current driver exactly. For multi-node runs:

1. build one placeholder `GPUDevice` per `(phase, node, local_gpu)`;
2. set Work and Judge allocations to disjoint placeholder slices;
3. render one multi-host run driver;
4. parse `LSB_MCPU_HOSTS` inside the allocation;
5. probe and freeze `AllocatedPools` before `run_native_engine()`;
6. bind placeholder UUIDs to `HOST:LOCAL_SELECTOR` for durable Engine artifacts;
7. persist `control/ALLOCATED_POOLS.json` atomically and add its SHA-256 to `RUN_INFO.json`.

The multi-node driver must not require one controller-wide `CUDA_VISIBLE_DEVICES` list. It must verify the controller is the first frozen Work host and delegate per-host CUDA checks to `probe_and_partition()`.

### Step 3: Build the multi-host spec only in the new branch

For `MultiNodeResources`, `_run_spec()` must use:

```python
LSFJobSpec(
    name=name,
    queue=self.profile.scheduler.queue,
    group=self.profile.scheduler.group,
    cpu_slots=resources.total_nodes * resources.cpu_slots_per_node,
    memory_mb=resources.memory_mb_per_node,
    walltime=resources.run_walltime,
    stdout_path=(run_dir / "run" / "lsf.%J.out").resolve(),
    stderr_path=(run_dir / "run" / "lsf.%J.err").resolve(),
    script_path=script.resolve(),
    gpu_count=resources.gpus_per_node,
    one_host=False,
    hosts=resources.total_nodes,
    slots_per_host=resources.cpu_slots_per_node,
    memory_per_host=True,
    exclusive=True,
    local_tmp_mb=resources.node_tmp_mb,
)
```

The build job remains one-host. Before submitting the run job, check free GPFS bytes at `run_root` against `shared_workspace_mb`; do not encode shared workspace capacity as LSF node-local `tmp`.

In dry-run and CLI output, add Work nodes, Judge nodes, total nodes, per-node CPU/memory, shared workspace, node scratch, and the exact multi-host argv only when `multi_node` is present. The old console output remains exact for single-node.

### Step 4: Run the integration slice

```bash
cd RSI-Harness
uv run pytest \
  tests/cluster/bluevela/test_resources.py \
  tests/cluster/schedulers/test_lsf.py \
  tests/cluster/bluevela/test_allocation.py \
  tests/cluster/bluevela/test_adapter.py \
  tests/cluster/bluevela/test_engine.py \
  tests/test_cli.py -q
```

Expected: all tests pass; no scheduler submission occurs in dry-run tests.

### Step 5: Commit

```bash
git add RSI-Harness/src/rsi_harness/cluster/bluevela/adapter.py \
  RSI-Harness/src/rsi_harness/cluster/bluevela/engine.py \
  RSI-Harness/src/rsi_harness/cli.py \
  RSI-Harness/tests/cluster/bluevela/test_adapter.py \
  RSI-Harness/tests/cluster/bluevela/test_engine.py \
  RSI-Harness/tests/test_cli.py
git commit -m "feat: dispatch native Engine across Blue Vela nodes"
```

---

## Task 6: Integrate the Work pool while preserving single-node runtime

**Files:**

- Modify: `RSI-Harness/src/rsi_harness/cluster/bluevela/runtime.py`
- Modify: `RSI-Harness/tests/cluster/bluevela/test_runtime.py`

### Step 1: Add Work runtime tests

Test both constructor paths:

- `allocated_pools=None` produces the current exact Apptainer command, GPU selectors, injected `rsi-submit`, and network behavior.
- A multi-node Work runtime starts only a Work broker, shadows `torchrun` with `/usr/local/bin/torchrun`, binds only the private Work endpoint, and never exposes `LSB_MCPU_HOSTS`, Judge hosts, Judge endpoint, or `/tests`.
- A normal Agent command remains one controller container; only fixed-size standard `torchrun --nnodes > 1` fans out.
- Work worker binds use the writable GPFS workspace, read-only SIF, public task assets, run-scoped node tmp, DNS, `/dev/infiniband`, and `/sys/class/infiniband`.
- Agent stop closes the Work broker and cleans exact active requests without touching shared SIF cache or another run.

### Step 2: Add allocation-aware runtime state

Change the constructor to:

```python
def __init__(
    self,
    payload: Any,
    plan: RunPlan,
    allocated_pools: AllocatedPools | None = None,
) -> None:
```

Keep the existing fields and methods for `None`. For multi-node:

- use the first Work node's local CUDA selectors for the Agent controller smoke and Agent container;
- copy `torchrun_shim.py` to `/usr/local/bin/torchrun` with mode 0755, while remote ranks execute the real `/opt/venv/bin/python -m torch.distributed.run` to prevent recursion;
- start `MultiNodeBroker(phase="work", run_id=payload.run_id, nodes=allocated_pools.work, pool_digest=allocated_pools.digest, root=self.control / "multinode/work", sif_path=payload.sif_path, profile=self.profile, workspace=self.workspace, workdir=self.plan.workdir)` during `initialize()`;
- bind its client directory into Work and add only `RSI_MULTINODE_ROOT` and the Work pool digest;
- build every remote Work Apptainer command from the same SIF, workspace, task environment, and network mode as the controller;
- record broker events through the native Engine event callback.

Do not put cluster branches in the compiler or generic coordinator.

### Step 3: Run runtime tests

```bash
cd RSI-Harness
uv run pytest tests/cluster/bluevela/test_runtime.py \
  tests/cluster/bluevela/test_multinode.py -q
```

Expected: new Work tests and all current single-node runtime tests pass.

### Step 4: Commit

```bash
git add RSI-Harness/src/rsi_harness/cluster/bluevela/runtime.py \
  RSI-Harness/tests/cluster/bluevela/test_runtime.py
git commit -m "feat: run Blue Vela Work subpools"
```

---

## Task 7: Enforce remote quiescence, snapshot once, and run Judge on its pool

**Files:**

- Modify: `RSI-Harness/src/rsi_harness/cluster/bluevela/runtime.py`
- Modify: `RSI-Harness/src/rsi_harness/cluster/bluevela/engine.py`
- Modify: `RSI-Harness/tests/cluster/bluevela/test_runtime.py`
- Modify: `RSI-Harness/tests/cluster/bluevela/test_engine.py`

### Step 1: Write lifecycle tests before implementation

Add tests that assert this exact order:

```text
pause Agent
require Work broker idle and fsynced
copy immutable GPFS workspace snapshot
start fresh Judge broker
blaunch Judge controller on first Judge host
run /tests/test.sh
wait for all Judge subpool requests
read reward only after successful verifier exit
stop Judge broker and remove round tmp
unpause Agent
```

Also assert:

- An active remote Work request raises `RetryableSubmissionError`, restores the submission budget, creates no snapshot, starts no Judge, writes no report, and unpauses the Agent.
- Every Judge node gets the same read-only snapshot, `/tests`, and SIF digest.
- The Judge controller has no GPU selectors; its transparent `torchrun` workers receive only Judge GPUs.
- Judge timeout, missing rank result, nonzero rank, incomplete aggregation, or missing `reward.json` leaves the round rewardless.
- A second submission creates a new Judge broker root, new master port, new node tmp, and new immutable snapshot.
- Cleanup paths are descendants of the exact run/round identity.

### Step 2: Reuse the existing retry protocol

In `NativeJudgeEvaluator.evaluate()`, immediately after pausing Work, call `work_broker.require_idle()`. Convert its active-writer error to the already-supported `RetryableSubmissionError` from `rsi_harness.errors`; do not create a `SubmissionReport`. The existing `SubmissionService` then rolls back `rounds_allocated`.

For multi-node Judge, replace the local `_run()` call with a Judge-controller launch on `allocated_pools.verifier[0]`. Mount:

```text
snapshot -> RunPlan.workdir                 read-only
task/tests -> /tests                        read-only
verifier log dir -> /logs/verifier         read-write
Judge broker protocol root -> /run/rsi-harness/torchrun read-write
round-local node tmp -> /tmp                read-write
```

Start a new `MultiNodeBroker(phase="verifier", run_id=payload.run_id, nodes=allocated_pools.verifier, pool_digest=allocated_pools.digest, root=round_root / "multinode", sif_path=payload.sif_path, profile=profile, workspace=snapshot, workdir=plan.workdir)` per round. The controller runs the fixed `("/bin/bash", "/tests/test.sh")`; fixed-size `torchrun` commands can lease only Judge nodes. Stop and attest every rank before reading reward. Never translate infrastructure failure to scalar zero.

### Step 3: Pass lifecycle tests

```bash
cd RSI-Harness
uv run pytest \
  tests/runtime/test_submissions.py \
  tests/cluster/bluevela/test_runtime.py \
  tests/cluster/bluevela/test_engine.py -q
```

Expected: retry budget, snapshot isolation, Judge pool, rewardless infrastructure failures, and legacy runtime tests all pass.

### Step 4: Commit

```bash
git add RSI-Harness/src/rsi_harness/cluster/bluevela/runtime.py \
  RSI-Harness/src/rsi_harness/cluster/bluevela/engine.py \
  RSI-Harness/tests/cluster/bluevela/test_runtime.py \
  RSI-Harness/tests/cluster/bluevela/test_engine.py
git commit -m "feat: judge Blue Vela snapshots on disjoint nodes"
```

---

## Task 8: Add a small standard rsi-task fixture and full Harness regressions

**Files:**

- Create: `RSI-Harness/tests/fixtures/tasks/minimal-bluevela-multinode/task.toml`
- Create: `RSI-Harness/tests/fixtures/tasks/minimal-bluevela-multinode/instruction.md`
- Create: `RSI-Harness/tests/fixtures/tasks/minimal-bluevela-multinode/environment/Dockerfile`
- Create: `RSI-Harness/tests/fixtures/tasks/minimal-bluevela-multinode/solution/solve.sh`
- Create: `RSI-Harness/tests/fixtures/tasks/minimal-bluevela-multinode/tests/test.sh`
- Modify: `RSI-Harness/tests/cluster/bluevela/test_adapter.py`

### Step 1: Create the fixture with standard fields only

Use this resource contract:

```toml
schema_version = "1.4"

[task]
name = "rsi/minimal-bluevela-multinode"

[environment]
os = "linux"
workdir = "/workspace"
gpus = 16
cpus = 8
memory_mb = 65536
storage_mb = 15360
network_mode = "no-network"

[agent]
timeout_sec = 60
network_mode = "public"

[verifier]
timeout_sec = 30
network_mode = "no-network"

[metadata.rsi_harness.verifier]
gpus = 8
```

The task must have no `cluster/`, no Blue Vela metadata, no IBM command, and no RSI-specific launcher command. Its public scripts use ordinary fixed-size `torchrun`; the adapter transparently supplies the current phase's nodes.

### Step 2: Compile and dry-run the fixture

Add an adapter test asserting standard compiler output becomes Work 2 nodes + Judge 1 node and one 3-node LSF request. Assert the frozen task tree has no generated `cluster/` directory.

Run:

```bash
cd RSI-Harness
uv run pytest tests/task tests/cluster/bluevela -q
```

Expected: compiler and all Blue Vela tests pass.

### Step 3: Run the whole Harness suite and lint

```bash
cd RSI-Harness
uv run pytest -q
uv run ruff check src tests
```

Expected: zero failures and zero lint errors. Any unrelated pre-existing failure must be documented with its exact command/output and must not be hidden by narrowing the test set.

### Step 4: Commit

```bash
git add RSI-Harness/tests/fixtures/tasks/minimal-bluevela-multinode \
  RSI-Harness/tests/cluster/bluevela/test_adapter.py
git commit -m "test: cover standard Blue Vela multinode tasks"
```

---

## Task 9: Convert `optimizer_update_geometry_stepmatched_v2` to schema 1.4

**Files:**

- Rewrite/create the step-matched files listed in the File Map.
- Remove the obsolete files listed in the File Map with `apply_patch`; do not use a recursive deletion command.

### Step 1: Lock the current scientific inputs before restructuring

Record SHA-256 values for the current baseline manifest, project overlay files, task tools, evaluator, source commits, tokenizer revision, and Paloma revision in `environment/source.lock.json`. Use:

```json
{
  "bridge_commit": "9e5b537db01db83e94264e3f6f35e9ba89101e41",
  "mcore_commit": "58bf14e9e68915a5e0c7d70451620e6e713c07d5",
  "tokenizer_revision": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
  "paloma_revision": "65cd6fc59dba021b21db414fa5e8d7765ffbe5e6",
  "source_remote": "https://github.com/RSI-Index/Megatron-Bridge.git"
}
```

Add the computed file digests under a sorted `assets` object. A contract test must recompute and compare them.

### Step 2: Rewrite `task.toml` as the actual 59-node rsi-task

Use standard schema fields and no second GPU configuration:

```toml
schema_version = "1.4"

[task]
name = "rsi/optimizer_update_geometry_stepmatched_v2"
description = "Discover one scale-general optimizer that improves exact-update Paloma BPB and fixed-window pre-training loss across a locked 550M-to-2.545B AdamH ladder."

[environment]
os = "linux"
workdir = "/workspace"
gpus = 256
cpus = 8
memory_mb = 262144
storage_mb = 921600
network_mode = "public"
build_timeout_sec = 10800

[agent]
timeout_sec = 244800
network_mode = "public"

[verifier]
timeout_sec = 14400
network_mode = "public"

[metadata.rsi_harness.verifier]
gpus = 216
```

The 32 Work nodes preserve the six-rung 27-node training wave plus five-node online evaluation capacity. The 27 Judge nodes independently cover the largest terminal six-rung wave. Together they produce one 59-node LSF allocation; neither phase borrows nodes from the other.

Set Work paths to `/workspace/project` and `/workspace/output`. Remove `docker_image`, `BLAUNCH`, `BLAUNCH_PROXY_ROOT`, LSF variables, and `/app` artifact assumptions.

### Step 3: Replace the Apptainer build with a Docker build context

Resolve the immutable base digest during implementation:

```bash
skopeo inspect docker://nvcr.io/nvidia/pytorch:26.06-py3 \
  --format '{{.Digest}}'
```

Put the returned `sha256:` value directly in the Dockerfile `FROM`; fail the task conversion if the digest cannot be resolved. The Dockerfile must:

1. clone `https://github.com/RSI-Index/Megatron-Bridge.git` and checkout the exact Bridge commit;
2. initialize/fetch Megatron-LM and checkout the exact MCore commit;
3. apply the checked-in optimizer overlay without network access after clone;
4. install only lockfile- or exact-version dependencies;
5. copy `/opt/project` to the seed `/workspace/project` and create `/workspace/output`;
6. install public task tools at `/task-tools` and public baseline data at `/task-data`;
7. set the existing offline Hugging Face/W&B and NCCL environment values;
8. verify source hashes and imports in a final `RUN` gate.

Do not call Apptainer, access `/proj`, or publish a SIF from the Dockerfile. The Blue Vela adapter converts this standard Docker context to its content-addressed SIF on a compute node.

### Step 4: Place baseline evidence on the correct trust sides

Copy the public aggregate baseline manifest to `environment/task-data/optimizer_adamh_baselines.json` for Work diagnostics. Put the independently certified manifest and provenance under `tests/baselines/` for Judge authority. `tests/test.sh` must hash both inputs and reject a mismatch before evaluation; it must not read `/run-contract/BASELINE_CERTIFIED.json` or any task-external certificate path.

If live baseline certification has not yet produced the exact locked Judge manifest, keep the task fail-closed and do not claim end-to-end readiness. Task 12 defines the certification gate.

### Step 5: Make `solution/solve.sh` idempotent and standard

The solution must only restore the locked AdamH starter into `/workspace/project`, initialize `/workspace/output/SOLUTION_BASELINE.json`, and record source hashes. Running it twice must produce the same bytes. It must not train, evaluate, read `/tests`, invoke `rsi-submit`, or write reward.

### Step 6: Remove obsolete task-owned cluster machinery

Delete each obsolete file listed in the File Map through an explicit `apply_patch` deletion. Confirm:

```bash
test ! -e task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2/cluster
! rg -n 'bluevela\.toml|\bbsub\b|\bblaunch\b|LSB_MCPU_HOSTS|BLAUNCH_PROXY|harbor run|\.sif' \
  task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2
```

Expected: both commands exit zero.

### Step 7: Commit the structural conversion

Run the static contract test first, then stage only this task directory:

```bash
uv run --project RSI-Harness pytest -q \
  task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2/tests/test_task_contract.py
git add task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2
git commit -m "feat: convert optimizer step-matched task to RSI format"
```

---

## Task 10: Replace task-owned host leasing with standard `torchrun`

**Files:**

- Create: `task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2/environment/task-tools/run_optimizer_node.sh`
- Modify: `task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2/environment/task-tools/optimizer_scaling_ladder_profile.sh`
- Modify: `task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2/environment/task-tools/run_optimizer_scaling_ladder.sh`
- Modify: `task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2/environment/task-tools/run_optimizer_candidate_ladder.sh`
- Modify: `task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2/environment/task-tools/online_eval_controller.py`
- Modify: `task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2/tests/evaluate.py`
- Modify: `task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2/tests/test.sh`
- Modify: `task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2/tests/test_task_contract.py`

### Step 1: Add task contract tests for the new launcher

Assert that public scripts:

- contain standard `torchrun --nnodes` and `--nproc-per-node`;
- contain no LSF, `blaunch`, SSH, Apptainer, hostname, or Blue Vela path logic;
- derive torchrun values only from `RSI_NUM_NODES`, `RSI_NODE_RANK`, `RSI_LOCAL_WORLD_SIZE`, `RSI_MASTER_ADDR`, and `RSI_MASTER_PORT`;
- request E0/E1/E2 = 1 node, E3/E4 = 4 nodes, and E5 = 16 nodes;
- can run the six full rungs concurrently using 27 Work nodes while a separate online-evaluation controller can use the five remaining Work nodes;
- terminal full wave requests 27 Judge nodes and terminal reduced wave requests 18 Judge nodes;
- never read `/tests` from Work scripts.

### Step 2: Implement one in-container node entrypoint

`run_optimizer_node.sh` runs inside the broker-launched Environment on one node. It must require the injected values:

```bash
: "${RSI_NUM_NODES:?}"
: "${RSI_NODE_RANK:?}"
: "${RSI_LOCAL_WORLD_SIZE:?}"
: "${RSI_MASTER_ADDR:?}"
: "${RSI_MASTER_PORT:?}"
```

Then execute one local torchrun worker group:

```bash
exec /opt/venv/bin/torchrun \
  --nnodes "${RSI_NUM_NODES}" \
  --nproc-per-node "${RSI_LOCAL_WORLD_SIZE}" \
  --node-rank "${RSI_NODE_RANK}" \
  --master-addr "${RSI_MASTER_ADDR}" \
  --master-port "${RSI_MASTER_PORT}" \
  /workspace/project/examples/training/optimizer/runtime/pretrain_gpt_marin_adamh.py \
  "${training_args[@]}"
```

Keep the existing exact-update scale profiles, checkpoint locations, optimizer candidate import, NCCL/IB settings, and failure markers. Remove the nested-container branch from the materialized project launcher.

### Step 3: Replace host leases with broker subpool leases

In `run_optimizer_scaling_ladder.sh`, replace `host_lease.py`, `LSB_MCPU_HOSTS`, manual master port, and `$BLAUNCH` with:

```bash
torchrun --nnodes "${EXPECTED_NODES}" --nproc-per-node 8 --no-python \
  /task-tools/run_optimizer_node.sh \
  --contract "${run_dir}/run_contract.json"
```

Because the transparent shim and broker allocate subpools atomically, the six background scale processes remain concurrent and disjoint without learning hostnames. Preserve the current cleanup traps and checkpoint pruning; terminating `torchrun` signals the shim, so the broker performs exact remote cleanup.

### Step 4: Make Judge recompute authoritative operands

Change `tests/evaluate.py` so terminal Judge does not trust Work-owned `trusted_fixed_window_loss`. Every full and reduced checkpoint evaluation must produce both Paloma BPB and the fixed-window loss in Judge-owned logs. Run each terminal scale through standard fixed-size `torchrun --nnodes` using the same node counts as Work, then aggregate only after every requested rank and every scale is complete.

Candidate-invalid policy/source/checkpoint failures may write the approved scalar `0.0`. Missing dependency, launcher failure, rank failure, timeout, incomplete metric, baseline mismatch, or evaluator crash exits nonzero without `reward.json`.

### Step 5: Run task tests

```bash
uv run --project RSI-Harness pytest -q \
  task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2/tests/test_task_contract.py
bash -n \
  task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2/solution/solve.sh \
  task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2/tests/test.sh \
  task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2/environment/task-tools/*.sh
```

Expected: static contracts and all shell syntax checks pass.

### Step 6: Commit

```bash
git add task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2
git commit -m "feat: launch optimizer task through RSI multinode"
```

---

## Task 11: Compile and dry-run the real task through RSI-Harness

**Files:**

- Modify: `RSI-Harness/tests/cluster/bluevela/test_adapter.py`
- Modify: `task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2/README.md`
- Create: `RSI-Harness/docs/bluevela-multinode.md`

### Step 1: Add the real task as an acceptance fixture

Compile the repository task with `HarborTaskCompiler` and assert:

```python
assert definition.task_id == "rsi/optimizer_update_geometry_stepmatched_v2"
assert definition.gpu_requirement.count == 256
assert definition.verifier.gpu_count == 216
assert definition.workdir == PurePosixPath("/workspace")
assert definition.service.build_context.name == "environment"
```

Run the adapter in dry-run mode with a fake scheduler and exact agent-version resolver. Assert Work 32 nodes, Judge 27 nodes, total 59 nodes, 472 CPU slots, 262144 MB per host, 8 GPUs per host, 921600 MB shared workspace, `span[ptile=8]`, `rusage[mem=262144]`, and one run submission argv. Assert dry-run creates neither the run root nor any task file.

### Step 2: Document the only supported user workflow

`RSI-Harness/docs/bluevela-multinode.md` and the task README must show:

```bash
rsi-harness run \
  /u/yuetai/more_task/RSI-Index-Public/task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2 \
  --cluster bluevela \
  --agent codex \
  --model "$RSI_MODEL" \
  --dry-run
```

Explain that users run only `rsi-harness ... --cluster bluevela`; task authors use standard GPU fields and ordinary fixed-size `torchrun`. LSF, `blaunch`, Apptainer, GPFS, hostnames, IPv4, InfiniBand, SIF cache, Work/Judge partition, transparent shim, and cleanup remain adapter-owned.

### Step 3: Run acceptance tests

```bash
cd RSI-Harness
uv run pytest tests/cluster/bluevela/test_adapter.py -q
uv run rsi-harness run \
  /u/yuetai/more_task/RSI-Index-Public/task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2 \
  --cluster bluevela \
  --agent codex \
  --model gpt-5.6-sol \
  --dry-run
```

Expected dry-run output includes:

```text
Work GPUs:        256
Verifier GPUs:    216
Work nodes:       32
Verifier nodes:   27
Total nodes:      59
CPU slots/node:   8
Memory MB/node:   262144
```

The command must not call `bsub` or mutate the task/run roots.

### Step 4: Run final static regression

```bash
cd RSI-Harness
uv run pytest -q
uv run ruff check src tests
cd ..
uv run --project RSI-Harness pytest -q \
  task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2/tests/test_task_contract.py
git diff --check
```

Expected: all commands exit zero.

### Step 5: Commit

```bash
git add RSI-Harness/tests/cluster/bluevela/test_adapter.py \
  RSI-Harness/docs/bluevela-multinode.md \
  task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2/README.md
git commit -m "docs: certify RSI multinode dry-run workflow"
```

---

## Task 12: Perform authorized Blue Vela certification and end-to-end run

**Files produced by the run:**

- `RUN_INFO.json`
- `control/ALLOCATED_POOLS.json`
- Engine `final_result.json`, `agent_output.txt`, and `run_agent.log`
- submission report and feedback artifacts
- Judge verifier logs and reward for valid candidates only
- baseline certification evidence used by `tests/baselines/`

### Step 1: Stop at the resource authorization gate

Before any live command, show the user the dry-run topology and ask for explicit authorization to submit one 59-node allocation for up to 72 hours 15 minutes. Do not infer authorization from approval of this implementation plan.

### Step 2: Certify the pinned AdamH baseline if needed

Use the exact task commits, six scales, update counts, dataset revisions, and evaluator paths locked in `source.lock.json`. Store raw run artifacts outside the task, then copy only the finite aggregate manifest and provenance into `tests/baselines/` after verifying their SHA-256 values. Never synthesize `BASELINE_CERTIFIED.json` from the current offline manifest.

Re-run the task contract, compiler, and dry-run after installing certified assets.

### Step 3: Run the actual standard command

After authorization:

```bash
rsi-harness run \
  /u/yuetai/more_task/RSI-Index-Public/task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2 \
  --cluster bluevela \
  --agent codex \
  --model "$RSI_MODEL"
```

Do not invoke task cluster scripts, `harbor run`, or a second `bsub`.

### Step 4: Verify the live evidence

Require all of the following before declaring completion:

1. One build job only on cache miss and one 59-node run job.
2. Frozen ordered host inventory with 32 Work and 27 Judge hosts and no overlap.
3. Eight visible GPUs, one safe IPv4, SIF digest match, GPFS, and InfiniBand on every node.
4. Concurrent Work subpool leases sum to at most 32 nodes and never use Judge hosts.
5. A live writer causes retryable `rsi-submit` rejection without consuming budget.
6. A completed Work phase snapshots once to GPFS.
7. Terminal evaluator runs only through the Judge pool and sees read-only snapshot/tests.
8. Infrastructure and incomplete paths create no reward.
9. A valid path creates one finite `reward.json`, feedback, submission report, and completed final result.
10. Cleanup removes only this run's processes and node tmp; shared SIF cache and other runs remain.

### Step 5: Final verification and handoff

```bash
cd /u/yuetai/more_task/RSI-Index-Public/RSI-Harness
uv run pytest -q
uv run ruff check src tests
cd ..
git status --short
git log --oneline --max-count=12
```

Report the exact run ID, LSF job IDs, task path, log path, pool digest, test counts, and any remaining unrelated dirty files. Only then claim the multi-node implementation and the new rsi-task are complete.

---

## Plan Self-Review Checklist

- [ ] Every approved design requirement maps to at least one implementation task and one test.
- [ ] Work 8 + Judge 4/8 remains on the current single-node path.
- [ ] Work 256 + Judge 216 becomes exactly 32 + 27 = 59 disjoint nodes.
- [ ] Standard task fields are the only resource source.
- [ ] Standard `torchrun --nnodes` transparently supports concurrent, disjoint subpool leases without exposing LSF hosts or another user CLI.
- [ ] Work quiescence is checked before snapshot and is retryable without submission-budget loss.
- [ ] Judge uses a fresh pool-scoped broker and a read-only snapshot/tests bind each round.
- [ ] The real task has schema 1.4, a Dockerfile, no `cluster/`, and no prebuilt-SIF-only path.
- [ ] Public and Judge baseline assets follow the trust boundary and are certified rather than fabricated.
- [ ] Single-node, other adapters, compiler, coordinator, and local runtime regressions run unchanged.
- [ ] Live `bsub` remains behind explicit user authorization.
- [ ] Every code block is an executable example or a complete interface contract; no deferred implementation marker remains.
