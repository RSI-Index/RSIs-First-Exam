# Blue Vela 原生 RSI-Harness 多节点设计

日期：2026-08-28

状态：已完成交互式设计确认，并已进入实施计划

## 目标

让标准 RSI-Harness `rsi-task` 在不改变任务格式、Harbor schema 或
Work/Judge 生命周期的前提下，通过以下命令运行在 IBM Blue Vela 的
多节点 LSF/Apptainer 环境中：

```bash
rsi-harness run /absolute/path/to/task \
  --cluster bluevela \
  --agent codex \
  --model "$RSI_MODEL"
```

当 task 的标准 GPU 字段超过 Blue Vela 单节点容量时，`bluevela` adapter
自动选择多节点逻辑。未指定 `--cluster bluevela`、使用其他 adapter、以及
现有 Blue Vela 单节点任务的行为保持不变。

本设计把“整个流程顺畅运行并完全兼容既有 RSI-Harness 设计”作为硬性
验收标准，而不是只完成 LSF resource request。

## 已确认的核心决策

1. task 唯一格式是标准 RSI-Harness `rsi-task` 格式。
2. task 使用 Harbor schema `1.4`、`rsi/<directory>` 名称和原生
   Work/snapshot/`rsi-submit`/Judge/reward 生命周期。
3. task 不包含 `cluster/`、`cluster/bluevela.toml`、任务自有 `bsub`、SIF
   发布脚本或嵌套 `harbor run`。
4. Work GPU 总数来自 `[environment].gpus`；Judge GPU 总数来自
   `[metadata.rsi_harness.verifier].gpus`。不存在第二套 GPU 配置。
5. Blue Vela 多节点第一版只使用完整的 8-GPU 节点。
6. Work 和 Judge 使用运行开始前就预留好的、永久不重叠的节点池；不复用
   GPU 或节点。
7. 一个顶层 LSF job 一次申请 Work nodes 与 Judge nodes 的总和。运行中
   不再提交第二个或嵌套的 `bsub`。
8. Blue Vela 的 LSF、Apptainer、GPFS、`blaunch`、runtime IPv4 和
   InfiniBand 行为属于 RSI-Harness adapter，不属于 task。

## 标准 task 结构

Blue Vela 多节点 task 保持以下结构：

```text
<task>/
├── task.toml
├── instruction.md
├── README.md
├── environment/
│   ├── Dockerfile
│   └── docker-compose.yaml    # 仅在标准 RSI-Harness 语义需要时存在
├── solution/
│   └── solve.sh
└── tests/
    ├── test.sh
    └── evaluator assets
```

多节点 Work 所需的公开训练辅助代码仍进入 `environment/` 并构建到共享
Environment。Judge-only 的分布式 evaluator、可信配置和隐藏资产仍位于
`tests/`。`solution/solve.sh` 仍只幂等物化 baseline，不训练、不评测、
不读取 `/tests`、不调用 `rsi-submit`、不写 reward。

任务确实需要 baseline manifest 时，其位置由 RSI-Harness 的信任边界决定：
可信评分权威放在 `tests/`，刻意对 Work 可见的公共证据才放进
`environment/`。`baselines/` 不是通用必需目录。

## 兼容性边界

### 保持不变的接口

- `HarborTaskCompiler` 继续只编译标准 task。
- 本地 Docker Engine、非 Blue Vela adapter 和普通 Harbor 字段不增加
  IBM 专用分支。
- `rsi-submit` 的同步提交、snapshot、反馈日志、最大提交次数和 reward
  读取协议不改变。
- `tests/test.sh` 仍是唯一 Judge 评分入口，成功结果仍为
  `/logs/verifier/reward.json`。
- 当前 Blue Vela 单节点 resource plan、LSF argv 和 runtime plan 由 golden
  regression test 固定。

### Blue Vela 多节点第一版的 eligibility gate

一个 task 只有在显式选择 `--cluster bluevela` 且总 GPU 需求超过单节点
容量时才进入新分支。新分支要求：

- Work GPU 为正且是 8 的倍数；
- 非零 Judge GPU 是 8 的倍数；
- 使用真实的 `environment/Dockerfile` build context；
- 使用非 `/` 的 split WORKDIR；
- 训练与 evaluator 有真实的分布式启动路径；
- task timeout、storage、CPU 和 memory 声明能够映射到一个合法的 LSF
  allocation。

这些约束是 Blue Vela 多节点运行资格，不是新的 task schema。当前 adapter
本来就不支持的 prebuilt-image-only 与 full-rootfs Blue Vela 形态不会被
伪装成已支持；标准 compiler 仍可编译它们，其他 backend 也不受影响。

## Dispatch 与资源推导

运行分派顺序固定为：

```text
未指定 cluster              -> 原有本地路径
cluster != bluevela         -> 对应原有 adapter
cluster == bluevela:
    max(work_gpus, judge_gpus) <= 8
                               -> 原有 Blue Vela 单节点路径
    max(work_gpus, judge_gpus) > 8
                               -> 新 Blue Vela 多节点路径
```

这里有意先保留现有 phase-sequential 单节点语义：例如 Work 8 GPU、Judge
4/8 GPU 的任务仍走已经存在的单节点 `RELEASE_ALL` 路径。只有 Work 或 Judge
某一个 phase 自身无法放入单个 8-GPU 节点时，才进入本设计的多节点分池
路径。这一优先级是“完全兼容既有 Blue Vela 行为”的回归边界。

多节点路径使用：

```text
work_nodes  = work_gpus / 8
judge_nodes = judge_gpus / 8       # judge_gpus == 0 时为 0
total_nodes = work_nodes + judge_nodes
```

例如：

```toml
[environment]
gpus = 32

[metadata.rsi_harness.verifier]
gpus = 16
```

推导为 4 个 Work 节点、2 个 Judge 节点和一次 6 节点 LSF allocation。

`[environment].cpus` 与 `[environment].memory_mb` 仍描述共享 `main`
service 的单实例需求。多节点 adapter 在每个参与节点运行一个 main
Environment 实例，因此每个节点使用 task 声明与 Blue Vela profile floor
两者中的较大值。LSF 总 CPU slots 按节点数乘以 per-node slots；memory
通过 Blue Vela 已验证的 per-host request 形态表达。多节点路径把标准
`storage_mb` 作为 GPFS split-WORKDIR 的共享容量计划，在提交前检查 run root
可用空间；每节点 node-local scratch 则由 Blue Vela profile floor 决定并在
allocation 内逐节点检查。现有单节点 `storage_mb -> local_tmp_mb` 行为不变。

## 单 allocation 与固定 pool

LSF scheduler 的多节点 request 必须表达：

- `total_nodes` 个完整 GPU 节点；
- 每节点 8 GPU；
- 每节点确定的 CPU slot 与 memory request；
- task timeout 加 Judge reserve 和 adapter operational margin；
- exclusive GPU process 模式；
- 一个顶层 `bsub`，无 nested scheduling。

allocation 启动后，adapter 解析 `LSB_MCPU_HOSTS` 的 host/slot 对，保留
LSF 给出的顺序，拒绝缺失、重复、slot 不符或节点数不符。主机列表按以下
规则一次性冻结：

```text
ordered_hosts[0:work_nodes]                  -> Work pool
ordered_hosts[work_nodes:total_nodes]        -> Judge pool
```

该分区写入 run control manifest 并参与 digest。运行期间不得重新排序、
扩缩容或跨 pool 借用节点。

## 多节点运行时

### Adapter-owned control plane

Blue Vela adapter 在 host 侧为 Work 与 Judge 分别建立 pool-scoped launcher
broker。broker 使用当前 allocation 内的 `blaunch` 和 Apptainer；它不是
task 文件，也不接受选择另一个 pool 的参数。

Work 容器只获得 Work broker endpoint，Judge 容器只获得 Judge broker
endpoint。完整 `LSB_MCPU_HOSTS`、Judge endpoint、可信 evaluator policy
和 `/tests` 不进入 Work。

用户的唯一入口仍然是：

```bash
rsi-harness run <task> --cluster bluevela
```

Harness 从标准 `task.toml` 的 Work/Judge GPU 字段自动判定单节点或多节点，
不要求用户再运行第二个 CLI。对于本版支持的 PyTorch 分布式 task，Blue
Vela runtime 在多节点容器的 `PATH` 前端透明注入 Harness-owned `torchrun`
shim。task 仍然使用标准命令：

```bash
torchrun --nnodes N --nproc-per-node 8 <script> <args>
```

shim 解析标准 `torchrun` 参数并把 `--nnodes N` 变成当前 phase pool 内的
原子 subpool lease；它通过 Harness 私有 broker 执行远端 launch，task 和
用户都不接触该 broker。并发请求的 subpool 不得重叠，节点不足时请求 fail
closed。每个远端节点最终执行真实的 `python -m torch.distributed.run`，并
由 runtime 注入：

- `RSI_NUM_NODES`；
- `RSI_NODE_RANK`；
- `RSI_POOL_SIZE`；
- `RSI_POOL_RANK`；
- `RSI_LOCAL_WORLD_SIZE=8`；
- runtime-derived `RSI_MASTER_ADDR`；
- adapter 分配且受 run identity 约束的 `RSI_MASTER_PORT`。

task 的公开训练脚本或 Judge evaluator 只使用普通 `torchrun`/PyTorch
distributed 接口。task 不解析 LSF host list、不直接调用 SSH/`blaunch`、
不硬编码 hostname/IP，也不提交 `bsub`。

标准 `torchrun --nnodes` 是当前 rsi-task 表达并发多规模实验所需的 subpool
能力；shim 不暴露 IBM hostname、`LSB_MCPU_HOSTS`、Harness broker 或另一个
phase 的拓扑。`--nnodes 1` 保持普通本地 torchrun 行为，`--nnodes > 1`
才进入透明多节点 launch。

### Work phase

Agent 与 RSI-Harness controller 位于 Work controller node 的 Work
Environment。普通命令保持单实例；标准 `torchrun --nnodes > 1` 由透明
shim fan out 到 Work pool。每个远端 Work 进程都由 broker 记录 PID、node、
argv、start/end 状态和 run identity。

### Snapshot 与提交

`rsi-submit` 保留现有同步协议。提交时按以下顺序执行：

1. 暂停 Agent 控制进程；
2. 要求 Work broker 证明所有远端 Work 命令已终止且输出已 flush；
3. 若仍有远端 writer，作为 retryable submission error 拒绝本次提交；
4. 在 GPFS run root 下创建该 round 的不可变 split-WORKDIR snapshot；
5. 将 snapshot identity 与 task、SIF、pool 和 round manifest 绑定；
6. 启动本轮 Judge；
7. Judge 完成并持久化反馈后释放本轮临时状态并恢复 Agent。

Work 和 Judge 节点虽然不复用，snapshot 一致性仍要求 Work 远端 writer 在
capture 前结束。

### Judge phase

`tests/test.sh` 只在 Judge controller 中启动。adapter 将同一只读 snapshot、
同一只读 task-owned `/tests` 和同一 SIF 提供给全部 Judge 节点。Judge 中的
标准多节点 `torchrun` 只能使用 Judge pool。每轮 Judge 获得新的临时目录、
端口和 process group；上一轮进程、socket、cache 或 GPU state 不得复用。

所有 Judge 节点完成后，controller 才允许 terminal evaluator 写一次有限
reward。Candidate-invalid scalar 仍由 task 自己的协议决定；adapter 不把
基础设施失败转成 candidate reward。

## 网络、存储与镜像

- SIF 仍由 adapter 在 compute node 从 task 的 `environment/Dockerfile`
  构建，并使用 content-addressed GPFS cache；不在 login node 构建。
- 每个节点从同一验证过 SHA-256 的 SIF 启动 Apptainer。
- Apptainer 使用 `--nv`、`--containall` 和明确的 GPFS、DNS、CUDA runtime、
  `/dev/infiniband`、`/sys/class/infiniband` binds。
- master address 只能从已分配 Work/Judge pool 的 controller hostname 在
  runtime 解析；拒绝 unspecified、loopback、link-local、multicast、多个
  候选和 `0.0.0.0` fallback。
- task 的 Work/Judge network policy 继续由 RSI-Harness 决定；host network
  inheritance 必须如实记录，不能描述为网络隔离。
- source、snapshot、tests 和 durable logs 位于 run-owned GPFS path；大量
  临时 cache 位于 node-local scratch，并按 run/round identity 清理。

## 失败分类与清理

### `bsub` 前失败

以下问题在占用 GPU 前作为 setup error 失败：

- 标准 task 编译失败；
- 不满足整节点 GPU 规则；
- Docker build context 或 split WORKDIR 不兼容；
- timeout/resource arithmetic 无法形成合法 allocation；
- Blue Vela profile、二进制或必需路径缺失。

### allocation/preparation 失败

Host inventory、GPU topology、SIF digest、Apptainer、CUDA、GPFS、runtime
IPv4、InfiniBand 或 launcher readiness 不满足时，不启动 Agent、不启动
Judge、不写 reward，并保留结构化基础设施诊断。

### runtime/Judge 失败

Scheduler、container、broker、snapshot、timeout、依赖、evaluator crash、
缺 case 或 incomplete aggregation 都不写 reward。只有 task verifier 已精确
识别的 candidate-owned invalid path 才能写已批准 scalar。

清理以 frozen run identity 为边界：终止本 run 的 Work/Judge process
groups、删除本 run 的 node-local scratch 和 round temp；不删除共享 SIF
cache、immutable data、其他 run 或 operator-owned assets。

## RSI-Harness validation 设计

### 静态与 compiler validation

1. RSI-Harness authoritative compiler；
2. Blue Vela adapter compatibility validation，仅在选择该目标时执行；
3. `rsi-harness run <task> --cluster bluevela --dry-run`，展示完整推导结果而
   不提交 LSF。

Dry-run 必须展示 Work/Judge GPU、节点数、per-node CPU/memory、总节点、
walltime、SIF cache plan、LSF argv 和固定 pool partition policy。

### 自动测试

- 原有 RSI-Harness 全套测试必须通过；
- 旧 Blue Vela 单节点 dry-run、LSF argv、GPU plan 和 runtime plan 使用
  golden regression 固定；
- 32 Work + 16 Judge 推导为 4 + 2 节点；
- 非整节点 multi-node request 在 `bsub` 前拒绝；
- LSF host 缺失、重复、slot 不符和数量不符均 fail closed；
- Work/Judge broker 越界请求被拒绝；
- 每个 phase 只能看到自己的 pool metadata；
- 远端 Work writer 存活时 `rsi-submit` 可重试地拒绝且不消费 submission；
- 所有 Judge 节点看到同一只读 snapshot、`/tests` 与 SIF；
- Judge failure/incomplete 不写 reward；
- cleanup 不触及共享或其他 run 的路径；
- 一个没有 `cluster/` 的标准 task 能完成 Blue Vela multi-node dry-run。

### Live validation

在明确授权后，RSI-Harness 集成验证使用 task 声明的完整拓扑，而不是缩小
资源的替代任务，依次验证：

1. compute-node SIF build/cache；
2. 一个顶层多节点 LSF allocation；
3. host inventory 与固定 pool partition；
4. 每节点 Apptainer/CUDA/InfiniBand/runtime IPv4；
5. Work pool 多节点命令与 checkpoint/output；
6. `rsi-submit` quiescence 和 snapshot；
7. Judge pool 的只读 snapshot、`/tests` 和多节点 evaluator；
8. valid、candidate-invalid、infrastructure、timeout 和 incomplete reward
   paths；
9. feedback、durable Engine artifacts 和 run-scoped cleanup。

实际运行命令只能是 `rsi-harness run ... --cluster bluevela`，不能调用
`harbor run`。

## 向后兼容测试门

实现只有在以下条件同时成立时才可合入：

- 非 Blue Vela 的 compiler/runtime 测试没有变化；
- 所有既有 Blue Vela 单节点测试通过；
- 单节点生成的 scheduler argv 与 RunPlan 保持既有语义；
- 多节点字段在 scheduler models 中都有保留旧行为的默认值；
- 标准 task 不需要新增或迁移文件；
- 没有 task 因未包含 `cluster/` 而失败；
- 只有 `--cluster bluevela` 且资源需要多个节点时才进入新逻辑；
- 多节点端到端验证保留 RSI-Harness 的 Work/Judge/snapshot/reward artifacts。

## 验收标准

本设计完成的定义是：

1. 满足 Blue Vela eligibility gate 的标准 `rsi-task` 无需结构转换即可
   运行；
2. 用户只需选择 `--cluster bluevela`，无需 task-owned cluster scripts；
3. adapter 自动识别单节点与多节点并选择对应路径；
4. 一个 allocation 预留永久隔离的 Work/Judge 完整节点池；
5. Work、snapshot、每次 `rsi-submit`、多节点 Judge、反馈和 reward 全流程
   完成；
6. 当前单节点 Blue Vela、其他 adapter 和本地 RSI-Harness 行为不回归；
7. live mismatch 被明确分类为 setup、infrastructure 或 task/runtime failure，
   绝不伪造成成功或 candidate reward。

此外，本次实现的具体 task 验收目标是：

```text
task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2
```

该目录最终必须是 Harbor schema 1.4 的 `rsi/*` task，并通过标准 compiler
和 Blue Vela multi-node dry-run。它不得保留 task-owned `cluster/` 运行入口、
`bluevela.toml`、`BLAUNCH`/`LSB_MCPU_HOSTS` contract 或预构建 SIF 作为唯一
Environment 来源；其多规模 Work 与 terminal Judge 必须改用
标准 `torchrun`。任务的公开 baseline 证据可保留，但 `baselines/` 不是格式
要求，权威 Judge 资产应按 RSI-Harness trust boundary 放入 `tests/`。

## 实施范围划分

后续实现计划应拆成三个有序工作包，全部位于 RSI-Harness 与标准
`rsi-task` 兼容范围内：

1. RSI-Harness Blue Vela adapter/scheduler 的资源模型、host partition、
   pool-scoped launcher 与 scheduler regression tests；
2. RSI-Harness Blue Vela multi-node Work、snapshot、Judge runtime 与集成
   tests；
3. 将
   `task_collect/zf_tasks/pre-training/optimizer_update_geometry_stepmatched_v2`
   落成标准多节点 `rsi-task`，完成 compiler、dry-run 与经授权的完整 Blue
   Vela end-to-end run。

任何工作包都不得通过修改 task schema、添加 task-owned `cluster/` 或绕过
RSI-Harness Engine 来缩短实现路径。
