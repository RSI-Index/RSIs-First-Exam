# Blue Vela 单节点 Harbor Task 验证与运行 Skill 设计

日期：2026-09-01

状态：交互式设计已批准，等待书面设计确认

## 目标

新增仓库级 skill：

```text
.agents/skills/bluevela-single-harbor-tasks-validator/
```

它接受一个已经完成静态生成与编译检查的 RSI-Harness Harbor task，在 IBM
Blue Vela 上完成 execution-readiness 验证；验证通过后，立即沿同一套
single-node 运行契约启动并监控正式 Codex 轨迹到终态。

`harbor-task-validator` 是“验证什么”的权威；
`bluevela-single-node-running-harbor-tasks` 是“如何在集群执行”的唯一权威。

## 非目标

- 不新增或修改 RSI-Harness CLI、Blue Vela adapter、LSF launcher 或 task
  schema。
- 不创建直接调用 `bsub`、Docker、Podman、Apptainer 或 `harbor run` 的旁路。
- 不复制两份上游 skill 的完整 runbook，不维护第三套集群参数。
- 不把 single-node 不兼容的 task 静默缩小、改成 multi-node 或改变科学契约。
- 不把 dry-run、SIF 构建、Agent 启动或一条 Judge 日志误报为验证完成。

## 文件结构

```text
.agents/skills/bluevela-single-harbor-tasks-validator/
├── SKILL.md
└── references/
    └── validation-on-bluevela.md
```

不增加脚本或 UI metadata。`SKILL.md` 保持短小，负责触发条件、权威顺序、
工作流和终态；详细证据映射放进按需读取的 reference。

## 权威与冲突解析

新 skill 必须要求使用以下两个 sub-skill：

1. `harbor-task-validator`：资源授权、Environment acceptance、Baseline Judge
   acceptance、task-local 修复边界和 `EXECUTION READY` 证据标准。
2. `running-harbor-tasks-bluevela`，其仓库目录为
   `.agents/skills/bluevela-single-node-running-harbor-tasks`：登录节点限制、
   dry-run、compute-node image materialization、single-node LSF/Apptainer 执行、
   日志监控、重试和终态证据。

冲突时按以下所有权处理：

- 验证内容、奖励生命周期和通过条件由 validator 决定。
- 所有集群执行、调度、镜像构建、GPU 选择、路径和监控由 single-node
  runbook 决定。
- validator 的本地 Docker 命令不能在 Blue Vela login node 上执行；新 skill
  使用 native Harness/Blue Vela artifacts 证明同一验收事实。
- 如果 native Harness artifacts 无法证明某项 validator 必需事实，则返回
  `BLOCKED`，不得降低或跳过该检查。

## 输入与前置条件

输入是一个 absolute task directory。必须存在：

- `task.toml`
- `instruction.md`
- `environment/Dockerfile`
- `tests/test.sh`

Harness 必须解析为当前 Git worktree 下的 `RSI-Harness`，不得选 sibling
checkout。model、reasoning effort、primary reward、score direction 和有意设置的
submission cap 只能来自 task README、用户或已有运行配置，不得猜测。

## 工作流

### 1. 完整读取与只读规划

读取完整 task、两个 required sub-skill 及它们当前阶段要求的 references。
检查 task 的 build assets、网络、Work/Judge 资源、timeouts、reward 和起始
baseline。执行 single-node runbook 的 lightweight login checks 和
`rsi-harness run ... --cluster bluevela --dry-run`。

dry-run 必须解析并记录 Work/Judge GPU、single-node eligibility、CPU、memory、
local scratch、build/run walltime、GPFS paths、SIF cache 状态和命令参数。任何
profile-capacity 错误都 fail closed；不添加 local-only `--gpus`，不切换
multi-node profile。

### 2. 一次联合授权

在第一次 mutation 前披露一个 authorization envelope。它合并 validator 和
Blue Vela 所需信息：

- image/SIF、assets、cache、scratch、logs、retained runs 和一次合理修复重试
  的证据化 peak storage；
- build/runtime endpoints、Agent allowlist、Judge no-network、凭据和许可；
- Work/Judge GPU、single-node device pool、CPU、memory、storage、shm、timeouts
  和预计时长；
- exact task/image/output identities、collision-free run/log namespace 和 profile
  roots；non-dry-run 才能生成的 run ID、job ID 与 device selectors 在提交后
  立即记录，并且必须落在已批准的 namespace/device pool 内；
- task-local 修复与 fresh-run retry 范围；
- 只清理由本流程创建且已经 superseded 的资源，保留最终和失败审计证据。

展示 envelope 后停止，等待一次明确授权。授权后，范围内验证、正式轨迹和
普通重试不再二次暂停。

### 3. Native 集群验证与正式轨迹

删除 dry-run flag，使用完全相同的 scientific options 启动第一次真实
`rsi-harness run ... --cluster bluevela`。这是唯一 production-shaped
allocation：cache miss 时由 adapter 在 compute node 构建并校验 SIF；禁止在
login node 构建或手工打包 SIF。

该 allocation 同时承担 live compatibility validation 和正式 Codex 轨迹。
一旦健康，不得因为“验证已经完成”而取消并启动第二个 real run。只有运行中
出现有证据的 portability/runtime defect，才允许最小 task-local 修复并用新
UTC run ID 重试。

### 4. Validator 证据映射

详细映射写入 `references/validation-on-bluevela.md`。核心要求为：

- Environment gate：verified SIF identity、compute-node build success、未改动
  starting workspace、有效 public pre-scoring/editable-scope checks、依赖与
  assets 完整、无未声明 volume/network/host bind，以及声明的 WORKDIR/user/
  environment/shm 语义。
- Judge gate：声明的 Judge GPU/CPU/memory/network/timeout 下完整执行
  `/tests/test.sh`，输出边界正确，有限 scalar reward 恰好写入一次，失败、
  timeout 或 malformed reward 不得变成成功，Judge 进程退出且 GPU 释放。
- Harness gate：native Work、submission、Judge、feedback 和 artifact writer
  生命周期成立；不得用 Harbor CLI 日志替代。

当 validator 要求 unchanged baseline Judge，而当前 native run 没有产生可验证
的 unchanged-baseline execution 时，状态保持 `BLOCKED`；新 skill 不把任意
candidate submission 冒充 baseline evidence。

### 5. 失败处理与终态

失败时先按 single-node debugging reference 分类边界并保留完整 exact-job
evidence。一个 root-cause hypothesis 对应一个最小 source-controlled repair
和 focused regression。冻结的 run 不可原地修改；每次重试使用新 run ID。

终态只有：

- `END_TO_END_VALIDATED`：validator 要求和 Blue Vela/Harness success contract
  全部由 artifacts 证明，正式 Codex 轨迹已经完成且至少一次 Judge submission
  产生有效 finite reward。
- `FAIL`：在授权范围内穷尽安全 task-local 修复后仍存在可复现 task defect。
- `BLOCKED`：需要新凭据、quota、权限、profile/GPU 模型变更、multi-node、
  Harness 修改或科学契约决定。

`PEND`、`RUN` 和正常长时间静默不是终态。

## Single-node 容量边界

新 skill 不硬编码历史 GPU 数量；以 packaged profile 和 resolved dry-run 为
准。它必须验证 runbook 要求的 Work/Judge topology 与 single-node capacity。
如果当前 profile 拒绝 task 的解析资源，则报告 `BLOCKED`，不得缩减 GPU、
更换 task 字段或选择 multi-node 路径。

`depth-width-allocation-d13` 的实际结论只能由当次 dry-run 输出决定，不能用
旧集群经验或未验证的 GPU 复用假设代替。

## 测试设计

按 `writing-skills` 的 RED-GREEN-REFACTOR 流程测试。书面设计获批即授权在
无外部 mutation 的隔离 fresh-context Codex evaluations 中使用 subagents；
测试不得提交 LSF job 或启动真实 Harness run。

RED baseline 至少覆盖：

1. Agent 绕过 validator，只看 dry-run 就启动长跑。
2. Agent 在 login node 直接构建 Docker/SIF。
3. single-node capacity 失败后静默减 GPU 或切 multi-node。
4. Environment gate 通过后停止，没有继续正式轨迹。
5. 任意 candidate reward 被冒充 unchanged-baseline evidence。
6. 失败 run 被覆盖、复用或按模糊 job name 取消。

GREEN/REFACTOR 使用相同场景验证：权威分工、一次授权、native execution、
fail-closed 证据、健康 allocation 继续到终态以及 fresh-run retry 均被遵守。
最后运行 skill initializer/quick validator 提供的 `quick_validate.py`，并检查
frontmatter、链接、路径、占位符和 word count。

## 验收标准

- skill 名称和目录为 `bluevela-single-harbor-tasks-validator`。
- description 只描述触发条件，以 `Use when...` 开头。
- 新 skill 明确要求两个 sub-skill，不复制它们的完整内容。
- 集群 mutation 只通过 native `rsi-harness ... --cluster bluevela`。
- 未证明 validator gate 时不能继续或报告成功。
- 验证通过后同一个健康 production allocation 继续到正式轨迹终态。
- 所有场景测试、`quick_validate.py` 和链接检查通过。
