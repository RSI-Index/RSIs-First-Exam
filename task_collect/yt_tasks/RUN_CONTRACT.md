# 五个环境的运行与科研审计契约

更新日期：2026-07-25。active task 的科学 source of truth 是 pinned 作者仓库；
本文区分仓库发布内容、缺失公开资产和当前 `envs/tasks` 的基础设施适配。

逐项源码差异、pinned repository 与 current environment 的对应关系见
[`SETUP_FIDELITY.md`](./SETUP_FIDELITY.md)。

## 1. 资产位置和当前完整性

正确的 GPFS 前缀是 `/proj/datasets`，不是 `/proj/dataset`。本项目当前根目录是：

```text
/proj/datasets/interns/yuetai/agent_envs/scale_autoresearch_tasks
```

在 2026-07-17 原始 snapshot staging 完成、DeepOCR 派生转换开始前，主要占用约为：
`images/` 77 GiB、`hf-cache/` 179 GiB、
`runtime/` 81 GiB、`cache/` 11 GiB；同一份官方权重在 `hf-cache` 和 `runtime`
之间使用 hard link，因此当时根目录实际总占用约 264 GiB，不能把这些子目录数字直接相加。
Stage-2 解包/PNG 转换会继续增加 `data/`，所以 264 GiB 是有时间点的审计值，不是容量上限。
八个 8.5--13.6 GB 的 SIF 在
`images/`；模型和数据快照在 `hf-cache/`；之前 verifier smoke 的候选模型、
输出和日志在 `runtime/`。因此“大资产在这个 GPFS 根下面”是对的，但当前还没有
做到 input assets 与历史 run outputs 完全分层，不能把 `runtime/*/output/model`
误当成新的官方训练资产。

旧的资产清单只看目录是否存在，因此曾把只有 `refs/main` 的
Qwen3-30B-A3B/easy_deepocr 目录和只有 `README.md` 的 OOLONG 目录误报成完整缓存。
2026-07-17 已改成检查 snapshot 中的真实权重、数据 shard、文件数量/大小和 GDN
manifest 内容。严格 asset 检查后五项输入资产均通过；active GDN 已沿 NVlabs README 声明的
Samba 上游完成公开预 tokenized SlimPajama staging 和 versioned manifest。DeepOCR 已有
easy_deepocr、sam_clip_ckpt、OmniDocBench、
固定 revision 的完整原始 `olmOCR-mix-1025` snapshot（95 files / 77,114,799,826 bytes），
以及由它转换并由 job 221471 严格验证的 LLaVA JSON/PNG 训练目录与 manifest。旧 GDN 的
tokenizer/PG19 snapshot 只是 `gated_deltanet_old` 历史资产，不是 active GDN 输入。全部 9 个
OOLONG validation shards 已封存并在 SIF 内以 `HF_HUB_OFFLINE=1` 实读成功；RLM 当前固定
筛选实际得到完整 50 条 trec_coarse@131072 样本。

SuperBPE 固定 `UW/olmo-mix-1124-subset-p99@64b9a7c...` 中 repository `meta.json`
规定的精确 10GB 顺序，并把 `train/70.txt` 的 557,641,266-byte prefix 单独物化和 hash。
agent hard-link view 只有这 11 个训练对象；verifier hard-link view 只有 pinned
`eval/135.txt`（944,259,457 bytes，SHA256 `81e2f413...`）。两类角色不能看到对方的数据路径，
完整清单为 `asset_manifests/superbpe-official-64b9a7c5.json`。

Qwen3-30B-A3B、easy_deepocr 和 sam_clip 的 runtime 权重不是按目录名直接当作资产：所有
safetensors 先逐文件匹配对应 HF revision 的 LFS SHA256 和 byte size，再 hard-link 到正式
snapshot；非权重 metadata 从该 revision 重新取得。机器可读记录在
`asset_manifests/promoted_snapshots-20260717.json`。

full-run preflight 要求 GDN 在
`data/gated_deltanet-official/slimpajama/manifest.json` 固定 Samba README 链接的
`jsun/slimpajama_Llama2_Tokenizer@cdc67c...`、20 个 LFS transport parts、解压后同一
`packed/slim/` 下的 `train_slim*` 与 `validation*` 文件、完整 SHA256 index、15B nominal
budget、14,999,879,680 processed tokens、2,097,152 tokens/update、7,152 updates 和
4×8 topology 下每 rank 14,305 micro-iterations。staging 时全量读/hash 一次；每次运行前验证
sealed records、精确文件集合和 64 个确定性抽样文件，避免每次 GPU 提交前重读超过 1 TB；
DeepOCR 要求固定 revision 的原始 `olmOCR-mix-1025` 经 upstream 脚本转成 144-DPI PNG 和
LLaVA JSON：267,962 raw 中 5,752 条缺失文本，262,210 条进入 renderer；upstream 对
219 个异常超大 MediaBox 报 `PyMuPDF Overly large image` 并跳过，最终 261,991 rows。
manifest 固定 source/conversion revision、所有计数、失败原因和 skipped-path digest。
转换完成后还要求 JSON 中 261,991 个 image path 唯一并与
PNG tree 精确相等，逐文件检查 PNG signature、IHDR、非零宽高，并记录 validated count/index
hash；preflight 重算 JSON SHA256、PNG 总数和完整 index hash。缺件时 launcher 会在提交 GPU
job 前失败，不会偷偷在线下载，
也不会把 verifier-only run 命名为 full run。`oolongbench/oolong-synth` 整仓约
11.99 GB；当前 official train/eval 都读 validation split，9 个 validation shards 约
1.99 GB，preflight 要求这 9 个 shard 全部真实存在，不能由 README 代替。

preflight 还比较两个 SIF 与其 definition、workspace 和 tests source 的 mtime；任一 image
早于当前输入就标为 stale 并禁止 full run。这避免在源码/策略已经修正后继续悄悄使用旧 SIF。

## 2. Web search 和网络边界

完整 run 必须由 `lsf_validation/submit_full_run.sh` 提交。它固定使用 Codex 并传入：

```text
--ak web_search=disabled
```

run 完成后，driver 再扫描 ATIF trajectory；只要出现
`tool_name = web_search_call`，该 run 的审计状态就是失败。五份 policy 也把外部
web research 列为 hard-zero，并要求 provenance 写明 `web_search: disabled`。full-run
launcher 将提交时冻结到该 run `control/` 中的 task policy 以只读方式挂到 verifier 的
`/tests/policy.yaml`，因此 policy 生效不依赖旧 SIF 中曾经 bake 的 YAML 副本。

`task.toml` 仍然写 `network_mode = "public"`，不是忘了关网。当前自定义
LSF+Apptainer backend 共享宿主网络 namespace，Harbor 会拒绝一个后端无法兑现的
`no-network` 声明；同时 Codex 模型 API 和 verifier policy judge 需要控制面网络。
所以当前保证是“关闭 Codex web search + 禁止把外网作为研究输入 + HF 离线加载”，
不是物理 egress sandbox。若需要硬隔离，必须换支持 egress control 的 Harbor provider
或由集群侧提供独立网络 namespace/firewall。

已归档 `rlm_old` 的最早 verifier smoke 在这套 full-run launcher/严格 preflight 完成之前执行；其
`oolong.log` 明确出现 unauthenticated Hugging Face Hub request，那个 raw 0.2400 结果只保留为
作废的计算路径历史。归档 verifier 后来在 `HF_HUB_OFFLINE=1` 下从封存的
9 个 validation Parquet 完成旧版 25 条筛选：raw OOLONG 0.200010，平均 855.56 subcalls /
9.96 root iterations，旧 cost reward 0.004569。该结果使用强制 temperature 0 和已废弃的
task-authored cost scalar，不是 active RLM baseline。旧 job 233113/238576 的 25-row
zero-update/60-update training 结果也只作历史校准，不能替代当前 matched evaluator。当前
pooled qualification job 304037 在 model-default sampling 下完成 50 rows×5=250 rollouts：
trusted raw mean `0.25718349663568113`、11 timeout 计零、0 error、0 reward mismatch。

## 3. GPU、trajectory 与优化次数

每个完整 run 只有一个 Harbor trial、一个 Codex agent trajectory。GDN 不设 6h task-level
agent timeout；默认 submit wrapper 为它请求可配置的 24h scheduler envelope。每个训练由
scheduler-neutral async task command 提交，Codex turn 随即退出；generic Harbor wrapper 在
无 LLM 调用时每 30 分钟读状态，完成后 resume 同一 Codex session。当前 LSF/Apptainer mapping
只存在于可替换的 Blue Vela adapter；单次 Codex turn 的 1h transport guard 不是训练 timeout。
其他 task 使用各自 `task.toml` 中声明的 agent timeout。verifier 随后仍在同一个
LSF allocation 中运行；GDN 按 release script 分配 4 个节点、每节点 8 GPU，其余 task 强制
`span[hosts=1]`。`agent trajectory steps` 是模型/工具交互步数，
不是科研优化次数。科研优化次数通常以 `/app/output/experiments.jsonl` 的有效 JSON 行数为准；
GDN fixed-time autoresearch 使用 Karpathy-style `/app/output/results.tsv`，一行对应一个测试 commit：
每个真正用于选择下一次修改的 train/evaluate 或 profile/A-B cycle 记一次，最终 verifier
不计入。

| Task | Agent/LSF allocation | 当前 verifier 实际并行 | 固定科研预算 | “完整”与“smoke”的界线 |
|---|---:|---:|---|---|
| Gated DeltaNet | 4×8 H100 = 32 GPU | 4×8 FSDP / 32 GPU | repository `tsz512x4k_15B` lane | 少于 15B token 或少于 4×8 ranks 只能叫 smoke；完整 run 为 7,152 optimizer updates |
| GDN fixed-time autoresearch | 4×8 H100 = 32 GPU | 4×8 FSDP / 32 GPU | 每个 candidate 固定 1,200 measured training seconds；现场 baseline + 2 probes + 2 synthesis 至少 5 个 valid experiments | 不足 1,200 秒、非 32 ranks 或没有 checkpoint 计量的 candidate 不算 valid；tokens/parameters 可因设计和吞吐变化 |
| DeepSpec | 4×H100 | 4 ranks / 4 GPU | 固定权重，6h 内做 profile/A-B | 没有 optimizer update；每个测量过的 scheduler candidate 算一次优化尝试 |
| DeepOCR | 8×H100 | upstream inference loop 为单进程、实际 1 GPU | report 首轮 lane 为每个 candidate 100 个 Stage-2 optimizer steps | 只评 easy_deepocr checkpoint 是 0 次优化，不是训练闭环 |
| RLM | 8×H100 | 两个相同 vLLM TP4 worker | 每次 decision-relevant eval 固定 50 rows×5=250 条；两个 bounded rolling pool 按 3+2 分配，每 worker 最多 50 条 active outer rollouts；每条 active rollout 最长 1800 秒 | agent 自评与 final verifier 使用相同 schedule/model-default sampling；queue wait 不占 rollout timeout；timeout 留在固定分母计 0；每个用于决策的完整 250 条候选算一次 optimization attempt |
| SuperBPE | 0 GPU / up to 64 CPU | 32 CPU verifier allocation | 每个 full candidate 最多读取 fixed 10GB corpus，最终 artifact 为 fixed 200K BPE | prefix/8K 是 smoke；10GB/200K full-training wall 尚未测量，不能按比例外推成六小时内 task |

因此表中的 32/8/4 是整个 full-run LSF allocation，不应自动推断每个 verifier 的并行度。
GDN training 和 scored verifier 都由共享 workspace 上的 4×8 ranks 执行；RLM 的 8 卡显式
分成两个独立 TP4 服务；DeepOCR verifier 当前只用一张卡，这是资源效率问题；
不改变模型分数，但 benchmark 时间只能解释为单 GPU inference。后续若把 evaluator 做成多 rank，
必须先用相同样本证明指标等价，并作为新的 verifier/SIF 版本重跑 baseline。

优化次数不预先伪造为固定数字，因为 agent 在可用 allocation 内能完成多少候选取决于实际训练
吞吐和各 task 的运行计划。driver 会把 trajectory step 数和相应 ledger 的实际行数都写入
`run_manifest.json`，两种计数不会混在一起。

GDN 作者仓库 README 对这条 4-node/15B lane 给出的公开估计是约 4 小时；这属于 repository
信息，不是论文输入。32-rank qualification job 234716 已测得 H100 稳态约 430 ms/iteration、
约 77.4K tokens/s/GPU；按 14,305 iterations 推算 training 约 1.8 小时，连启动、最终 checkpoint
export 和 4×8 verifier 保守按 2–2.5 小时。随后 pristine job 238329 实测 training
6,082.90 秒、完整 LSF 6,445 秒，`val_ppl@1x=15.084142519039398`、
`val_ppl@2x=14.536125056457275`、reward `0.269277`，成为 active matched baseline。
该 qualification 的远端 workers 未被本地停止信号
清理，继续到 iter 8060 后被一小时 runlimit 截断，因此只作为吞吐测量，不能当 baseline。
6 小时只是 SOP 的自动进入/review 阈值，
不是 task timeout，也不会在 6 小时时强杀训练。RLM 当前 pooled qualification job 304037
实测决策 eval 43 分 21 秒，含双服务冷启动、policy gate 和 trusted recomputation 的完整 verifier
56 分 50 秒；250/250 全部进入固定分母。早期 on-policy training task 的环境修复与 jobs
233873/234111/234338/234447/238576 只作历史审计，该 60-update task 已退休，不定义当前
orchestrator optimization 的科学预算。
GDN 没有 6 小时
强制限制；是否继续由可配置的 LSF allocation 和运行计划决定。
任何提前终止的 run 都必须按实际 tokens/updates 标为 incomplete/qualification，不能自动
缩预算后仍叫 full。改变 GPU 或具有科学含义的训练预算必须版本化并重跑 baseline；调整纯
运维 walltime 不改变科研协议。

SuperBPE immutable released-artifact reference job 324917 实测 verifier 1,115 秒、LSF CPU
time 1,113.5 秒、peak 13.4GB；虽然 allocation 是 32 CPU slots，official sequential
`encode_file` 路径的有效 CPU peak 只有 1.00 core。64MiB/8K custom-trainer pilot 则曾起到
175 threads。两者不能互相外推，10GB/200K candidate training 仍需 Step-6 review。

## 4. 每次完整 run 的目录

以后每次提交使用独立目录：

```text
/proj/datasets/interns/yuetai/agent_envs/scale_autoresearch_tasks/full_runs/
  <task>/<UTC timestamp>-<id>/
    run_manifest.json
    lsf.<jobid>.out
    lsf.<jobid>.err
    control/
      task/{task.toml,instruction.md,policy.yaml}
      task-tools/                 # GDN frozen launcher + checkpoint tool
      envs_backend/               # frozen Harbor provider patch
      harbor_patches/             # frozen async Codex wrapper
      {run_full_task.sh,run_contract.py,SETUP_FIDELITY.md,RUN_CONTRACT.md,SHA256SUMS}
    workspace/                    # GDN shared candidate source
    candidate-output/             # GDN attempts + selected submission
    harbor/
      <job>/<trial>/agent/trajectory.json
      <job>/<trial>/verifier/
      <job>/<trial>/artifacts/
      <job>/<trial>/result.json
    trajectories/
      trajectory-001.json -> ../harbor/.../agent/trajectory.json
```

`control/` 是提交时冻结的小型控制面；作业从该副本读 instruction/task/policy/driver，
不会在运行中因主仓库 shell 或描述文件被编辑而改变。`run_manifest.json` 记录
control 内每个文件的 SHA256，以及 task、model、期望/实际 GPU 数、LSF job id、镜像、
content-aware asset audit（含 HF revision、实际 file count/bytes）、
两个 SIF 的 size/mtime/SHA256、web-search 审计、trajectory SHA256/step 数、
实际 optimization-attempt 数和 Harbor exit
code。即使 run 失败，也保留同一个目录及已有轨迹和日志。

GDN task-facing runtime API 只有 `gdn_async_run.py` 与 `gdn_task_tool.py`；它只向 shared
output 写 request/status JSON，不出现 LSF 命令。宿主 dispatcher 独占 Blue Vela 的 4×8
`blaunch`/`torchrun` adapter，接口见冻结的 `ADAPTER_CONTRACT.md`；candidate 与 verifier
container 都不调用 LSF。迁移集群时只替换 host dispatcher/adapter（必要时再替换 Harbor
environment provider），不改 scientific task、训练 CLI、checkpoint schema 或 verifier。

## 5. 官方、report 与当前实现的差异

共同边界先说明：五个 agent workspace 都来自 `task.toml` 标出的 upstream commit；
我没有把自写训练框架伪装成官方 repo。但是 CUDA/Ubuntu/Python 依赖组合、Apptainer
definition、Harbor instruction、policy gate、trusted wrapper 和最终 reward 全部是本项目的
环境工程，不是上游官方发布物。DeepOCR workspace 还把
`PS3VisionTower` 的 module-level import 移到 `if config.ps3` 分支内，避免未使用 PS3 路径
时因可选依赖失败；SAM/CLIP branch 还允许显式 pinned `DEEP_OCR_VISION_TOWER` 覆盖公开
checkpoint 内嵌的作者 `/lustre` 路径，并把其他私有数据/vision 路径替换成 pinned read-only
`/datasets`/HF snapshot，并让原本 Slurm-only 的 setup 在 LSF 中显式使用声明的单节点 8 卡；
这四处是 lazy-import、文件解析与 scheduler 兼容 patch，不改变被评 DeepOCR 路径的模型
计算或 global batch 64；8 卡下 derived per-device batch 为 8，而 upstream 非 Slurm fallback
的 4 卡会得到 16，这是明确的 topology/microbatch 适配。RLM workspace 允许通过 `OOLONG_DATASET_PATH` 从同一 pinned
local Parquet snapshot 加载 validation split，这是 HF offline 兼容 patch；RLM tests image
还固定 `verifiers==0.1.15.dev9` wheel 和 `vllm==0.22.0`，也不是
官方 repo 自带的完整 lockfile。所有这些差异都必须随 run manifest 和镜像版本解释，
不能只写“official setup”。另外，部分 image definition 仍通过未完全锁死的 pip
requirements 安装依赖（policy judge 的 `litellm` 也是 build-time resolution）；所以
upstream commit 不能唯一确定运行环境，`run_manifest.json` 中两个 SIF 的 SHA256 才是
每次 run 的实际环境身份。

### Gated DeltaNet

- matched-token release-lane task 的 source：`NVlabs/GatedDeltaNet@b53d6d3...`。baseline 是仓库
  `lit_gpt/config.py` 的 `GatedDeltaNet_H1_0.4B` 与
  `scripts/tsz512x4k_15B_gated_deltanet_h1_0.4B.sh`。
- active checkout 只允许 3 个基础设施 patch，整体 diff SHA256 固定为 `63c451bf...`：
  `chunk.py`/`wy_fast.py` 删除 Triton 2.3.0 在 H100 backward 不支持的
  `num_warps=8/16/32` autotune 候选（保留相同 kernel 和 `1/2/4`），`pretrain.py` 修复
  resume/output-dir/multiprocessing 的 32-rank 冷启动竞态。它们不改 named config、数据、
  loss 或训练超参数，但可能影响 kernel 选择、浮点归约和速度，必须随结果披露。
- 训练 contract：repository packed SlimPajama `train_slim*`/`validation*`、4 nodes×8 ranks、
  micro batch 8、gradient accumulation 2、global batch 512×4096、LR 1e-4、seed 3407、
  15B nominal tokens、7,152 optimizer updates。LSF `blaunch`/`torchrun` 只替换 Slurm
  `srun`，共享 GPFS workspace 让 32 ranks 读取同一 candidate source；模型和训练参数不改。
- verifier：按 release code 启动 4×8 bf16-mixed FSDP，并用 rank-local validation partition
  （world size 32、2 chunks、batch 4、15 batches）计算 2048/4096 loss 与 PPL；主 reward 仍是本项目基础设施定义的
  `1 / (1 + ln(val_ppl@1x))`。
- 数据资产：NVlabs 脚本保留作者机器绝对路径，但其 README 指明 dataloader 来自 Samba；Samba
  README 公开链接预 tokenized PackedDataset。当前固定该资产的 HF revision 和 LFS SHA256；
  缺 manifest/文件时 full-run preflight 必须失败，不能换 FineWeb、PG19 或第三方 checkpoint。
- 旧 FLA/Flame/FineWeb/PG19 task 与 2026-07-16/17 smoke 已整体移到
  `gated_deltanet_old`，只作事故历史，不能作为 active baseline 或分数。

### Gated DeltaNet fixed-time autoresearch

- 独立 task `gated_deltanet_autoresearch` 不修改或替代上面的 15B task；task baseline 为
  `f13c929...`，含同一 H100 兼容 patch 和显式 fixed-time 训练协议。
- 每个 experiment 使用 32 H100、计满 1,200 秒训练；首 10 iterations 的 cold-compile warm-up
  与最终 15-batch eval/save 不入训练时钟。checkpoint 同时记录实际 seconds、tokens、updates、
  GPU count、parameters 和 raw 1x/2x loss。
- 第一条必须是现场 H1 baseline。至少两个受控 architecture mechanism probes 要先于至少两个
  synthesis candidates；至少 5 个 non-crash experiments。只有 unrounded geomean PPL 严格下降
  才能 keep，discard 后 source 精确回到 incumbent commit。
- `results.tsv` 是主 ledger，含 training seconds、end-to-end wall seconds、GPU-hours、tokens、
  parameters、peak memory 和 keep/discard/crash。final 只能选所有 valid rows 中最低 metric。
- 这条 lane 是 hardware-local fixed-time comparison；模型大小和 throughput 是优化变量，不能把
  结果称为 matched-token、matched-parameter、release-recipe 或 cross-platform comparison。

### DeepSpec / DSpark

- 上游：固定 `DeepSpec@005e03b...`、Qwen3-4B target 和
  `dspark_qwen3_4b_block7` draft；官方 `eval.py` 测 acceptance/verification 行为。
- report：固定 workload trace、SGLang/vLLM engine commit 和 SLO，先做 100 requests、
  batch 1→256 profile，主指标是 SLO-constrained goodput，另看 P50/P99；最终用 sealed
  trace。1--4 H100 是 smoke 起点，不是论文硬件要求。
- 当前：4 GPU 直接运行官方 decode loop，在 gsm8k/math500/aime25/humaneval/
  livecodebench 的公开样本上计时；trusted wrapper 审计所有 official cached
  target-verification call 的每个 committed token。每 task/rank 第一条另跑独立 greedy
  generation，共 20 条诊断但不作 gate。相同 target logits 的精确并列在独立 BF16 forward
  中可走不同后续路径，不应误判为 official verification 失败。没有 serving
  queue、batch/SLO grid、P50/P99 或 sealed trace。吞吐计时只包住 official
  `generate_one_sample`，两端显式 CUDA synchronize；in-path gate 在区间内只增加 committed
  probability 的异步 gather，CPU 判定和独立 greedy diagnostic 均在计时外。分母是四个 rank
  各自 decode 秒数之和的最大值，不包含模型加载、dataset 加载、TensorBoard 或额外 diagnostic。
  end-to-end wall time 另报。
  reward 是 `(tokens/s)/(tokens/s+25) × mean_verify_rate`，为本项目自定义。
- 结论：当前可用于检查 DSpark 机制、正确性和单一离线吞吐回归，但不能回答 report 中
  “同 tail-latency SLO 下 goodput 是否提高”的科研问题；当前 reward 不是 report-faithful。

### DeepOCR

- 上游：固定 `DeepOCR@cd49a8f...`，DeepOCR 是 SAM + 16× compressor + CLIP
  DeepEncoder、Qwen2-VL-7B decoder 的 VILA 复现；公开 Stage-1/Stage-2 脚本、
  easy_deepocr 和 sam_clip checkpoint。它不是 DeepSeek-OCR-2 官方训练实现。
- report：先复算公开 checkpoint，再做 8×H100、100-step Stage-2 smoke；最终设计包含
  hidden documents、OmniDoc-style edit distance/visual-token Pareto 以及 olmOCR-bench、
  layout/reading-order hard gates。
- 当前：使用上游 `llava/eval/omini_doc_bench.py` 和固定
  `OmniDocBench@2b161d0...`，另加 trusted media-token meter；只有公开 OmniDocBench，
  没有 hidden split 和 olmOCR-bench hard gate。reward 把四项质量等权平均，再除以
  `sqrt(max(1, visual_tokens/250))`；权重和 250-token anchor 都是本项目选择。
- 结论：OCR quality--visual-token Pareto 本身有科研价值，official inference/metric 也是真
  路径；但当前标量化、单公开集和无多 seed 使它只能算有意义的第一版 evaluator，
  不是 publication-grade 的 report 完整设计。

### RLM

- 唯一科学来源：`rlm@72d6940...` 的
  `training/configs/rlm-qwen3-30b-example.toml`、`training/src/rlm_train/` 和
  OOLONG environment。当前 task 固定作者发布的 Qwen3-30B-A3B-Instruct-2507 base 与
  `mit-oasys/rlm-qwen3-30b-a3b-v0.1` adapter，agent 优化 orchestrator/policy workspace；
  `max_iterations=20`、root/subcall 4096-token caps 和 inference-server default sampling
  temperature 保持不变。早期 60-update on-policy training task 已退休，只作历史审计。
- 依赖固定为 `prime-rl@84f072b...`（v0.5.0）：这是 RLM config 发布前 main 上最近的
  commit，实际 parser 可原样接受该 TOML。旧 task 的 v0.6 pin 将 `eval_base_model`/
  `filters` 改名且不保留 alias，原配置会在启动时失败，故已随旧设计归档。
- current workspace 的唯一上游源码 patch 是用 `OOLONG_DATASET_PATH` 读取同一 immutable
  validation Parquet，以满足 `HF_HUB_OFFLINE=1`；筛选、shuffle、split 和 scorer 不变。
  final verifier 使用完整固定 50-example selection，每 row 5 次独立随机 rollout，共 250 条。
- 8×H100 分成两个相同 TP4 vLLM worker，按 3+2 repeats 分配。每个 worker 使用 bounded、
  work-conserving rolling pool，最多 50 条 active outer rollouts；首批 50 条覆盖每个 row 一次，
  之后随完成补位而不设 wave barrier。queue wait 不计入 1800 秒 per-rollout timeout。
- agent 决策自评从只读 `/task-tools/rlm_eval_pool.py` 启动同一 schedule；final verifier 独立
  启动相同服务并用 masked gold 复算所有 reward。可确定的退化枚举输出计零，submitted
  reward、iterations、REPL calls、subcalls、errors/timeouts 只作诊断。
- primary reward 是 trusted recomputed raw OOLONG rewards 对全部 250 条的算术平均，再做
  `[0,1]` identity clip；timeout 留在固定分母计零。任何 decision-relevant 候选都必须完成
  整个 250-rollout eval，不能用小样本 smoke 作选择。
- matched pooled qualification job 304037 完成 250/250：reward `0.25718349663568113`，五个
  repeat 为 `[0.220018, 0.220000, 0.305900, 0.260000, 0.280000]`，11 timeout、0 error、
  0 reward mismatch；eval 43 分 21 秒，完整 hardened verifier 56 分 50 秒。job
  233113/238576 及 temperature-0 cost-aware 结果均是旧协议历史，不是当前 baseline。

### SuperBPE

- 唯一科学来源是 `PythonNut/superbpe@bbd0976...` 与其 pinned
  `tokenizers-superbpe@757f2a5...` submodule。baseline 是仓库发布的
  `olmo2_p99_truncate_10G_80K_extend_200K_mw4_colon/tokenizer.json`，不是从论文补出的配置。
- task 固定 repository 的 exact 10GB corpus、200K BPE model vocabulary、五个 released
  added tokens、deterministic lossless encoding 和 evaluator；agent 可广泛改变 merge objective、
  boundary schedule、pretokenization、merge allocation、corpus order/weighting 和实现。
- 主指标是 pinned held-out file 的 bytes/token。candidate 与 released artifact 在同一进程、
  相同 paragraph chunks 上 matched 计数；reward `C/(C+B)` 只是单调有界映射，baseline 为 0.5。
- repository notebook 的 6.634021 t80 输出来自 private 1GB eval path；current public eval 为
  944,259,457 bytes，job 324917 完整重算得到 141,911,558 tokens、
  6.653858715299285 bytes/token、reward 0.5。仓库同时缺 exact 200K/t80 training
  command 与 final serialized-pretokenizer transformation，所以当前能严谨复现的是 released
  artifact 的 matched evaluation 和 custom trainer path，而不是 author training command。
- 64MiB/8K two-stage job 324882 完成实际 trainer path（104s，5,782MB）；job 324889 修正了
  wrapper 对 inherited merges 必须构成 exact prefix 的错误假设。完整 10GB/200K training 尚未
  qualification，Step 6 前必须由用户 review runtime envelope。

## 6. 已执行作业的真实含义

动态 Harbor full-run、agent optimization attempts 与 trajectory 状态只以各 run 的不可变
`run_manifest.json`、ATIF trajectory 和 `experiments.jsonl` 为准。下表记录已经完成的
镜像/CUDA qualification 与 author-side baseline/smoke：

| Task | 已跑 GPU 验证 | 实际输入 | 结果用途 |
|---|---|---|---|
| GDN（active matched baseline） | job 238329，4×8 H100，DONE | pinned NVlabs H1、sealed SlimPajama、14,999,879,680 tokens、release validation | 14,305 iterations / 7,152 updates；training 6,082.90 s、端到端 6,445 s；PPL@1x 15.0841425、PPL@2x 14.5361251、reward 0.269277；0 agent optimization |
| GDN（已归档旧 task） | job 221041，1×H100，DONE | pinned Flame 随机初始化；offline PG19 `test[:1]` | 仅是 `gated_deltanet_old` 启动历史，旧 PPL/reward 禁止复用 |
| DeepSpec | job 221609，4×H100，DONE；更早严格 job 221081/221229 均 EXIT | 五个公开 tasks、temperature 0、430 samples；失败分别暴露 replay 梯度和独立 BF16 replay 非 bit-exact 问题 | 当前 pristine qualification：199,512 tokens、780.371 token/s、verify rate 0.729801、199,679 in-path token checks/0 invalid、reward 0.707147；单次 current-env baseline，不是 paper/repeated result |
| DeepOCR | job 221042，1×H100，DONE | easy_deepocr，确定性 1 页 OmniDoc slice | reward 0.503778、904 visual tokens；0 optimization，不能外推完整 benchmark |
| RLM（active pooled qualification） | job 304037，8×H100、两个 TP4 worker，DONE | pinned 50-row OOLONG selection×5；model-default sampling；3+2 bounded rolling pools；masked-gold recomputation | 250/250、raw mean 0.25718349663568113、11 timeout、0 error、0 mismatch；eval 43:21、完整 verifier 56:50；当前 matched baseline |
| RLM（历史 zero-update） | job 233113，8×H100 TP8，DONE；25-row loop 55:03，端到端 3,747 秒 | pinned local OOLONG 25 条 trec_coarse@131072；model-default sampling；raw scorer | raw 0.36000535799246636、25 valid / 0 error、785.16 subcalls、11.04 REPL calls、10.68 root iterations；旧协议 reference |
| RLM（upstream-path qualification） | job 234447，8×H100，stable step 20/40 后在 step-40 长尾 eval 中终止 | upstream 200-step config、4 train + 4 inference、base/20-step eval | 证明完整训练/推理/checkpoint/eval 路径；普通 step 实测均值 128.10 秒；不是 complete 200-step reproduction，也不是 active 60-step baseline |
| RLM（历史 60-update matched baseline） | job 238576，8×H100；RL process DONE，post-run wrapper audit false-positive EXIT 1 | 旧 task protocol：4 train + 4 inference、60 updates、1,920 train rollouts、final-only 25 rows、600 s/row cutoff | 60/60 updates，stable 20/40/60，final 25/25 in 602.73 s，raw 0.360077631820896、0 error、2 timeout rows 计零，总 8,248 s；已退休协议 |
| RLM（已归档旧 task） | job 220894，8×H100 TP8；evaluator 完成，wrapper 最终 EXIT 127 | pinned local OOLONG Parquet 的 25 条 trec_coarse；HF offline；强制 temperature 0 | `rlm_old` 历史：raw 0.200010、855.56 subcalls、9.96 root iterations、cost reward 0.004569；不得替代当前 pooled baseline |
| SuperBPE（two-stage smoke） | jobs 324882/324889，0 GPU、CPU，DONE | 64MiB fixed training prefix，8K stage1/stage2；第二个 job 只重跑修正后的 validation | trainer 两阶段均成功；stage1 7.92s、stage2 86.16s、LSF wall 104s、max memory 5,782MB；不是 10GB/200K full candidate |
| SuperBPE（immutable reference） | job 324917，0 GPU、CPU，DONE | released t80 artifact、verifier-only pinned full public `eval/135.txt`、frozen control/project hashes | 21 chunks、141,911,558 tokens、6.653858715299285 bytes/token、ratio 1.0、reward 0.5；verifier 1,115s、peak 13.4GB；0 agent optimization |

另有 job 221610 曾在一张 H100 上把旧版四套 GPU environment/tests SIF 的源码 `cmp`、CUDA 和关键 package import
全部跑通（DONE）。那一版 GDN SIF 后来因 active task 替换而 stale；当前 active SIF 已重建并由
job 234714 通过最终 H100 backward/optimizer 与 verifier smoke。历史 TinyStories/366M GDN 与联网 RLM
结果仍保留为作废证据，不与上表当前结果混用。“五个 agent 已经完成优化迭代并得到科研结论”
仍然是假的；后者必须由新的 full-run 目录、trajectory、`experiments.jsonl`、训练 provenance
和完整 evaluator 共同证明。
