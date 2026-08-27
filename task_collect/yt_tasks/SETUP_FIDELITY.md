# 五环境 setup 忠实度与改动总账

更新日期：2026-07-25。这个文件是判断“pinned repository 一致”或“环境自定义”的入口。
任何 run 都应把本文版本与两个 SIF 的 SHA256 写进 `run_manifest.json`。task 的科学设计、输入
和 baseline 以作者公开仓库为优先且唯一来源；论文比较只作为旧审计记录，不再用于补配置。

## 1. Repository 与环境适配不要混称

| 层级 | 含义 |
|---|---|
| Pinned repository | task 固定 commit 中实际发布的代码、config、script 和 evaluator；task 设计的科学 source of truth |
| Missing public artifact | repository 指向但没有发布的 tokenizer/data/checkpoint；必须写 missing，不能从论文补 |
| Current env | Harbor/LSF/Apptainer、离线路径、topology mapping、verifier、policy 和 reward |
| Historical context | 旧 report/paper 对照，仅用于解释历史，不能作为新 task 输入或 baseline |

对外表述必须指出 repository 原样内容和环境适配的边界。

## 2. 上游源码逐文件状态

以下状态按 `task.toml` 固定 commit 与容器 workspace 比较；生成文件、`.git` 和缓存不计。

| Task | 当前源码相对 pinned upstream | 结论 |
|---|---|---|
| GDN / NVlabs | 3 个基础设施兼容 patch | `GatedDeltaNet_H1_0.4B` 与 `tsz512x4k_15B` 是唯一 active baseline；H100 上收窄两个 Triton kernel 的 autotune 候选，另修复 32-rank 启动竞态；旧 FLA/Flame task 已归档为 `gated_deltanet_old` |
| GDN fixed-time autoresearch | 独立 task baseline commit `f13c929...`：复用上述 2 个 kernel patch，并把 `pretrain.py` 改为 1,200 秒计时训练、计时进度 LR、最终固定验证及 checkpoint 计量 | 这是有意设计的 hardware-local autoresearch harness，不冒充 NVlabs release recipe，也不替代 matched-token `gated_deltanet` task |
| DeepSpec | 无差异 | 官方 decode/eval source；环境 verifier 另置于 `/tests` |
| DeepOCR | 4 个源码文件含基础设施适配 | lazy `PS3VisionTower` import；SAM/CLIP branch、Stage-2 registry 与 train script 把作者私有 `/lustre` 数据/vision 路径解析到 pinned read-only `/datasets`/HF snapshot；Slurm-only launcher 在 LSF 下显式使用 1 node×8 GPU |
| RLM | 1 个基础设施 patch | OOLONG loader 可由 `OOLONG_DATASET_PATH` 读取相同 pinned local Parquet；其余官方 RLM source 不变，另固定 prime-rl commit |
| SuperBPE | 无差异 | clean `PythonNut/superbpe@bbd0976...` 与 pinned `tokenizers-superbpe@757f2a5...`；所有 task tool、verifier 和数据路径适配均在 upstream workspace 外 |

DeepOCR lazy-import patch 的原因是 upstream 在未选择 PS3 的 SAM/CLIP DeepOCR 路径上也
强制 import 未声明的可选 `ps3` package。另两处只把作者机器的绝对数据/vision 路径改为
本环境的固定只读路径；所指文件 revision 进入 provenance。第四处保留 Slurm 分支，同时在
没有 `scontrol` 的 LSF/Apptainer 中尊重显式 `NNODES=1, GPUS_PER_NODE=8`，否则 upstream
默认会悄悄只起 4 卡。这四处不改变所选模型计算、global batch 64 或其他官方脚本超参数；
8 卡使 derived per-device batch 从 fallback 的 16 变成 8，属于已披露的 topology/microbatch
适配，不是论文硬件要求；
如果实验涉及 PS3，必须另做 upstream-exact 环境，不能沿用这个兼容性结论。

RLM patch 是因为当前 `datasets` 在 `HF_HUB_OFFLINE=1` 下不能仅凭 repo ID 从一个只读 partial
Hub snapshot 重建 builder；loader 在设置 `OOLONG_DATASET_PATH` 时显式读取该 revision 的 9 个
validation Parquet，否则仍保留 upstream Hub ID 行为。离线 SIF 已实际读出 sample，数据过滤、
shuffle、split 和 scorer 没有改动。

GDN 的两个 kernel patch 只删除 Triton 2.3.0 在 Hopper 上不受支持的 `num_warps=8/16/32`
autotune 候选，保留作者实现与 `1/2/4` 候选；不改算子公式、模型 config、loss 或训练 recipe，
但可能改变 kernel 选择、浮点归约次序和速度，因此必须作为 H100 基础设施差异披露，不能称作
bit-exact clean clone。`pretrain.py` patch 让所有 ranks 仅在真实 checkpoint 存在时 resume、由
rank 0 建目录后 barrier，并强制设置 multiprocessing start method；它不改变成功训练路径的
优化器、数据顺序或超参数。

release-fidelity task workspace 的完整 patch inventory 只有下面 8 个源码文件；其余差异是
workspace 外的 task/verifier/container 工程。独立的 GDN fixed-time autoresearch harness
另有一项明确列出的研究协议改动，不计入 release-fidelity inventory：

| Task / file | 相对 pinned upstream 的准确语义差异 |
|---|---|
| GDN `lit_gpt/gated_delta_rule_ops/chunk.py` | 删除 Hopper/Triton 2.3.0 会在 backward 触发 layout assertion 的 `num_warps=8/16/32` autotune configs；保留相同 kernels 与 `1/2/4` configs |
| GDN `lit_gpt/gated_delta_rule_ops/wy_fast.py` | 同上，只收窄该文件七处 autotune 候选，不改 kernel body |
| GDN `pretrain.py` | resume 判据从“输出目录存在”改为“`latest-model-ckpt.pth` 存在”；rank 0 幂等建目录后 barrier；multiprocessing start method 使用 `force=True`，消除 32-rank 冷启动竞态 |
| GDN autoresearch `pretrain.py` | 在上述兼容修复上增加固定 1,200 秒训练计时（首 10 iterations 作为冷编译 warm-up）、按计时进度调度 LR、最终 15-batch 1x/2x 验证，并把 seconds/tokens/updates/GPU/parameters/metric 写入 checkpoint；这是实验协议而非基础设施等价 patch |
| DeepOCR `llava/model/multimodal_encoder/builder.py` | 把 `PS3VisionTower` 从模块顶层无条件 import 移到实际选择 PS3 的分支；SAM/CLIP 分支在显式设置时用 pinned `DEEP_OCR_VISION_TOWER` 覆盖 checkpoint 内嵌的作者 `/lustre` 路径 |
| DeepOCR `llava/data/registry/datasets/default.yaml` | `olmOCR-mix-pretrain.ann_file` 从作者 `/lustre/...` 改到 `/datasets/deepocr/olmOCR-mix-1025/transformed_data_png.json`，data root 同步改为该只读 mount |
| DeepOCR `scripts/NVILA-Lite/pretrain_ocr.sh` | vision tower 的作者 `/lustre/...` 路径改为 pinned `sam_clip_ckpt` snapshot，并允许记录在 provenance 的 `DEEP_OCR_VISION_TOWER` override |
| DeepOCR `scripts/setups/train.sh` | Slurm 存在时保留原逻辑；无 Slurm 时从显式 `NNODES/NODE_RANK/GPUS_PER_NODE/MASTER_*` 读取 LSF topology，不再静默 fallback 到 4 GPU |
| RLM `training/environments/oolong/oolong/env.py` | 仅在设置 `OOLONG_DATASET_PATH` 时，把同一 9 个 validation Parquet 作为 local `parquet` dataset 载入；未设置时完全保留 upstream Hub-ID loader |

matched-token `gated_deltanet` workspace 是 pinned `NVlabs/GatedDeltaNet` checkout 加上述 3 个逐文件锁定的
H100/分布式启动 patch；整体 diff SHA256 为
`63c451bf94a954445c89850ae4fd7efd3ae8188483f4974050e4088685e5eba4`。旧 Flame 配置和旧
PG19 verifier 只存在于 `gated_deltanet_old` 与历史运行记录中，不得复用为新 baseline。

## 2.1 已纠正的事故账（旧结果不得混入当前 baseline）

这些不是“研究变量”，而是我在搭环境时引入或漏报、现已修正的错误：

1. **GDN 配置冒充官方**：曾把第三方同名 HF checkpoint 的 `expand_v`、heads/layers、
   embedding tying 和 vocab 形状写进 Flame，得到 366,763,200 参数并错误称为官方 340M；
   该旧任务后来恢复到 pinned Flame 的 512,272,124 参数，但现在已整体归档。active GDN
   从未把这两个配置作为 baseline，旧 TinyStories/PG19 分数全部作废。
2. **asset ready 假阳性**：旧 preflight 只检查目录存在，曾把仅有 `refs/main`、README 或
   partial snapshot 的 Qwen/easy_deepocr/OOLONG 报成 ready；当前检查真实权重/shard、revision、
   bytes/hash 和转换 manifest。
3. **RLM 在线数据污染**：历史 25-example smoke 的日志有 Hugging Face Hub request；当前
   loader 使用同一 pinned 9 个 local Parquet 并在 `HF_HUB_OFFLINE=1` 下实读，旧分数只算
   计算路径 smoke。
4. **GDN data/source 混合**：旧任务把 Flame config、FineWeb recipe 和 PG19 verifier 拼在一起；
   当前已整体归档，active task 只使用 NVlabs release 的 packed SlimPajama train/validation contract。
5. **DeepOCR 空转换假成功**：第一次 multiprocessing import 因 `PicklingError` 产生空结果，
   旧检查的 `0 == 0` 又误放行；当前强制 267,962 raw / 5,752 missing-text /
   262,210 post-filter，并做唯一路径、精确集合、PNG header/dimension 和 hash 审计。
6. **DeepOCR 私有路径与 GPU fallback**：公开 checkpoint 内嵌作者 `/lustre` vision path，
   upstream 非 Slurm branch 又会默认 4 GPU；当前分别用已校验 snapshot override 和显式
   `1×8` LSF topology 解析，二者均列入 patch inventory。
7. **DeepSpec 指标/gate 过弱**：旧吞吐把模型/数据加载、TensorBoard 和 correctness diagnostic
   算入 decode；旧 exact-greedy gate 又会把 target-logit tie 误判失败，且完成条件只要求
   至少 5 条而不是 4 ranks×5 tasks。当前只计 official decode call，并严格要求 20 个唯一
   `(rank, task)` 独立诊断。
8. **DeepSpec replay 不是 cached verifier 的 bit-exact oracle**：第一次严格 lane 的 replay
   未禁梯度导致显存失败；禁梯度后又实测到独立 full-sequence BF16 forward 与 cached target
   verification 在 exact logit tie/归约顺序上可分叉。因此当前 correctness gate 不再用 replay：
   它直接审计所有 official cached target-verification call 的每个 committed token；独立 greedy
   只作诊断。两次失败作业和输出均保留。
9. **DeepSpec correctness audit 污染 timing（在跑分前截住）**：中间版曾在每次 verification
   内对全词表做 `max`/`.item()`，会按 verification 次数强制 GPU 同步并偏置 scheduler
   comparison；对应未完成的 build/lane 已取消，没有分数。当前 timed interval 两端显式
   CUDA synchronize，区间内只异步 gather committed-token probability，CPU gate 与独立
   diagnostic 都在计时后。
10. **DeepOCR 把过滤后数量当渲染成功数量**：完整 144-DPI run 确实处理完 262,210 个
   post-filter rows，但 upstream converter 对 219 个异常超大 MediaBox 报
   `PyMuPDF Overly large image` 并按自身实现跳过，真实输出为 261,991。旧 hardcode 把这个
   上游行为错误判成全局转换失败；当前 manifest 固定 source/filter/failure/final 四层计数、
   失败原因和 skipped-path SHA256，并逐个重放 219 次失败。
11. **RLM 长作业读到运行中被编辑的 launcher**：离线 25-example evaluator 已写完
   metrics/reward 并通过独立结果检查，但宿主 launcher 在它阻塞运行时被编辑，bash
   恢复后读到移位的 `--env` 行，因此 LSF wrapper 最终为 exit 127。这不是模型/
   evaluator 失败，但是运行不可变性失败。正式 full-run launcher 现在提交时冻结
   instruction/task/policy/driver/auditor 到该 run 的 `control/` 目录，从副本执行并记录 hash。
12. **RLM parser-only smoke 漏掉运行时子进程 import**：首个 Candidate B pristine 作业 233873
   在 61 秒内因缺 `flash_attn` 退出；仅补该 extra 后，job 234111 又在 91 秒内因 orchestrator
   缺 `orjson` 退出。两者都发生在模型加载和任何 train/eval rollout 之前，不是 scientific
   run/optimization attempt。根因是环境没有执行 pinned prime-rl README 的完整官方安装命令。
   环境定义现改为 `uv sync --frozen --all-extras`，所有版本/hash 仍来自同一 `uv.lock`；smoke
   直接 import inference server、orchestrator 和 trainer 三条重路径，不再只验证 TOML parser。
13. **RLM 把官方自动转换缓存写进只读输入 snapshot**：all-extras 环境通过后，job 234338
   的 trainer 在任何 rollout 前尝试创建 HF snapshot 下的 `prime/` DCP cache，因共享输入被
   正确只读挂载而退出；我随后终止仍在等待的其他子进程，作业为 EXIT 130/364 秒，仍计
   0 scientific run。launcher 现为每个 run 建立可写 HF namespace，大权重用 hard link
   引用，但 16 个原始 safetensors、共享 blobs 和 OOLONG repo 在容器内再次嵌套只读挂载；
   只有新生成的 `prime/` 留在 run 私有目录。mount smoke 已验证所有共享路径为 `ST_RDONLY`、
   私有 `prime/` 可写且共享 snapshot 未改变；job 234447 已实际完成 13-shard 转换并进入 base eval。
14. **GDN 把不兼容的 FLA snapshot 当成 release pin**：NVlabs Dockerfile 对 FLA 只写未固定
   commit 的 `pip install -U git+https://github.com/sustcsonglin/flash-linear-attention`；先前环境错误
   选了 2024-12-10 的 `affeddd...`。H100 forward 实测该版本的 `ShortConvolution` 返回 tuple，
   而 NVlabs `gated_delta_net.py` 要求 Tensor。当前根据两个公开仓库的代码历史固定在接口变化
   commit `46706a...` 的直接父提交 `326184c...`（2024-11-21）：这是最后一个保持 NVlabs 调用
   contract 的公开 FLA revision。该修正发生在任何训练前，相关作业只有环境 smoke、0 次优化。
15. **GDN 的 forward smoke 漏掉 H100 backward 与 32-rank 冷启动**：首次正式 4×8 作业
   234028 在 iteration 1 前退出，完成 0 optimizer updates；一类错误是 Triton 2.3.0 的
   Hopper MMA layout assertion，另一类是共享 Transformers cache、multiprocessing start
   method 和仅凭输出目录存在判断 resume 的 rank 间竞态。当前 qualification 增加真实
   530,765,016 参数模型的 forward/loss/backward/fused-AdamW step；H100 smoke 234566 已通过。
   上述 3 个源码 patch 和每节点私有 Transformers cache 均已进入构建/run 门禁，旧失败不计
   scientific run 或 optimization attempt。
16. **32-rank qualification 的停止信号没有传到远端 `blaunch` workers**：job 234716 在
   启动 120 秒时已经达到目标 `iter 10 / step 5`，但本地 launcher process group 退出后，
   四节点 torchrun 仍继续到一小时 LSF runlimit，最大日志为 iter 8060 / step 4030 /
   8,451,522,560 tokens，最后完整 checkpoint 为 iter 8000 / step 4000。该 run 从提交前就明确
   标为 infrastructure qualification，且被截断、无 final checkpoint，不能事后升级成 baseline；
   runtime `qualification.json` 已按实际最大日志和 checkpoint 更正。它测得稳态约 430 ms/iter、
   约 77.4K tokens/s/GPU，据此完整 14,305-iteration H100 training 约 1.8 小时，连 export/verifier
   保守估计 2–2.5 小时。正式 reference 必须从全新目录和随机初始化重跑。
17. **已归档 RLM training task 的 outer-rollout wall-clock guard**：job 234447 的 step-0 eval 被单条
   `APITimeoutError/ReadTimeout` 拖到 14,180.97 秒，step-20 eval 又被单条 1,840-subcall
   rollout 拖到 3,661.05 秒；step-40 eval 的前 23/25 条约五分钟完成，最后两条超过十分钟后
   随 qualification 一起停止。当时的 training task 因此版本化为 60 training updates、final-only
   25-row eval，并通过 prime-rl config 给每条 outer eval rollout 设置 600 秒 timeout。
   `MultiTurnEnv` 用 `asyncio.wait_for` 取消 active request、标记 `timeout_reached`；该 row 不从
   固定分母删除且由原 OOLONG scorer 计 0。这是披露的 task protocol，不是上游 200-update
   reproduction 的原生设置。
18. **已归档 RLM training task 的 source manifest 把运行时 Python cache 当成源码改动**：matched task job 238576
   已由 RL process 正常完成 60/60 updates、step-60 checkpoint 和 25/25 final eval，但收尾
   wrapper 将运行中新生成的 `__pycache__/*.pyc` 纳入 after-manifest，因而在科学流程完成后
   误报 LSF EXIT 1。排除这些 cache 后 before/after manifest byte-identical，SHA-256 均为
   `57c50975a4df801cd129925ee929b0624b363752a9ba9448035b56d746d377d0`；共享资产和 task config
   也分别通过复核。两个 RLM qualification launcher 现都 prune `__pycache__` 并忽略 `*.pyc`，
   真实源码文件仍全部受完整性 gate 保护。
19. **GDN author-side launcher 通过不等于 Harbor 内能嵌套启动**：reference job 238329 是由
   LSF host 直接执行 launcher；Harbor agent 主容器则使用 `--containall`，看不到宿主
   `/proj/...` 原路径，且未指定长命令 transport timeout 时底层 HTTP 会在 600 秒断开。
   这在首次 agent run 前由静态/runtime preflight 发现，未产生 optimization attempt。
   full-run driver 现把 Apptainer 目录、environment/tests SIF 和冻结 task tools 显式只读挂载到
   原宿主路径，传入 LSF host/job 环境，并区分 host shared output 与容器内 `/app/output`；
   GDN 使用 scheduler-neutral async command；当前 Blue Vela launch adapter 单独实现 LSF
   mapping。Codex 提交训练后结束该 turn，Harbor wrapper 每 30 分钟在无 LLM 调用状态下读
   status，完成后 resume 同一 session。默认 scheduler envelope 24h、单次 Codex turn
   transport 1h，均可运维配置，不是科学训练预算或 6h 限制。

以上修正只让环境可审计，不会把任何历史 smoke 自动升级为 matched baseline。full-run 的提交与
trajectory 状态以每个不可变 `run_manifest.json` 为准，不在本文中硬编码动态计数。

## 3. 我加入或改动的环境工程

这些都不是论文或 upstream 自带内容，必须随结果披露：

- 每个 task 在 `environment/project/` 之外的文件全部是本项目自写，而不是 upstream：
  `task.toml`、`instruction.md`、`policy.yaml`、`environment/Dockerfile`、GDN 的
  `environment/launch_gdn_4x8_lsf.sh`，以及
  `tests/{Dockerfile,evaluate.py,policy.yaml,policy_check.py,score.py,test.sh}`；共享的
  `_policy_check.py`、本文档和 `lsf_validation/` 下全部 launcher/preflight 也都是自写；

- 五份 Harbor `task.toml`、AutoLab 风格 instruction、policy 和 Reward-Integrity Gate；
- LSF launcher、Apptainer definitions、CUDA/Ubuntu/Python 组合、依赖 pin、Harbor 目录采集；
- 关闭 Codex web search、HF offline、provenance/trajectory/experiment 记账；这不是物理 egress
  隔离，因为 Codex API 与 policy judge 仍需要控制面网络；
- 每个 full run 的 GPU：GDN 32（4 nodes×8）、DeepSpec 4、DeepOCR 8、RLM 8、SuperBPE 0（CPU-only）；GDN 不设 6h task-level agent timeout，默认 submit envelope 为 24h；训练期间 Codex offline，由 generic async Harbor wrapper 每 30 分钟监控；
- 上述是 LSF/agent allocation；当前 verifier 有效并行度分别为 GDN 32、DeepSpec 4、
  DeepOCR 1、RLM 两个独立 TP4 worker、SuperBPE 32 CPU threads 的 evaluator allocation。GDN training 与 scored validation 都保留 release 4×8 topology；
  DeepOCR 单进程 evaluator 不得被描述成使用完整 training allocation 的 benchmark；
- 每次完整 run 独立目录和 `run_manifest.json`；trajectory 与 agent optimization attempt 的动态
  计数只从该 run 的 manifest、ATIF trajectory 和 `experiments.jsonl` 读取；
- 所有最终 scalar reward 都是本项目定义，不来自论文；所有公开 evaluator 都需要 matched
  pristine baseline、多次重复/多 seed 和原始指标，才有科学比较价值。

当前容器栈也属于环境适配，而非论文 setup：

| Task | CUDA base | 关键显式 pin / 未锁项 |
|---|---|---|
| GDN | upstream Docker base PyTorch 2.3.1 / CUDA 12.1 / cuDNN8 | NVlabs commit、code-compatible FLA `326184c...`、Lightning 2.1.2、datasets 2.20.0、Triton 2.3.0、xformers 0.0.27、Transformers 4.47.0；NVlabs 没有锁 FLA，当前 pin 是公开 FLA 将 `ShortConvolution` 从 Tensor 改为 tuple 的 commit `46706a...` 的直接父提交，并已由 H100 forward 验证调用 contract。Triton 保持作者镜像的 2.3.0，但 H100 下从两个上游 kernel 的 autotune lists 删除 `num_warps=8/16/32`，保留 `1/2/4`；这是公开记录的 Hopper layout assertion 的最小兼容适配。Transformers 4.47.0 避免新版内置 BitNet config 与旧 FLA 同名注册冲突。上游未锁的 CUDA 扩展固定为 causal-conv1d 1.5.0.post8、mamba-ssm 2.2.4、flash-attn 2.7.2.post1；作者 Docker 另行安装的 `rotary_emb`、`dropout_layer_norm`、`xentropy_cuda_lib` 固定到 Dao flash-attention commit `f86e3dd...`，并用 `--no-build-isolation` 让 build 看到基础镜像 PyTorch |
| DeepSpec | CUDA 12.8.1 / Ubuntu 24.04 | 使用 pinned repo 的 `requirements.txt`，但其中依赖是否精确锁定由 upstream 决定 |
| DeepOCR | CUDA 12.1.1 cuDNN8 / Ubuntu 22.04 | flash-attn 2.5.8、protobuf 3.20.x；其余 extras 有未锁项 |
| RLM | CUDA 12.8.1 / Ubuntu 24.04 | uv 0.11.1、prime-rl v0.5.0 `84f072b...`，按其 README 从同一 lockfile 安装 `--all-extras`；tests 固定 verifiers 0.1.15.dev9 与 vLLM 0.22.0 |
| SuperBPE | Debian bookworm slim / CPU-only | Python 3.12.9、Rust 1.84.1、pinned custom tokenizer fork 0.20.1；tests 用相同版本的标准 `tokenizers` 做 encoding；SIF SHA256 是完整环境身份 |

所以 upstream commit 不能单独确定一个 run。matched baseline/candidate 必须复用同一对 SIF；
`run_manifest.json` 记录的 SIF SHA256 才是实际软件环境身份。重新 build 后旧分数默认失效，
除非用 matched rerun 证明等价。

## 4. 逐环境对照

### Gated DeltaNet

| 项目 | Pinned NVlabs repository | Current environment |
|---|---|---|
| 源码 | `NVlabs/GatedDeltaNet@b53d6d3...` | pinned checkout + 3 个锁定的 H100/32-rank 启动 patch；diff SHA256 `63c451bf...` |
| 模型 | `GatedDeltaNet_H1_0.4B` | 相同 named config；verifier 从 pinned checkout 计算 530,765,016 whole-model parameter count 并要求 candidate 精确相等，不再自设 size band |
| 训练 | `tsz512x4k_15B`、packed SlimPajama、LR 1e-4、seed 3407、1% warmup | 相同 recipe 与官方代码路径 |
| global batch/topology | 4 nodes×8 GPU，micro batch 8，gradient accumulation 2，总计 512×4096 | 相同 4×8 ranks、accumulation 2 与 PackedDataset rank partition；共享 GPFS workspace 供四节点读取 |
| GPU SKU | release script 只要求每节点 8 GPU，不写具体型号 | 当前集群分配 H100；这是环境资源选择，不冒充 repository 参数 |
| launcher | Slurm `srun` + release container | LSF `blaunch` + `torchrun` + Apptainer；保持 world topology 与 CLI 参数；`pretrain.py` 仅含上述 resume/multiprocessing 竞态修复 |
| FLA dependency | Dockerfile 从 Git main 安装且未固定 commit；当前 main 与 release 调用不兼容 | 固定 `326184c...`，即公开 FLA `ShortConvolution` tuple API 变化前的最后一提交；这是可复现的 code-compatibility resolution，不冒充作者发布过 lockfile |
| validation | repository `validate()` 的 4×8 FSDP、rank-local packed `validation*`、2048/4096 loss 与 PPL | scored verifier 同样启动 4×8 ranks，复用 world-size 32 的 filename shuffle/partition、2 chunks、batch 4 与 bf16-mixed FSDP；最终主指标 `val_ppl@1x`，同时报 `@2x` |
| 当前 reward | repository 不提供 scalar reward | `1/(1+ln(val_ppl@1x))`，仅为 Harbor 基础设施映射 |
| matched baseline | repository 给曲线图但不发布本环境数值 | job 238329：14,999,879,680 tokens；`val_ppl@1x=15.0841425`、`@2x=14.5361251`、reward `0.269277`；training 6,082.90 s，端到端 6,445 s |

NVlabs shell script 的 `/data/SlimPajama-627B_tokenized/{train,validation}` 是作者机器挂载路径，
但 NVlabs README 同时明确说 release dataloader 来自 Microsoft Samba；Samba 的公开 Data
Preparation README 链接了预 tokenized 的 `jsun/slimpajama_Llama2_Tokenizer`。当前 Step 4 固定
该公开资产 revision `cdc67c74692d46ec880e3a14f2bc2012d9aed6bd`：20 个 LFS transport parts
共 851,836,788,801 bytes，全部直接 staging 到 GPFS，解压后的 `train_slim*`/`validation*`
仍由原 loader 按 prefix 选择。审计固定 Samba commit `617c7a0...`，并确认它与 pinned GDN 的
`lit_gpt/packed_dataset.py` byte-identical（SHA256 `34beb28c...`）。staging 对 transport parts
和所有 LITPKDS 文件做一次完整 SHA256，
full-run preflight 验证 sealed index、精确文件集合与确定性抽样哈希；manifest 未完成时仍失败，
不能改用 FineWeb、PG19 或第三方 checkpoint。旧 FLA/Flame 512M task、PG19 evaluator 和所有
相关 smoke 已整体移到 `gated_deltanet_old`，只保留事故审计意义。

### Gated DeltaNet fixed-time autoresearch

这个独立 task 只借用同一 GDN dependency SIF、4×8 H100 mapping、packed SlimPajama 和冻结
1x/2x evaluator；其 candidate baseline 是 task-local clean commit `f13c929...`，运行时把该
快照分别绑定为 agent workspace 与 verifier `/opt/project`，不会改写原 `gated_deltanet`
目录或镜像。每个有效 candidate 计满 1,200 秒训练，首 10 iterations 用于吸收 cold Triton
编译且不计时，最终 eval/save 也不计时。模型大小、tokens/second、batch 与 kernel 速度因此
成为优化变量；没有 iso-parameter 限制，15B 只保留为安全上限。

这个 setup 的可比范围仅是同一 32×H100 平台内的 commits。它不支持 matched-token、
matched-parameter、NVlabs release fidelity 或跨硬件排名结论。第一条结果必须现场重跑 H1；
至少两个 mechanism probes 必须先于两个 evidence-led synthesis experiments，`results.tsv`
由冻结工具从 checkpoint/log 自动记录 seconds、wall seconds、GPU-hours、tokens、parameters、
memory 和 metric。
五个最小有效 experiments 的训练计算为 `5×1200×32/3600 = 53.33 GPU-hours`，与已测
pristine 15B 单 candidate 的 `6082.90×32/3600 = 54.07 GPU-hours` 近似；这个选择用大致
相同训练计算换取 baseline、诊断与 synthesis 的闭环，而不是声称更短训练等价于 15B 质量。

### DeepSpec / DSpark

| 项目 | Paper Table 1 | Current env |
|---|---|---|
| 模型 | Qwen3-4B + released block-7 draft | 相同 |
| tasks | 9 个公开 tasks | 5 个：gsm8k/math500/aime25/humaneval/livecodebench |
| sampling | temperature 1.0 | temperature 0，seed 980406 |
| scheduling | confidence scheduler disabled，隔离 raw draft quality | 允许优化 scheduler |
| systems metric | paper 离线 accepted length；私有生产流量另报收益 | custom timed tokens/s × verify rate |
| correctness | target verification semantics | 所有 official cached target-verification call 的全部 committed token 都受 probability-one gate；每 task/rank 第一条另做独立 greedy 诊断（共 20 条，不作 gate） |

current env 没有 report 设想的固定 workload trace、batch 1→256、SLO-constrained goodput、
P50/P99 或 sealed trace。因此它适合做 decode 机制和离线性能回归，不足以证明生产 SLO 收益。
reward 为 `(G/(G+25))*V`，25 的 anchor 和 scalarization 都是本项目选择。
pristine current-env qualification 已由 job 221609 在 4×H100 上完成一次：430 samples、
199,512 output tokens、780.371 token/s、mean verify rate 0.729801；全部 199,679 个 in-path
committed-token checks 有 0 invalid。它尚无重复/dispersion，不能升级为 paper Table 1 结果。

### DeepOCR

| 项目 | DeepSeek-OCR paper | Princeton DeepOCR/current env |
|---|---|---|
| decoder | DeepSeek-3B-MoE | Qwen2-family 7B VILA reproduction |
| vision/compression | DeepEncoder | public SAM/CLIP-style DeepEncoder + 16× compressor |
| training | paper recipe | upstream `pretrain_ocr.sh`: global batch 64、bf16、1 epoch、LR 5e-5 等 |
| current eval | 不等价 | public OmniDocBench + trusted visual-token meter |

训练脚本/registry 中作者的 `/lustre/...` 绝对路径已固定替换为 pinned HF snapshot 与
`/datasets/deepocr/olmOCR-mix-1025`；这是基础设施适配，不是研究变量。easy_deepocr 和
sam_clip 权重已逐文件通过 upstream LFS SHA256 校验后正式封存。固定 revision
`olmOCR-mix-1025@d3c0450...` 的原始 snapshot 也已完整 staging：95 个文件、
77,114,799,826 bytes。upstream 144-DPI conversion 与严格 manifest 审计已由 LSF job
221471 完成；原始 tar/Parquet 下载完成本身仍不算可训练。原始 267,962 个 train rows 中有
5,752 个缺失
`natural_text`；upstream converter 的 falsy check 被 pandas `NaN` 绕过，所以 host prep
wrapper 会按其意图先得到 262,210 rows。固定 144-DPI conversion 又按 upstream 的逐项
exception handling 跳过 219 个异常超大 MediaBox，最终为 261,991 个 JSON/PNG；两层丢弃、
失败原因和 path digest 都写入 manifest。这是披露的数据 sanitation/上游 renderer 行为，
不是 agent 变量。没有 report 设想的 hidden documents 或 olmOCR-bench
hard gate。reward 的四项等权、平方根 token penalty 与 250-token anchor 均为本项目定义。

### RLM

| 项目 | Pinned RLM repository | Current environment |
|---|---|---|
| source/config | `rlm@72d6940...` 的 `training/configs/rlm-qwen3-30b-example.toml` | 同一 vendored source/config；不从外部 publication 补设置 |
| dependency | config 发布前最近的 compatible `prime-rl@84f072b...`（v0.5.0） | 原 TOML 可直接加载；environment 按 pinned README 运行 locked `uv sync --all-extras`；旧 v0.6 pin 不兼容并已归档 |
| released policy | Qwen3-30B-A3B-Instruct-2507 + released `mit-oasys/rlm-qwen3-30b-a3b-v0.1` adapter | base revision `0d7cf239...`、adapter revision `7d722535...` 均只读固定；agent 优化 orchestrator/policy workspace，不做 on-policy weight training |
| data/eval | OOLONG trec_coarse@131072；raw correctness scorer | 同 revision local Parquet；完整固定 50 rows，每 row 5 次独立随机 rollout，共 250；`OOLONG_DATASET_PATH` 仅解决 offline path |
| final serving | repository config 委托 inference-server 默认 sampling temperature | 8×H100 分成两个相同 TP4 vLLM worker，rollout allocation 3+2；每 worker 最多 50 条 active outer rollouts，排队时间不计入 1800 秒 per-rollout timeout |
| primary metric | raw OOLONG reward | trusted verifier 用隐藏 gold 复算 250 条 raw reward 后取算术平均；error/timeout 与 iterations/calls 另报，timeout 留在固定分母计零 |

RLM vendored source 相对 pinned upstream 只有 `OOLONG_DATASET_PATH` patch。兼容的 prime-rl
公开 workspace submodules 被逐项初始化；仓库中的无关私有 `configs/private` 不参与公开
RLM config 或 `uv` workspace，因此不作为构建依赖。tests image 的两个 TP4 worker、
`--enforce-eager`、Triton MoE 和 all-reduce flags 是 Apptainer/H100 compatibility workarounds，
不是 policy 研究变量。

active verifier 不显式设置 evaluation temperature，保持 repository config 对 inference-server
default 的委托。两个 worker 都使用 bounded、work-conserving rolling pool；3-repeat worker 与
2-repeat worker 的首批 50 条各自覆盖 50 个不同 row，随后只按完成顺序补位，不设 wave barrier。
agent 决策自评通过只读 `/task-tools/rlm_eval_pool.py` 使用同一 5×50 schedule；final verifier
独立启动相同服务、使用 masked gold 复算 reward，并把可确定的退化枚举输出计零。

pooled qualification job 304037 已完整跑完 250/250：raw mean `0.25718349663568113`，五个
诊断 repeat 为 `[0.220018, 0.220000, 0.305900, 0.260000, 0.280000]`，11 timeout、0 error、
0 reported-reward mismatch；决策 eval 43 分 21 秒，含冷启动/policy gate 的完整 verifier
56 分 50 秒。旧 job 233113/238576 的 25-row zero-update/60-update training 结果和更早
temperature-0 cost smoke 只保留为历史校准，不能替代当前 250-rollout matched baseline。

### SuperBPE

| 项目 | Pinned SuperBPE repository | Current environment |
|---|---|---|
| source | `PythonNut/superbpe@bbd0976...` 与 submodule `tokenizers-superbpe@757f2a5...` | clean vendored checkout；无 upstream source patch |
| train data | `meta.json` 指定的 `UW/olmo-mix-1124-subset-p99` 顺序与 10GB truncate | revision `64b9a7c...`，十个整文件加 `70.txt` 的 557,641,266-byte prefix，合计精确 10,000,000,000 bytes，并全量 SHA256 封存 |
| baseline | released 200K t80 tokenizer artifact | job 324917 在 pinned current public `eval/135.txt` 上 matched 重算：141,911,558 tokens、6.653858715299285 bytes/token、reward 0.5；不把 notebook 的 private 1GB eval 输出当 baseline |
| task | repository 没有 Harbor reward | 固定 200K/lossless/data budget，允许研究 merge objective、boundary schedule、pretokenization、merge allocation 和实现；主指标 bytes/token，reward `C/(C+B)` |
| compute | tokenizer scripts 为 CPU trainer；仓库未给 exact 200K/t80 wall time | 64 CPU / 512GiB safety envelope；64MiB/8K 两阶段 pilot 为 104s/5,782MB，full 10GB/200K 尚未测量，必须 Step-6 review |

仓库没有发布 exact 200K/t80 command，example shell 是 128K/100K；released artifact 的
serialized pretokenizer 也不能由 committed Stage-2 script 或 `construct_hf_tokenizer` 直接得到。
因此 `prepare-reference-shape` 只称 repository-derived 起点，不称作者命令复原。两阶段 smoke
还实测到 Rust trainer 的 colon/whitespace hardcoded filters 会跳过少数 inherited merges，故
不能把 stage-2 output 的 literal prefix preservation 当官方不变量。详细审计见
[`superbpe/DESIGN_AUDIT.md`](./superbpe/DESIGN_AUDIT.md)。
immutable full reference job 324917 已 DONE：verifier 1,115 秒、LSF peak 13.4GB，
candidate/reference 所有 21 chunks 完全一致。它只测 released artifact 的 matched encoding；
full 10GB/200K candidate training wall 仍未测量，所以 Step 6 保持 review gate。

## 5. 什么时候结果才可称为科学比较

每个 claim 至少同时满足：

1. 当前 SIF SHA256 与本文件/源码一致，不使用 stale 镜像；
2. pristine baseline 和 candidate 使用相同 GPU 型号/数量、软件栈、data revision/order、
   seed、预算与 evaluator；
3. `provenance.json` 和 `experiments.jsonl` 可重建每个优化尝试，完整 trajectory 被保存；
4. 报 raw metrics、失败率、重复次数/多 seed 和不确定性，不只报自定义 reward；
5. 只声称被当前 workload 支持的结论：公开/qualification 结果不得冒充 sealed、hidden、
   full-budget 或 paper-exact 结果。

当前 asset readiness：五项严格 asset 检查均通过。active GDN 已完成 NVlabs 明确引用的
Samba 公开预 tokenized 资产 staging，`train_slim*`/`validation*` packed files 与 versioned
manifest 已封存；DeepOCR 的
模型/eval snapshot 与 261,991 个 Stage-2 JSON/PNG 对已封存，219 个 upstream renderer skips
也由 manifest 固定。
SuperBPE 的 exact 10GB train view、verifier-only eval view、dataset revision 和逐文件 hash
也已封存；agent view 不含 eval，也不含 `70.txt` 未使用的尾部。
GDN matched pristine baseline 已由 job 238329 完成，可以让 agent 优化；其他 task 是否可进入
优化仍按各自 active matched baseline 状态判断，不能用 smoke 代替。

## 6. Active repository references

- GDN: [authors' implementation](https://github.com/NVlabs/GatedDeltaNet) and
  [0.4B/15B public script](https://github.com/NVlabs/GatedDeltaNet/blob/main/scripts/tsz512x4k_15B_gated_deltanet_h1_0.4B.sh).
- DeepSpec: [repository](https://github.com/deepseek-ai/DeepSpec).
- DeepOCR: [public repository](https://github.com/Princeton-AI2-Lab/DeepOCR).
- RLM: [current official repository](https://github.com/alexzhang13/rlm).
- SuperBPE: [official repository](https://github.com/PythonNut/superbpe).
