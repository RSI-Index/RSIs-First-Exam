# AutoResearch instruction 编写规范

这份规范供以后新增或审查环境时复用。目标不是把 instruction 写得长，而是让 agent 在第一屏
就拿到做实验所需的真实边界，并让结果可以被另一个研究者复核。

## 永久 source priority

1. 只要作者公开仓库中存在可运行的模型、config、训练脚本或 evaluator，task 的输入、baseline、
   metric 和固定超参数就优先且只从该 pinned repository 取得。
2. 不用论文为公开仓库补模型尺寸、数据、训练超参数、评测协议或 baseline；仓库没有公开的内容
   就明确写 `missing/unmeasured`，不能从论文猜出一套混合 setup。
3. 环境只允许增加运行所必需的基础设施适配，例如离线资产路径、容器依赖 pin、GPU topology
   映射、artifact 导出、审计和 scalar reward。适配不得改变 upstream 的科学语义，并必须逐项披露。
   若仓库已发布明确的多节点/GPU topology，应直接分配该 topology，不得为省卡而缩成单节点；
   scheduler/container 可以翻译，但 world size、rank partition、global batch 和数据顺序必须保留。
4. 第三方 checkpoint、同名 config 和后续独立训练框架不能替代作者仓库 baseline。若必须采用，
   应建立不同 task/version，不能继续称为作者公开 recipe。
5. 论文信息只可保留为历史审计背景，不参与以后 task 的设计决策、输入或 baseline。

## 必须按顺序出现的信息

1. **一句话目标与提交物**：优化什么，最后交 checkpoint、workspace 还是其他 artifact；
2. **Setup table**：可写目录、pinned code/model/data revision、train/eval entrypoint、只读资产、
   GPU topology、时限、网络边界和输出路径；
3. **机制/mental model**：解释一次 rollout/decode/train candidate 实际发生什么，指出真正有
   因果意义的 research variables；
4. **Fidelity**：写清 pinned upstream、缺失的公开资产、以及 Current env 的每一项基础设施
   适配，不能只写“official setup”；
5. **Baseline 与可复制命令**：先测 pristine baseline；给出能直接执行的 train/eval 命令和
   固定 seed/budget。没测过就写 `unmeasured`，禁止填推测分数；
6. **Iteration loop**：每轮先写 falsifiable hypothesis，再改变一个可解释变量组，记录失败与
   rejected candidate；明确一次 optimization attempt 的计数单位；
7. **Reward 与 gates**：同时给原始指标、公式、常数来源、hard-zero 条件和不支持的科学 claim；
8. **Editable/locked scope**：允许改什么，禁止改 evaluator/data/known answers 等什么；
9. **Required outputs**：artifact、`provenance.json`、`experiments.jsonl`、日志和完整 trajectory。

## 科研表述底线

- upstream raw metric 与环境自定义 scalar 不得混称；
- smoke/qualification/full-budget/sealed run 必须分名；
- baseline 和 candidate 必须匹配 hardware、SIF、data order/revision、seed、budget 和 evaluator；
- GPU allocation 与 verifier 实际使用的 GPU 数分别报告；trajectory steps 与科学优化次数分别计数；
- reward 只是选择信号。科学结论至少报告 raw metrics、失败率、重复/多 seed 和不确定性；
- 资产或复现信息缺失时让 preflight 失败，不临时换数据、联网补件或编造 baseline。
