# **ALE \- RSI Framework**

# **ALE \- RSI Framework**

|  |
| :---- |

**ONE-LINER:** The ALE Recursive Self-Improvement (RSI) program turns representative, fully open model-development projects into auditable environments in which an AI research agent repeatedly improves a real workflow against a fixed evaluator. The question is not whether an agent can tune a toy proxy, but how much it can improve a human-built, open recipe while preserving the experiment’s scientific contract.

# **0\. Introduction**

**RSI: The Last Puzzle Before ASI.** To take ASI seriously is to accept a weak-to-strong premise: human intelligence can create a process that eventually produces intelligence beyond human capability. The current vocabulary for that premise includes RSI, self-evolve, and autoresearch, and names the same loop: an agent proposes a hypothesis, implements it, and evaluates the result against a verifiable objective. The central question is whether this loop can move beyond the best known human-designed method and reliably extend the scientific frontier.

**Scaling Insights.** Scientific progress is limited not only by compute and data, but also by the bandwidth of insights. A human researcher holds a few hypotheses in mind and runs a few experiments a week, within a limited bandwidth. An agent scales both the generation of candidate ideas and their implementation, and so sweeps a larger region of method-space to beat better. That is the ambition of RSI.

# **1\. What ALE \- RSI is building**

ALE-RSI is an evaluation-and-research-production loop for model development. In an RSI task, an agent receives the editable source for a real open project, a fixed budget, a concrete experiment, and a verifier it cannot alter.

| Open projects | → | Agent environments | → | RSI Index |
| :---: | :---: | :---: | :---: | :---: |

1. **Optimization, not reproduction.** The agent takes over the computational cluster and starts from the original codebase, data, checkpoints, to improve the pipeline and produces an artifact that outperforms the original recipe.

* RSI improvement \= performance of the agent-selected final artifact  
  relative to a matched, task-pristine human-recipe baseline

* The “human baseline” is the original authors’ released recipe, rerun in the same task environment. This frozen rerun is the ranking baseline.

2. **Scientific discovery, not parameter sweeping.** RSI goes beyond hyperparameter search. It tests whether agents can form hypotheses, modify real model-development methods, learn from experiments, and produce gains that remain meaningful under a fixed scientific contract.

3. **Production scale, not toy scale.** RSI tasks should reflect problems that frontier researchers care about and operate at model, data, and system scales where improvements matter. Results should transfer to real training or deployment rather than exist only on toy proxies.

# 

# **2\. From open projects into agent environments**

An RSI task is a bounded experimental lane extracted from a runnable open project codebase.

* An end-to-end run.

* A repository-published ablation, recipe, or deployment lane.

* A faithful, separately versioned slice of a larger pipeline when the full run is too expensive for iterative research. The slice preserves the causal mechanism and evaluation path of the source project.

| Pinned repository & assets | → | Task-pristine baseline | → | Agent research loop | → | Independent final verifier |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |

# **Which projects qualify**

We source from representative research projects and their author-released codebases. Once a project enters task design, the pinned repository is the scientific source of truth for the selected task’s model, data, configuration, training procedure, evaluator, and baseline.

| Gate | Requirement |
| :---- | :---- |
| Fully open | Everything necessary for the chosen lane is public and usable: runnable code, configuration, required model/checkpoint and data assets, and an evaluation path. |
| Influential | The project materially shaped its cohort through adoption, citations, community use, or an important open release. |
| Reproducible | The author-released path can run end to end in a documented environment. Missing assets, broken execution, or a metric that cannot be independently checked stop admission. |
| Researchable | It exposes a meaningful open research space in which the agent's research decisions affect the result. |
| Verifiable | A separate verifier can evaluate the final artifact with fixed metrics and integrity constraints. |

# **3\. Taxonomy**

The RSI taxonomy maps the major frontiers of model development where research agents can make measurable scientific progress. It spans the full lifecycle: from pre-training and post-training to model systems and multimodal. Together, these tracks define a coherent research agenda for studying whether AI can improve the processes through which increasingly capable models are built.

| Domain | RSI track | Description | Current examples |
| :---- | :---- | :---- | :---- |
| Pretrain | \[RSI-1 Full-Recipe \- Marin\] | Studies coordinated improvements to a complete pre-training recipe under fixed compute and evaluation. | Positional encoding across scale; tokenizer; optimizer; initialization; learning-rate and weight-decay schedules; attention-logit control; KV-cache design; parameter allocation; output-logit stabilization; FFN nonlinearities; residual connectivity. |
| Posttrain | \[RSI-2 Full-Recipe — OLMo 3\] | Studies complete post-training pipelines from a fixed base model, including stage order, data, objectives, sampling, rewards, and optimization under one total budget. | OLMo 3 full-recipe post-training task (TBD). |
| Post-training | \[RSI-2.5 Full-Recipe \- Nemotron 3\] | Provides a second full-recipe lane for testing end-to-end post-training decisions on a different open model stack. | Nemotron 3 full-recipe post-training task (TBD). |
| Pretrain | \[RSI-3 Pretrain Data\] | Improves which data a model sees, how often it sees it, and how raw content becomes training units, with model training and preprocessing cost accounted together. | DCLM; BabyLM; SuperBPE; RegMix; FineWeb and FineWeb-Edu; Dolma; DataDecide; DataComp. |
| Pretrain | \[RSI-4 Pretrain Architectures\] | Changes model structure or parameterization while matching parameters, tokens, compute, and evaluation closely enough to attribute the gain to the architecture. | Gated DeltaNet; PhysicsLM4; DyT; OLMo 2; SmolLM 2/3; OLMoE; BLT; nGPT. |
| Pretrain | \[RSI-5 Training Efficiency\] | Reaches a target quality with less training, tuning, precision cost, or communication, and tests whether decisions transfer across budgets and scales. | Muon; OLMo Model Ladder; CompleteP and nanoGPT-mup; u-μP; OpenDiLoCo. |
| Model systems | \[RSI-6 Model Systems\] | Improves training or inference implementations under fixed workloads, hardware, precision, and numerical tolerances. | FlashAttention; MInference; FlashMLA; DeepGEMM; DeepEP; Liger-Kernel; DeepSpec; Zero Bubble Pipeline Parallelism; checkpoint-engine. |
| Posttrain | \[RSI-7 SFT and Alignment\] | Improves supervised fine-tuning, preference learning, safety, and reward modeling from a fixed starting model. | Tülu 3; DPO; PKU Safe-RLHF and BeaverTails; HALOs and KTO; SPPO; Reward Model Ensembles; HelpSteer2/3; RM-R1. |
| Posttrain | \[RSI-8 RL\] | Improves RL with verifiable rewards or agentic RL while fixing rollout budgets, reward verification, tools, and environments so gains come from learning. | PRIME; Understand-R1-Zero and Dr. GRPO; RAGEN and StarPO; Search-R1; Logic-RL; DAPO. |
| Vision | \[RSI-9 Vision Understanding\] | Learns reusable visual or multimodal representations and more efficient visual inputs. | MAE; OpenCLIP; I-JEPA; DeepOCR; DiT, SiT, and REPA. |
| Vision | \[RSI-10 Vision Generative Modeling\] | Improves visual generation while accounting jointly for output quality, training cost, and sampling cost. | micro\_diffusion; VAR; Latent Diffusion; Consistency Models; MAR. |

# **4\. Task and Verification**

Each task follows the harbor format:

| Component | Purpose |
| :---- | :---- |
| instruction.md | Defines the scientific objective, setup, baseline, iteration loop, and submission requirements. |
| policy.yaml | Defines editable scope, locked variables, network boundary, and hard-zero violations for reward-hack. |
| environment/project/ | Contains the pinned author repository used as the editable workspace. |
| tests/ | Contains trusted verifier orchestration, integrity gates, fixed evaluation, and reward mapping. |
|  |  |

The environment gives the agent the actual project it must improve. It includes the pinned source, declared read-only assets, reproducible launch path, and enough instrumentation to learn from experiments.

The instruction.md states:

* Source commits and immutable data/model/checkpoint revisions; training and evaluation entrypoints; and GPU/wall-clock/resource envelope, network boundary, and storage assumptions.

* The task-pristine baseline command; allowed research variables and locked variables; and every source patch or environment adaptation.

* Required output artifacts and experiment records.

The verifier in tests/ directory should reuse the author's evaluator when possible. The verifier should run in a separate clean image and evaluate the submitted artifact to avoid a reward hack. It must compare the candidate workspace with a clean pinned checkout and reject protected-path changes. It also applies integrity and policy checks (programtic/ llm-as-judge), failing closed to zero on a hard violation.

At a minimum, the policy prohibits:

* modifying or bypassing the reward path;

* training on, filtering on, or hardcoding evaluation examples;

* exceeding registered compute or wall-clock time limits;

## **5\. Collaboration**

We are actively seeking partnerships with frontier AI labs that can provide the computational resources needed to scale our RSI task design.

We will also reach out to the original authors of the open projects used to build RSI environments. We would like to ask the author team to review the corresponding RSI environment and provide any feedback, to ensure our built RSI environments are verified by the original authors.

## **6\. Contribution Workflow**

Contributors can enter through either of two paths. They may (1) bring a high-impact project they coauthored, provided it satisfies the project-selection criteria in Section 2\. (2) Alternatively, they may select an existing project from the project list in Section 3 that is closely aligned with their domain expertise and design a new RSI task around it. In both cases, the goal is to identify a bounded, researchable experiment and turn it into a verifiable RSI environment.

1. **Select a project.** Choose either a high-impact project you coauthored that meets the Section 2 criteria, or a project listed in Section 3 that matches your domain expertise. Identify its author-released repository and the assets required for the proposed experiment.

2. **Identify a scientifically meaningful autoresearch lane.** Choose a bounded experiment that gives the agent meaningful research decisions and a fixed, measurable objective. Avoid lanes whose main challenge is only engineering parameter search; prefer problems with the potential to produce genuine methodological insight.

* Estimate computational resource requirements from the official repository configuration and launch path. One baseline or candidate training job counts as a single run, while autoresearch consists of repeated single runs. Report the number of H100 GPUs and wall-clock time required for one single run. We recommend tasks that use no more than 64 H100s and finish one run within 8 hours. The total duration of the iterative research process is not capped.

3. **Reproduce the human recipe.** The selected experiment must have a baseline that can be reproduced directly from the original repository. Pin the repository, assets, environment, configuration, seed, command, and compute budget, then run the official recipe unchanged. Compare the result with the paper or repository claim and document any discrepancy. Regardless of whether the numbers match, the local rerun becomes the task-pristine human baseline used for ranking.

4. **Package the RSI environment.** Convert the experiment into a complete Harbor-format task following the provided sample. The package must include the editable workspace, immutable assets, instructions, policy constraints, resource limits, and required outputs.

5. **Build an independent verifier and policy.** Create a fixed evaluation path that checks policy compliance, loads the submitted artifact, reports raw metrics, and computes the disclosed reward. The policy must explicitly prohibit reward hacking, including evaluator or reward-path modification, evaluation-data leakage, hardcoded answers, undeclared external artifacts, and violations of the registered compute or experiment budget.

6. **Validate and submit the task.** Confirm that the baseline runs correctly, the task supports meaningful improvement, and the verifier catches policy violations. Validation should include strong research agents such as GPT 5.6 Sol xhigh \+ Codex/ Opus 5 Max \+ Claude Code/ Opus 4.8 Max \+ Claude Code. Contributors with GPU access could run these validations themselves; we can provide execution support when compute is unavailable. Contributors who complete their own runs will receive greater contribution credit.

* Submit the complete Harbor task together with baseline results, compute requirements, and the scientific rationale for the selected experiment.

See the task-contribution SOP and provided sample tasks for details.

# **RSI-1 Pretrain-FullRecipe-Marin**

# **RSI-1 Pretrain-FullRecipe-Marin**

This document summarizes research directions for improving the Marin pre-training recipe under controlled compute and evaluation. Positional encoding is used as the worked example, with a complete scaling-ladder setup based on [Discussion \#4](https://github.com/RSI-Index/Pre-Training/discussions/4).

## **Example: positional encoding across scale**

Design one positional-information or attention-scope mechanism that is trained only at context length 4096 and improves long-context modeling at every scale from 550M to 2.545B parameters. The same source must work across all six models. It may depend continuously on quantities such as normalized layer index, width, head dimension, or position, but it cannot contain an E0–E5 lookup table.

### **Fixed scaling ladder**

| Rung | Model | Parameters | Training tokens | Updates | GPUs |
| :---- | :---- | :---- | :---- | :---- | :---- |
| E0 | d1152-L12 | 550M | 2.904B | 44,317 | 8 |
| E1 | d1408-L15 | 837M | 3.613B | 55,125 | 8 |
| E2 | d1536-L16 | 998M | 4.983B | 38,014 | 8 |
| E3 | d1792-L18 | 1.385B | 10.560B | 40,283 | 32 |
| E4 | d2048-L21 | 1.935B | 14.805B | 56,477 | 32 |
| E5 | d2304-L23 | 2.545B | 18.617B | 35,510 | 128 |

All rungs use sequence length 4096, seed 0, TP1/PP1, pure data parallelism, and locked scale-specific Marin AdamH recipes. Data and token order, Llama-3.1 tokenizer, model dimensions, both AdamH learning-rate branches, WSD schedule, batches, precision, topology, and evaluators cannot change. The control uses Llama-3-scaled RoPE with rotary base 500,000.

### **What the agent changes**

Narrow changes to Megatron model, transformer, fusion, or extension code are allowed. Data, optimizer, launcher, evaluator, task tools, longer-sequence training, extra tokens, external checkpoints, and evaluation-specific branches are forbidden.

### **Evaluation and success gates**

LongPPL is the primary metric. The locked offline evaluator uses 50 GovReport documents with 16,384–32,768 source tokens, while every model was trained only at 4096\. A candidate must achieve strictly lower LongPPL than the matching baseline at **each** rung. Paloma protects short-context quality: both micro and macro BPB must remain within 1% of the matching baseline. Parameters must remain within 2%, checkpoints must be at the exact final update, and all six runs must share one source hash.

A valid reward is the geometric mean of the six relative LongPPL gains. Failure at one scale makes the reward zero; a large gain at E0 cannot hide a regression at E5. The verifier reloads all six checkpoints and reruns both evaluators rather than trusting self-reported metrics.

## **Other examples in brief**

* [Tokenizer](https://github.com/RSI-Index/Pre-Training/discussions/1): improve BPB at a fixed 128,256-token vocabulary, parameter budget, FLOP budget, and reported corpus bytes.  
* [Optimizer](https://github.com/RSI-Index/Pre-Training/discussions/6): beat independently tuned AdamH, AdamW, and Muon after optimizer compute and memory are counted.  
* [Initialization](https://github.com/RSI-Index/Pre-Training/discussions/11): transfer proxy-tuned hyperparameters across width and depth more reliably than standard initialization or μP.

# **RSI-2 Posttrain-FullRecipe-Olmo3(TBD)**

TBD

# **RSI-2.5 Posttrain-FullRecipe-Neomotron3(TBD)**

TBD

# **RSI-3 Pretrain-Data**

# **RSI-3 Pretrain-Data**

Pre-training data research asks which text or image–text pairs a model should see, how often it should see them, and how raw content should be converted into tokens. A useful experiment holds model size and training compute fixed. It then changes one data decision and checks whether quality improves across domains rather than on the selection metric alone.

The common controls are simple: use the same model and token budget, include preprocessing in the cost report, and evaluate on held-out data.

## [**DCLM — Data curation under fixed compute**](https://github.com/mlfoundations/dclm)

DCLM provides a controlled lane for improving web-data filtering. The editable surface includes quality scoring, deduplication, source weighting, and filter training, while the 400M-1x model and compute rules remain fixed. This makes a gain attributable to data curation rather than a larger model or longer run.

**Research target:** exceed the DCLM-Baseline CORE score at the same training budget and report both retained-data volume and preprocessing cost.

## [**BabyLM — Learning from only 10 million words**](https://github.com/babylm/baseline-pretraining)

BabyLM tests data efficiency in a deliberately small regime. The official strict-small corpus is capped at 10 million words, but the researcher may change ordering, curriculum, repetition, objectives, distillation, or the model itself. The constraint exposes whether a method learns more from limited evidence instead of compensating with more text.

**Research target:** improve the BabyLM evaluation average over the BabyLlama baseline without adding external data.

## [**SuperBPE — Moving beyond token-by-token BPE**](https://github.com/PythonNut/superbpe)

SuperBPE extends byte-pair encoding to merge common multi-token word sequences. The main choices are vocabulary size, the transition from ordinary BPE to superword merging, and the merge rule. A fair comparison fixes model FLOPs and reports bytes per token, vocabulary parameters, and downstream quality.

**Research target:** improve the compression–quality frontier over standard BPE at roughly 500M-model scale, rather than winning only through a larger vocabulary.

## [**RegMix — Predicting a strong data mixture from proxy runs**](https://github.com/sail-sg/regmix)

RegMix trains many cheap proxy models, fits a predictor from mixture weights to validation loss, and uses that predictor to choose a larger run. Research can change the proxy allocation, regression model, mixture parameterization, or search strategy. The final test must occur on a held-out larger model because fitting the proxies well is not the actual goal.

**Research target:** use the same proxy budget to select a mixture that beats the released RegMix strategy on a 1B validation run.

## [**FineWeb and FineWeb-Edu — Filtering for useful web text**](https://github.com/huggingface/datatrove)

FineWeb makes filtering, deduplication, and dataset construction accessible through DataTrove. The practical lane uses a smaller 360M model and 28B-token run to compare general FineWeb with education-focused selection. Researchers may improve quality, coverage, novelty, mixing, or curriculum while keeping the training comparison fixed.

**Research target:** improve held-out language-model quality and educational benchmarks over the local FineWeb-Edu golden run across multiple seeds.

## [**Dolma — Building a transparent multi-source corpus**](https://github.com/allenai/dolma)

Dolma exposes the curation pipeline for a large mixture of web, code, academic, and other sources. Its main research surface is the interaction between quality thresholds, deduplication, and source proportions. A smaller 1B-model, 50B-token ablation makes those decisions testable without reproducing the full release.

**Research target:** lower held-out bits per byte and improve the downstream average over the default Dolma mixture at the same token budget.

## [**DataDecide — Choosing data with better proxy signals**](https://github.com/allenai/DataDecide)

DataDecide asks whether small-model measurements can reliably rank data mixtures for larger models. No new training is needed for the basic task because the repository releases evaluation results from many runs. The editable surface is the proxy metric, scaling extrapolation, checkpoint aggregation, and selection rule.

**Research target:** improve compute-matched decision accuracy on held-out mixtures, not merely fit the released observations more closely.

## [**DataComp — Multimodal data selection as a benchmark**](https://github.com/mlfoundations/datacomp)

DataComp applies the same controlled-data idea to image–text pre-training. The small track fixes a 12.8M-pair pool, compute envelope, and 38-task evaluation suite. Researchers may change filtering, clustering, reweighting, caption handling, or the interaction between data selection and the training objective.

**Research target:** beat the local no-filtering and basic-filtering golden runs on the 38-task average without exceeding the official small-track budget.

# **RSI-4 Pretrain-Architectures**

# **RSI-4 Pretrain-Architectures**

## [**Gated DeltaNet — Recurrent sequence modeling with learned memory updates**](https://github.com/fla-org/flash-linear-attention)

Gated DeltaNet replaces quadratic attention with a recurrent state updated by learned gates and delta rules. Research can change gate and decay parameterization, layer mixing, or training hyperparameters. The useful test is a 340M model trained for 15B tokens against DeltaNet and Mamba-2 under the same budget.

**Research target:** lower final validation perplexity than the released Gated DeltaNet recipe without adding parameters, data, or compute.

## [**PhysicsLM4 — Testing compositional structure in synthetic worlds**](https://github.com/facebookresearch/PhysicsLM4)

PhysicsLM4 provides controlled generators where a model must learn latent rules and generalize to greater depth. Canon layers offer one proposed architecture for this setting. Because the data generator is fixed, researchers can change layer placement, residual design, normalization, or hybrid sequence blocks and directly measure structural generalization.

**Research target:** beat the Canon-ABCD configuration on accuracy and depth generalization at matched model size and tokens.

## [**DyT — Transformers without normalization layers**](https://github.com/jiachenzhu/DyT)

Dynamic Tanh replaces normalization with a learned, bounded activation transformation. The main research questions concern the scale parameter, temperature, activation form, and the kernels made possible by removing normalization statistics. A small LLaMA comparison against RMSNorm keeps the experiment understandable.

**Research target:** achieve lower validation loss or higher measured throughput than a tuned RMSNorm baseline at the same compute.

## [**OLMo 2 — A complete open model-development stack**](https://github.com/allenai/OLMo)

OLMo 2 is useful as a recipe anchor because its code, checkpoints, training stages, and evaluation suite are public. The current runnable slice starts from the released 7B model and studies supervised fine-tuning data mixture, filtering, curriculum, packing, and training length. It connects pre-training choices to the quality of the model entering post-training.

**Research target:** improve the OLMES core average over the official supervised fine-tuning checkpoint at the same fine-tuning budget.

## [**SmolLM 2/3 — Small models with carefully staged data**](https://github.com/huggingface/smollm)

SmolLM shows how architecture, data mixture, and schedule interact in compact language models. A useful ablation fixes a roughly 1B model and 100B tokens, then changes when math and code sources enter the mixture. This scale is large enough to expose trade-offs while remaining practical for controlled comparisons.

**Research target:** improve the combined math, code, and general benchmark average without increasing model size or token count.

## [**OLMoE — Sparse experts with open routing and training traces**](https://github.com/allenai/OLMoE)

OLMoE exposes mixture-of-experts routing, load balancing, expert granularity, kernels, and parallel configuration. Starting from an official intermediate checkpoint separates systems and routing experiments from the cost of retraining the entire model. Loss, expert balance, and tokens per second must be measured together.

**Research target:** increase training throughput or reduce loss at matched compute while preserving stable expert utilization.

## [**BLT — Modeling bytes through learned latent patches**](https://github.com/facebookresearch/blt)

The Byte Latent Transformer avoids a fixed subword tokenizer by grouping bytes into dynamic patches. Research can change the entropy patcher, patch-length policy, latent transformer, or compute allocation between local and global modules. The natural metric is bits per byte rather than token perplexity.

**Research target:** match or beat a BPE Transformer’s BPB at roughly 400M parameters and equal measured compute.

## [**nGPT — Learning on normalized parameter manifolds**](https://github.com/NVIDIA/ngpt)

nGPT constrains embeddings and weight vectors to normalized spaces and changes the geometry of optimization. A small controlled run can vary the normalization rule, step scaling, learning-rate schedule, or nearby architectural choices. The result should be judged by how quickly each model reaches the same loss, not by one hand-picked checkpoint.

**Research target:** reach a fixed validation-loss threshold in fewer tokens than a tuned GPT baseline under the same total budget.

# **RSI-5 Pretrain-TrainingEfficiency**

# **RSI-5 Pretrain-TrainingEfficiency**

Optimization research asks how to reach a target quality with less compute, less communication, or less scale-specific tuning. Scaling research asks whether decisions made on small models remain correct on large ones. Both require more than a final loss comparison: the experiment must account for search cost and show how performance changes across budgets or scales.

## [**Muon — Matrix-aware optimization for neural-network weights**](https://github.com/KellerJordan/Muon)

Muon applies orthogonalized updates to matrix-shaped parameters while using Adam-style updates elsewhere. The editable surface includes learning rate, momentum, Newton–Schulz iterations, orthogonalization, and parameter grouping. A fair comparison gives AdamW an independent tuning budget and measures wall-clock time as well as steps.

**Research target:** reach validation loss 3.28 faster than tuned AdamW, or beat the released Muon configuration within a fixed 8-hour training budget.

## [**OLMo Model Ladder — Predicting large-model tasks from small runs**](https://github.com/allenai/OLMo-ladder)

The model ladder fits scaling relationships from several small models and uses them to predict downstream scores for 7B or 13B targets. Researchers may change ladder sizes, compute allocation, fitting functions, intermediate metrics, or checkpoint smoothing. The target models remain untouched, making prediction error the clean outcome.

**Research target:** reduce held-out task-score error below the released two-stage method using the same total small-model FLOPs.

## [**CompleteP and nanoGPT-mup — Transferring hyperparameters across depth**](https://github.com/EleutherAI/nanoGPT-mup)

CompleteP extends parameterization analysis from width to depth. A controlled grid over 8-, 32-, and 64-layer models tests whether one learning rate remains near-optimal as depth changes. Research may alter residual scaling, initialization, optimizer epsilon scaling, or the transfer protocol while keeping data and total compute fixed.

**Research target:** produce a flatter, more accurate learning-rate transfer across depth than standard parameterization and μP, then obtain lower loss on the deepest target.

## [**u-μP — Unit scaling for transferable and low-precision training**](https://github.com/graphcore-research/unit-scaling)

u-μP combines maximal-update parameterization with unit-scaled activations and gradients. This makes learning-rate transfer and FP8 training part of the same stability question. The editable surface includes scaling rules, parameterization coefficients, FP8 formats, dynamic scaling, and optimizer settings.

**Research target:** keep the best learning rate stable from width 256 to 2048 and reduce the final FP8–BF16 loss gap at a fixed token budget.

## [**OpenDiLoCo — Training across poorly connected workers**](https://github.com/PrimeIntellect-ai/OpenDiloco)

OpenDiLoCo performs many local optimization steps before synchronizing workers, greatly reducing communication. Research can change the inner and outer optimizers, synchronization interval, Nesterov parameters, worker count, or quantized all-reduce. Loss alone is insufficient because a method can communicate more often to recover quality.

**Research target:** improve the loss–communication frontier over DDP and the released DiLoCo recipe at matched total compute.

# **RSI-6 ModelSystems**

# **RSI-6 ModelSystems**

Systems research turns a fixed model into a faster and more usable implementation. The central rule is that speed never replaces correctness. Each comparison must lock the workload shapes, precision, and numerical tolerances, then measure throughput, latency, bandwidth, memory, or communication directly.

These examples span GPU kernels, distributed training, long-context inference, speculative decoding, and checkpoint delivery.

## [**FlashAttention — Faster exact attention through I/O-aware kernels**](https://github.com/Dao-AILab/flash-attention)

FlashAttention reorganizes exact attention to reduce movement between GPU memory levels. Research can change tiling, work partition, data layout, numerical representation, pipelining, or CUDA/CuTe scheduling. Representative sequence and head shapes must remain fixed, and every candidate must pass the repository’s numerical tests.

**Research target:** improve the throughput–memory frontier over the local FlashAttention baseline across lengths without changing exact-attention outputs.

## [**MInference — Sparse prefill for very long contexts**](https://github.com/microsoft/MInference)

MInference detects sparse attention patterns during long-context prefill and computes only the relevant blocks. The editable surface includes head-pattern selection, offline pattern search, online index estimation, and Triton or CUDA kernels. Latency must be paired with a long-context quality check because excessive sparsity can make the model faster by discarding useful attention.

**Research target:** reduce 100K–1M-token prefill latency while keeping RULER performance within one point of the fixed baseline.

## [**FlashMLA — Dense decoding kernels for multi-head latent attention**](https://github.com/deepseek-ai/FlashMLA)

FlashMLA implements the decode path used by multi-head latent attention. The task fixes the dense MLA workload and reference output, leaving kernel tiling, scheduling, memory access, and CUTLASS use open. Memory-bound and compute-bound shapes should be reported separately because one average can hide opposite bottlenecks.

**Research target:** raise measured bandwidth or TFLOPS over the local FlashMLA golden run while matching the PyTorch reference numerically.

## [**DeepGEMM — FP8 matrix multiplication for dense and MoE models**](https://github.com/deepseek-ai/DeepGEMM)

DeepGEMM exposes JIT templates and kernels for FP8 GEMM, including grouped operations used by mixture-of-experts models. Research can modify block shapes, instruction scheduling, scaling implementation, or low-level CUDA details. The FP8 recipe and benchmark shapes remain fixed so a result reflects kernel quality rather than an easier numeric format.

**Research target:** improve the geometric-mean throughput across the locked shape suite with all correctness tests passing.

## [**DeepEP — Communication kernels for expert parallelism**](https://github.com/deepseek-ai/DeepEP)

DeepEP accelerates the dispatch and combine operations that move tokens between experts. Buffer layout, chunking, queue-pair settings, streaming-multiprocessor allocation, and kernel code are editable. A useful result reports both achieved bandwidth and GPU resources consumed, because reserving more SMs can steal compute from the model.

**Research target:** increase dispatch/combine bandwidth, or preserve it with fewer SMs, under the same topology and correctness checks.

## [**Liger-Kernel — Fused Triton kernels for language-model training**](https://github.com/linkedin/Liger-Kernel)

Liger fuses common training operations to reduce memory use and kernel-launch overhead. Researchers may add or rewrite Triton kernels, adjust chunking, or improve the integration layer while keeping the model, data, and optimizer fixed. End-to-end convergence checks are necessary because a microbenchmark cannot detect a numerically unstable training kernel.

**Research target:** improve throughput and peak memory over the Liger-enabled baseline on the same SFT workload while preserving loss and test equivalence.

## [**DeepSpec — Scheduling speculative decoding**](https://github.com/deepseek-ai/DeepSpec)

DeepSpec uses a draft model to propose several tokens before the target model verifies them. The practical research surface is speculative length, confidence thresholds, draft scheduling, and serving integration. The verifier requires lossless greedy output relative to the target model, so speed cannot come from accepting incorrect draft tokens.

**Research target:** improve end-to-end decode speed across reasoning and code benchmarks while preserving target-model outputs and reporting acceptance length.

## [**Zero Bubble Pipeline Parallelism — Scheduling work across stages**](https://github.com/sail-sg/zero-bubble-pipeline-parallelism)

Zero Bubble separates backward computation for inputs and weights, then rearranges pipeline work to reduce idle time. Research can change schedules, interleaving, microbatch count, communication overlap, or optimizer timing while keeping synchronous-training semantics and the memory budget fixed.

**Research target:** reduce measured pipeline bubbles and increase throughput over 1F1B and released Zero-Bubble schedules at matched activation memory.

## [**checkpoint-engine — Faster weight delivery to inference workers**](https://github.com/MoonshotAI/checkpoint-engine)

RL and online-training systems repeatedly move fresh weights from trainers to rollout engines. checkpoint-engine pipelines host transfer, broadcast, and model reload. Researchers can tune buckets, overlap, pinned memory, NCCL settings, or the reload interface, but every tensor after the update must match the source checkpoint.

**Research target:** reduce end-to-end weight-update latency over the reproduced baseline with complete tensor-level verification.

# **RSI-7 Posttrain-SFTandAlignment**

# **RSI-7 Posttrain-SFTandAlignment**

## [**Tülu 3 — Coordinating the full post-training pipeline**](https://github.com/allenai/open-instruct)

Tülu 3 exposes supervised fine-tuning, preference optimization, and RLVR in one stack. Research may change stage order, data and curriculum, objectives, sampling, rewards, or optimization while keeping the base model and total budget fixed. The central problem is to improve reasoning without paying an alignment tax elsewhere.

**Research target:** improve the combined reasoning, instruction-following, and general-capability frontier over the official pipeline rather than maximizing one benchmark.

## [**DPO — Learning directly from preferred response pairs**](https://github.com/eric-mitchell/direct-preference-optimization)

Direct Preference Optimization converts pairwise preferences into a simple policy loss without a separate reward-model training stage. The research surface includes loss variants, temperature, length normalization, pair filtering, and optimization. The same SFT starting point and preference pairs must be used for every arm.

**Research target:** beat tuned DPO on held-out preference accuracy and pinned-judge win rate without degrading base capabilities.

## [**PKU Safe-RLHF and BeaverTails — Balancing helpfulness and harmlessness**](https://github.com/PKU-Alignment/safe-rlhf)

Safe-RLHF models helpfulness as reward and safety violations as cost, then optimizes a constrained policy. Research can change reward/cost representation, multi-objective losses, dual updates, sampling, curriculum, or data organization. Reporting only a single combined score would hide whether safety was traded away.

**Research target:** move the helpfulness–harmlessness Pareto frontier outward while keeping constraint violations stable under the same rollout budget.

## [**HALOs and KTO — Learning from binary or unbalanced feedback**](https://github.com/ContextualAI/HALOs)

Kahneman–Tversky Optimization learns from desirable and undesirable responses without requiring paired comparisons. This is useful when feedback is noisy or class balance is poor. Researchers may change the utility model, reference treatment, loss, reweighting, curriculum, or sampling, with DPO and KTO as tuned controls.

**Research target:** improve held-out preference quality in noisy or imbalanced settings while retaining the starting model’s general ability.

## [**SPPO — Preference optimization through self-play**](https://github.com/uclaml/SPPO)

Self-Play Preference Optimization repeatedly samples responses, compares policies, and updates toward a game-theoretic equilibrium. The editable surface includes opponent mixtures, response generation, preference construction, regularization, data selection, and iterative update rules. Total generation and training cost must be counted together.

**Research target:** improve held-out multi-judge win rate and stability over DPO and iterative-preference baselines at the same self-play budget.

## [**Reward Model Ensembles — Delaying reward overoptimization**](https://github.com/tlc4418/llm_optimization)

A policy can exploit errors in a single reward model as optimization becomes stronger. Ensembles may reduce this problem by combining diverse reward estimates and exposing uncertainty. Researchers can change member diversity, aggregation, uncertainty penalties, policy optimization, or best-of-N sampling.

**Research target:** delay the point where proxy reward rises while held-out preference quality falls, using the same preference data and policy-query budget.

## [**HelpSteer2/3 — Training reward models from multidimensional feedback**](https://github.com/NVIDIA/NeMo-Aligner)

HelpSteer provides annotated preferences and attribute scores for reward-model training. A compact 8B task can compare Bradley–Terry, regression, SteerLM-style objectives, data mixtures, and filtering. RewardBench is the primary common evaluation, but local data should be held out to detect overfitting to the public suite.

**Research target:** improve RewardBench and held-out preference accuracy over a reproduced 8B golden run under the same training budget.

## [**RM-R1 — Reward modeling as explicit reasoning**](https://github.com/RM-R1-UIUC/RM-R1)

RM-R1 trains a model to reason before assigning a reward. The practical slice distills released reward-reasoning trajectories into a 7B model. Research can change trajectory filtering, ordering, mixture, supervised-training hyperparameters, or add a small verified-reward stage within budget.

**Research target:** beat the released Distill-7B checkpoint on RewardBench and RM-Bench without exceeding the fixed training envelope.

# **RSI-8 Posttrain-RL**

# **RSI-8 Posttrain-RL**

Reinforcement learning with verifiable rewards (RLVR) trains a model from outcomes checked by rules, programs, or environments. The reward is cheap to score but easy to misuse: a model may exploit formatting, generate unnecessarily long answers, or overfit the training distribution.

Agentic RL adds tools and stateful environments. In those tasks, the interface and environment remain fixed so an improvement comes from learning.

## [**PRIME — Learning process rewards from policy rollouts**](https://github.com/PRIME-RL/PRIME)

PRIME updates an implicit process reward model alongside the policy, using more information than a final correct/incorrect signal. The research surface includes reward-model update frequency, weighting, difficulty-band filtering, process/result mixing, and RL hyperparameters. Sample efficiency must be compared at an equal rollout count.

**Research target:** improve the MATH-500 learning curve and final accuracy over both PRIME and outcome-reward baselines under the same rollout budget.

## [**Understand-R1-Zero and Dr. GRPO — Removing biased length incentives**](https://github.com/sail-sg/understand-r1-zero)

Dr. GRPO studies how group normalization and length normalization can unintentionally reward longer incorrect answers. Research can alter advantage estimation, normalization, sampling temperature, response limits, or the difficulty mixture. Accuracy and answer length must be reported together.

**Research target:** improve MATH-500 and AIME-24, or match their scores with fewer generated tokens and less length inflation, within the same training time.

## [**RAGEN and StarPO — Stable RL in interactive environments**](https://github.com/RAGEN-AI/RAGEN)

RAGEN trains agents in environments such as Sokoban, where vanilla StarPO can improve and then collapse into repetitive behavior. Researchers may change uncertainty filtering, clipping, KL control, rollout sampling, or other stability mechanisms while keeping the environment and step budget fixed.

**Research target:** increase final Sokoban success and prevent the late-training collapse across multiple seeds.

## [**Search-R1 — Learning when to retrieve information**](https://github.com/PeterGriffinJin/Search-R1)

Search-R1 trains a model to interleave reasoning with calls to a fixed retrieval system. The research surface includes query formation, retrieval penalties, number of search rounds, reward shaping, and the PPO/GRPO update. Retrieval infrastructure and evaluation corpora remain locked.

**Research target:** improve average exact match over seven held-out QA datasets at the same rollout budget, with retrieval calls and response tokens reported.

## [**Logic-RL — Curriculum design for verifiable logic problems**](https://github.com/Unakar/Logic-RL)

Logic-RL uses procedurally generated Knights-and-Knaves problems with exact verification. Researchers can change the mixture of difficulty levels, introduce harder generated instances, adjust format rewards, or tune GRPO. The controlled generator makes it possible to separate memorization from systematic generalization.

**Research target:** improve accuracy across all difficulty levels and transfer positively to held-out math reasoning benchmarks without extra training episodes.

## [**DAPO — Stabilizing large-scale reasoning RL**](https://github.com/BytedTsinghua-SIA/DAPO)

DAPO combines asymmetric clipping, dynamic sampling, token-level losses, and overlong-response shaping. A scaled 7B run makes these components testable within one day. Researchers may change their interaction, difficulty filtering, rollout policy, or hyperparameters while keeping the base model and training budget fixed.

**Research target:** exceed the local DAPO AIME-24 golden run while preventing entropy collapse and uncontrolled response growth.

# **RSI-9 Vision-Understanding**

# **RSI-9 Vision-Understanding**

## [**MAE — Learning images by reconstructing masked patches**](https://github.com/facebookresearch/mae)

Masked Autoencoders hide most image patches and train a Vision Transformer to reconstruct them. Research can change the mask ratio and pattern, decoder size, target normalization, augmentation, schedule, or data pipeline. The downstream probe remains fixed so a lower reconstruction loss cannot substitute for a more useful representation.

**Research target:** improve linear-probe or fine-tuning accuracy over the MAE golden run at the same pre-training epochs or GPU-hours.

## [**OpenCLIP — Open image–text representation learning**](https://github.com/mlfoundations/open_clip)

OpenCLIP provides a complete contrastive image–text training and evaluation stack. Researchers may change the loss, encoders, pooling, tokenization, resolution, data curriculum, regularization, or kernels. Because historical LAION streams cannot be reproduced exactly, the task uses a pinned open subset and a local ViT-B/32 golden run.

**Research target:** improve retrieval and the zero-shot evaluation aggregate at fixed data, FLOPs, and model size while maintaining stable training.

## [**I-JEPA — Predicting visual representations rather than pixels**](https://github.com/facebookresearch/ijepa)

I-JEPA masks target regions and predicts their latent representations from a larger context. The research surface includes context and target sampling, predictor and encoder design, teacher exponential moving average, masking, curriculum, and optimization. Loss, representation stability, and downstream probes should move together.

**Research target:** improve k-nearest-neighbor and linear evaluation over the local I-JEPA baseline under the same ImageNet and compute envelope.

## [**DeepOCR — Compressing documents into fewer visual tokens**](https://github.com/Princeton-AI2-Lab/DeepOCR)

DeepOCR offers an open training path for document optical compression. The decoder and public data remain fixed while the researcher changes the projector, 16× compressor, dynamic tiling, resolution policy, data proportions, or training hyperparameters. This makes OCR quality and visual-token count the two explicit axes.

**Research target:** improve the OmniDocBench and olmOCR-bench quality–token frontier over the released DeepOCR checkpoint without using unavailable OCR-2 components.

## [**DiT, SiT, and REPA — Aligning generative features with semantic representations**](https://github.com/sihyun-yu/REPA)

REPA adds a representation-alignment objective to diffusion or flow-transformer training. Researchers may change the target encoder, aligned layer, projection head, loss weight, or schedule while keeping the SiT-B/2 backbone, step budget, and sampler fixed. The task tests whether better internal features accelerate generative learning.

**Research target:** reduce FID over vanilla SiT and the released REPA recipe at 400K steps and matched wall-clock time.

## **What these examples have in common**

Representation quality must be measured outside the pre-training objective. Reconstruction loss, contrastive loss, and latent prediction loss can all improve without better transfer. Fixed probes and held-out domains keep the task focused on reusable visual information rather than objective-specific shortcuts.

# **RSI-10 Vision-GenerativeModeling**

# **RSI-10 Vision-GenerativeModeling**

Generative vision research balances output quality, training cost, and sampling cost. A method that lowers Fréchet Inception Distance (FID) with far more denoising steps has not moved the full frontier. Comparisons therefore lock data and training compute, standardize evaluation, and report quality together with the number of function evaluations, latency, or total sampling FLOPs.

## [**micro\_diffusion — Strong text-to-image training on a smaller budget**](https://github.com/SonyResearch/micro_diffusion)

micro\_diffusion studies how masking, patch mixing, data selection, resolution stages, and optimization can reduce the cost of text-to-image training. The GPU-time envelope is fixed, so improvements must come from a better recipe rather than a longer run. Both distributional and prompt-alignment metrics are required.

**Research target:** lower zero-shot COCO FID or improve GenEval over the reproduced recipe within the same end-to-end compute budget.

## [**VAR — Autoregressive generation from coarse to fine scales**](https://github.com/FoundationVision/VAR)

Visual Autoregressive Modeling predicts the next resolution scale rather than the next discrete token. With the d16 architecture and tokenizer fixed, research can change the learning-rate and weight-decay schedule, scale curriculum, loss weighting, augmentation, or progressive-training policy.

**Research target:** reach the local d16 FID in fewer steps, or obtain lower FID at the same epochs and wall-clock time while preserving Inception Score.

## [**Latent Diffusion — Generating in a compressed visual space**](https://github.com/CompVis/latent-diffusion)

Latent Diffusion separates perceptual compression from denoising, reducing the spatial cost of generation. Researchers may change the autoencoder, latent representation, denoiser, conditioning, noise path, training data, optimization, sampler, or guidance. Each gain must identify whether it came from training, compression, or more expensive sampling.

**Research target:** improve the quality–training-cost–sampling-cost frontier over a local open-data latent-diffusion golden run.

## [**Consistency Models — High-quality generation in one or a few steps**](https://github.com/openai/consistency_models)

Consistency Models learn outputs that agree along a probability-flow trajectory, enabling one-step or few-step sampling. Research can change the consistency objective, teacher–student distillation, time sampling, solver, architecture, curriculum, or joint training/sampling design. Results should be shown across several numbers of function evaluations.

**Research target:** improve FID at one and a few sampling steps without increasing the fixed ImageNet-64 training budget.

## [**MAR — Autoregressive generation with continuous visual tokens**](https://github.com/LTH14/mar)

MAR removes vector quantization and predicts continuous image patches with a diffusion loss. The research surface includes patch representation, DiffLoss, prediction order, masking schedule, grouping, conditioning, sampler, and optimization. The small ImageNet-256 setup provides a full training and FID loop.

**Research target:** improve FID, sampling cost, or stability over the local MAR \+ DiffLoss golden run at matched FLOPs.

