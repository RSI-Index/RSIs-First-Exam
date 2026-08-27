# single_node_tasks — 1 node task

Tasks filtered out of all 25 tasks in `task_collect` that run on a **single node (1 node)**,
determined from the GPU count / topology fields in each task's `task.toml` (RSI-style tasks
also cross-checked against `contract.yaml`). Original task directories are left untouched under
their respective `*_tasks` folders; these are copies, organized by their original family
subfolder.

## Explicitly labeled "single node" (high confidence)

| Path | Evidence |
|---|---|
| `chenhao_tasks/openclip_datacomp_v2_hardened_baseline` | `topology_mapping = "single node x 8 H100"` |
| `hh_tasks/vlmr1-rec-self-curriculum` | scope states "the 8-GPU topology" |
| `yt_tasks/deepocr` | `NNODES = "1"`, `GPUS_PER_NODE = "8"` |
| `zx&qj_tasks/molmo2-video-pointing` | `topology_mapping = "single-node 8-GPU torchrun"` |

## GPU count ≤8, no cross-node topology mentioned (inferred single node)

| Path | Evidence |
|---|---|
| `fq_tasks/magpie_data_pipeline` | `gpus = 8`, no node/topology field |
| `fq_tasks/slime_search_r1_algorithm` | `actor_gpus=4 + rollout_gpus=4 = 8` |
| `hh_tasks/datacomp-small-self-filtering` | no gpu/node field anywhere; DataComp small scale |
| `hh_tasks/nanovlm-finevision-self-selection` | `world_size=8`; contract notes "fits comfortably inside one node-hour" |
| `yc_tasks/bigvision_lit_coco_alignment` | `world_size = 8`, `gpus = 8` |
| `yc_tasks/datacomp_s_filter_discovery` | `world_size = 4`, `gpus = 4` |
| `yt_tasks/deepspec` | `full_run_gpus = 4` |
| `yt_tasks/rlm` | `gpus = 8` (two TP4 workers, 8 GPUs total) |
| `zy_tasks/concurrent-agent-serving-optimizer-qwen36-27b` | fixed backend runs on the same machine's "physical GPUs 6 and 7" (2xH100) |
| `zy_tasks/gemm-h100-kernel-lab` | `gpus = 1` |

## Excluded

- `yt_tasks/superbpe`: pure CPU tokenizer training, `full_run_gpus = 0`; the upstream repo has
  no accelerator topology at all, so "node" doesn't strictly apply — not included here.
- The remaining 10 tasks are explicitly multi-node and excluded: `dinov3_imagenet_semdense`
  (4x8), `tmax_dppo_mvp` (8x8), `yt_tasks/datacomp` (2x8), `gated_deltanet_autoresearch_run`
  (4x8), `olmo3-7b-zero-rl-math` (72 GPUs, disaggregated), `zf_tasks/post-training/search_rl`
  (32 GPUs/experiment, 256 total), and the 4 `zf_tasks` pre-training tasks (each 32 nodes x 8
  GPUs = 256).
