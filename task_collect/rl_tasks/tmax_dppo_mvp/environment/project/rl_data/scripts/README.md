# `rl_data/scripts/` — launcher scripts

Thin shell / Slurm launchers for every stage of the RL-data pipeline. The
Python code these scripts call lives under `rl_data/` (e.g.
`rl_data.generate_tasks`, `rl_data.generate_solutions`, `rl_data.analyze`,
`rl_data.comparison`). See [`../README.md`](../README.md) for the pipeline
overview.

## Layout

```
scripts/
├── generate_tasks/         # STAGE 1 — create tasks from the skill taxonomy
│   ├── run_generate_tasks.sh          # generic, env-overridable wrapper (legacy/sft_v2/rl_v2)
│   ├── run_generate_tasks_1k.sh       # legacy 1k preset
│   ├── run_generate_tasks_10k.sh      # legacy 10k preset
│   ├── run_generate_tasks_sft_v2_1k.sh # v2 SFT preset (CORPUS_KIND=sft_v2)
│   └── run_generate_tasks_rl_v2_5k.sh  # v2 RL preset (CORPUS_KIND=rl_v2)
│
├── generate_solutions/     # STAGE 2 — solve tasks with LLM agents (pass@k)
│   ├── run_generate_solutions.sh      # generic wrapper
│   ├── run_generate_solutions_1k_gemini.sh
│   ├── run_generate_solutions_10k_gemini.sh
│   ├── run_generate_solutions_skill_tax_*.sh   # corpus-specific presets (combined / vanillux / thinking)
│   └── launch_vllm.sh                 # spin up a local vLLM server (Qwen etc.) for solving
│
├── analyze/                # STAGE 3 — stats, plots, cost, and format conversions
│   ├── run_analyze.sh                 # pass@k + task-distribution report for a corpus
│   ├── estimate_cost.sh               # project API cost for a proposed run
│   ├── classify_difficulty.py         # bin tasks into Frontier/Advanced+/Advanced/Core tiers
│   ├── convert_to_harbor.py           # export tasks into Harbor-compatible layout
│   └── peak_context.py                # report peak context length across solutions
│
├── upload/                 # STAGE 4 — publish a corpus to the Hugging Face Hub
│   ├── upload_data_to_hf.sh
│   └── upload_data_to_hf_verified.sh  # skip tasks with 0 pass@k
│
├── combine/                # merge corpora (e.g. legacy + v2) into one symlinked root
│   └── combine_corpora.py             # `balanced` (SFT) and `union` (RL) modes
│
├── decontamination/        # n-gram overlap of a corpus vs eval benchmarks
│   └── run_decontamination.sh
│
├── repair/                 # re-materialise broken v2 fixtures via a SIF (host missing ffmpeg/gcc)
│   ├── run_repair_video_fixtures_in_sif.sh
│   └── run_repair_stripped_binary_in_sif.sh
│
├── comparison/             # head-to-head vs external terminal-task baselines
│   ├── run_comparison.sh              # one-shot pipeline: ingest -> classify -> solve -> compare
│   ├── run_ingest_*.sh                # pull + flatten a baseline (et, openthoughts, swe_smith, r2e_gym, …)
│   ├── run_classify_taxonomy.sh       # LLM-classify external tasks into OUR taxonomy
│   ├── run_generate_solutions_*.sh    # solve each baseline with the same model as ours
│   ├── run_local_qwen3_pass_at_8.sh   # local-model pass@8 helper
│   ├── _vllm_local.sh                 # shared local-vLLM helper
│   └── COMPARISON.md                  # full reference: modules, outputs, local-model usage, costs
│
└── predownload_model.sh    # pre-fetch a HF model into the cache (for offline compute nodes)
```

## Conventions

- Every script does `cd "$PROJECT_ROOT"` so it can be run from anywhere.
- Key parameters are env-overridable without editing files, e.g.
  `NUM_TASKS`, `OUT_DIR`, `TASKS_DIR`, `MODEL`, `CORPUS_KIND`, and the
  local-model env vars (`HOSTED_VLLM_API_BASE` / `OLLAMA_API_BASE` /
  `OPENAI_API_BASE`).
- Slurm-ready scripts include `#SBATCH` headers and can be launched with
  `sbatch`; bare `bash` also works for interactive runs.
- v2 task generation (`sft_v2` / `rl_v2`) needs
  `rl_data/containers/base_intricate.sif`; build it once on a build node
  (see [`../README.md`](../README.md)).
- See [`comparison/COMPARISON.md`](comparison/COMPARISON.md) for the comparison
  pipeline in detail.
