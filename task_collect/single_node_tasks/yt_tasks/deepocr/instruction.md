# Optimize DeepOCR optical compression

Train and submit a DeepOCR checkpoint that improves document OCR quality versus
the measured number of visual tokens, while keeping the public DeepOCR decoder
lineage, training data, and evaluation protocol fixed. Training-code changes
matter only through the submitted checkpoint.

## Setup

| Item | Path / value |
|---|---|
| Writable workspace | `/app/project` |
| Source | `Princeton-AI2-Lab/DeepOCR@cd49a8f6129689cbda94c8cb25cfd128ab69b1f6` |
| Starting checkpoint | `pkulium/easy_deepocr@dce86e836ee10a4e5fb54a1d7e12dcba8a51d02a` |
| Vision checkpoint | `pkulium/sam_clip_ckpt@ff2becef5c02580c7571dc1ef07627df287bcbeb` |
| Stage-2 data | `allenai/olmOCR-mix-1025@d3c045014fac170a5af53d953a6d72071a023c60` through mixture `olmOCR-mix-pretrain` |
| Training entrypoint | `/app/project/scripts/NVILA-Lite/pretrain_ocr.sh` |
| Public inference | `/app/project/llava/eval/omini_doc_bench.py` |
| Final evaluator | pinned OmniDocBench code plus a trusted visual-token meter |
| Eval revisions | `OmniDocBench@2b161d010d2e3aff77a0edef359ea3a6411d23cd`; data `@aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec` |
| Offline assets | read-only HF cache at `/root/.cache/huggingface`; `HF_HUB_OFFLINE=1` |
| Submission | `/app/output/model` plus `/app/output/provenance.json` |
| Compute | Agent/training: 8 H100s, one six-hour trajectory; current upstream inference verifier is single-process/1-GPU inside that allocation |
| Network | Codex web search/external research disabled and HF offline; Apptainer does **not** provide hard egress isolation because model/policy-judge control traffic remains |
| Matched baseline status | `unmeasured`: the prior one-page checkpoint smoke is neither the full public baseline nor a matched 100-step training baseline |

The complete converted Stage-2 dataset must already exist under
`/datasets/deepocr/olmOCR-mix-1025`; easy_deepocr and `sam_clip_ckpt` are loaded
from pinned offline snapshots. A checkpoint-only evaluation has zero
optimization attempts. Missing assets may not be replaced with a different
dataset or undeclared OCR model.

Its host-side `manifest.json` must bind the source dataset/revision above,
DeepOCR conversion commit, `render_dpi=144`, raw/missing-text/post-filter counts
267962/5752/262210, the upstream converter's 219 reproducible PyMuPDF
`Overly large image` skips, 261991 final records/PNGs, and the SHA256 of
`transformed_data_png.json`. It also records `validated_png_count=261991` and a
path/size/dimension index SHA256 after proving that JSON image paths are unique,
match the PNG tree exactly, and every PNG has a valid signature/IHDR/nonzero
dimensions. The exact 219 skipped paths/reason classes are bound by a second
SHA256 and are re-run through the pinned upstream per-item converter during
preparation; they are disclosed upstream renderer failures, not successful
examples. The 5,752 initially dropped rows have missing
`natural_text`; upstream's intended falsy check misses pandas `NaN`, so the
host prep wrapper performs this fixed sanitation before calling the official
converter. Preflight also checks that the
JSON and converted PNG tree contain real data, not just directory names.

## What is actually being reproduced

This task uses the public Princeton DeepOCR reproduction: a SAM/CLIP-style
DeepEncoder and compressor in VILA with a Qwen2-family 7B decoder. The upstream
Stage-2 script tunes the projector and language model, freezes the vision tower,
uses dynamic image aspect ratios, bf16, global batch 64, learning rate `5e-5`,
cosine decay, 3% warmup, sequence length 4096, and one epoch over
`olmOCR-mix-pretrain`.

This is not the DeepSeek-OCR paper setup. DeepSeek-OCR uses DeepEncoder with a
DeepSeek-3B-MoE decoder and its full training recipe is not reproduced by this
repository. It is also not DeepSeek-OCR-2 and contains no causal-flow query
mechanism. Results may be described as DeepOCR-native optical-compression
research, not as reproduction of DeepSeek's closed training numbers.

Four infrastructure compatibility changes are present and disclosed:

- In `llava/model/multimodal_encoder/builder.py`, the optional
  `PS3VisionTower` import is lazy instead of unconditional. Upstream otherwise
  crashes on the DeepOCR SAM/CLIP path because the package does not declare the
  optional `ps3` dependency. The patch does not change the selected SAM/CLIP
  model path. On that SAM/CLIP branch only, the same file also honors the
  pinned `DEEP_OCR_VISION_TOWER` path so the published checkpoint's embedded
  author-private `/lustre` path resolves to the verified offline weight.
- `olmOCR-mix-pretrain` registry paths point to the read-only `/datasets`
  mount instead of the authors' private `/lustre/...` directory;
- `pretrain_ocr.sh` points to the pinned offline `sam_clip_ckpt` snapshot,
  with `DEEP_OCR_VISION_TOWER` as an auditable override.
- `scripts/setups/train.sh` keeps its Slurm behavior when Slurm is present but
  otherwise respects the explicit single-node LSF values; this task fixes
  `NNODES=1`, `GPUS_PER_NODE=8`, rank 0, and localhost rendezvous.

The path changes only alter file resolution, and the scheduler patch only
selects the declared eight visible GPUs. With global batch 64 and accumulation
1 this derives per-device batch 8, instead of the script's non-Slurm 4-GPU
fallback/per-device 16. Model/data content, global batch, and the remaining
upstream training arguments stay unchanged; 8 GPUs come from this task report,
not a DeepOCR paper hardware requirement.

## Before training

The environment has already replaced the authors' absolute `/lustre` paths and
Slurm-only launcher assumptions as described above. Do not make a second path
or topology substitution. Record the resolved vision file, dataset manifest,
eight-GPU topology, and SHA256 in provenance.

The baseline-shaped command is:

```bash
cd /app/project
BASE=/root/.cache/huggingface/hub/models--pkulium--easy_deepocr/snapshots/dce86e836ee10a4e5fb54a1d7e12dcba8a51d02a
bash scripts/NVILA-Lite/pretrain_ocr.sh \
  "$BASE" \
  olmOCR-mix-pretrain \
  /app/output/runs/deepocr-stage2
```

The script defaults relevant Stage-2 settings to global batch 64, gradient
accumulation 1, projector+LLM trainable, vision tower frozen, dynamic image
aspect ratio, bf16, one epoch, save every 100 steps, LR `5e-5`, weight decay 0,
3% warmup, cosine schedule, and maximum text length 4096. Any departure must be
recorded as an experimental variable.

## Baseline and iteration loop

1. Run the full public `easy_deepocr` checkpoint baseline through the same
   inference, token meter, and evaluator used for candidates.
2. For the report's qualification lane, one optimization attempt is exactly
   100 Stage-2 optimizer updates followed by scoring. Baseline and candidate
   must use identical examples, order, seed, global batch, and update count.
3. Treat a 100-step result as screening evidence only. A final training claim
   must rerun the chosen method under a matched full-epoch budget and multiple
   seeds or explicitly state that it did not.
4. Change an interpretable group: projector/compressor, compression ratio,
   dynamic tiling/resolution, sampling/weighting within the fixed Stage-2
   records, or Stage-2 hyperparameters.
5. Record per-slice OCR metrics and visual-token distribution, not only the
   scalar reward.

Public inference can be run with:

```bash
python /app/project/llava/eval/omini_doc_bench.py \
  --model-path /app/output/model \
  --input-folder /root/.cache/huggingface/hub/datasets--opendatalab--OmniDocBench/snapshots/aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec/images \
  --output-folder /app/output/omnidoc_predictions \
  --text "Free OCR."
```

## Final evaluation and reward

The current verifier uses all staged public OmniDocBench pages. It runs the
pinned upstream inference path, measures the actual inserted visual-token
sequence, and computes:

- text-block edit distance `D_text` (lower is better);
- formula edit distance `D_formula` (lower is better);
- table TEDS `T_table` (higher is better);
- reading-order edit distance `D_order` (lower is better);
- average measured visual tokens per page `N`.

After the Reward-Integrity Gate:

```text
Q = ((1-D_text) + (1-D_formula) + T_table + (1-D_order)) / 4
reward = clip(Q / sqrt(max(1, N/250)), 0, 1)
```

The equal weights, square-root penalty, and 250-token anchor are environment
choices, not values from DeepOCR or DeepSeek papers. The current verifier also
does not implement the report's hidden-document split or olmOCR-bench hard
gate. Scientific reports must include all raw components and must not call this
scalar publication-grade evidence by itself.

## Editable and locked scope

You may edit the DeepOCR projector/compressor, dynamic tiling/resolution policy,
sampling/weighting within the declared Stage-2 records, and training
hyperparameters. Keep the Qwen2-family 7B decoder skeleton, declared
datasets/revisions and records, and final inference semantics.
Do not train/calibrate on OmniDocBench or olmOCR-bench examples; modify the
evaluator or token meter; hardcode pages, filenames, hashes, or predictions;
add OCR-2 causal-flow queries; or use an undeclared finished checkpoint.

## Required outputs and experiment record

- `/app/output/model`: a complete clean-loader-compatible Hugging Face
  safetensors checkpoint, not only a local training shard.
- `/app/output/provenance.json`: base/component models and revisions, model
  config, data revision/mixture, measured training examples/tokens and optimizer
  updates, global batch, command, upstream commit, downloads, evaluation
  commands, measured `average_visual_tokens`, `optimization_attempts`, and
  `web_search: "disabled"`.
- `/app/output/experiments.jsonl`: one row per matched train-and-score decision
  cycle with hypothesis, change, command, GPU count, optimizer steps, data/seed,
  timestamps, status, and raw metrics.
