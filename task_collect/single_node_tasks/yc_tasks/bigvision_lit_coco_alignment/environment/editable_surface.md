# Editable Surface

Only three files are editable:

- `candidate/text_tower.py`
- `candidate/alignment_loss.py`
- `candidate/config.toml`

The Harbor starter files live under `environment/project/candidate/`. The fixed adapters enforce their Python interfaces and closed TOML schema. The demo profile seals a `111,000,000` trainable-parameter ceiling and `2.0e18` training-FLOP ceiling.

## Expected interfaces

`text_tower.py` exports Big Vision-compatible `Model` and `load` symbols and maps harness-tokenized integer IDs to one text embedding per caption. The immutable adapter fixes token length, vocabulary identity, input dtypes/shapes, output dimension, and training/evaluation calls.

`alignment_loss.py` exports `bidirectional_contrastive_loss(zimg, ztxt, t, mask=None, reduction=False)` and returns the scalar/vector loss plus diagnostics expected by the pinned trainer. It cannot load data, mutate the image tower, change collectives, write checkpoints, or calculate official retrieval scores.

`config.toml` selects only allowlisted text/alignment options. The implemented harness parses a closed schema, rejects unknown keys, resolves defaults, and writes a canonical sorted JSON representation to `artifacts/config.json`. Candidate code cannot provide executable configuration objects.

## Allowed research families

- **text architecture:** replace or restructure the BERT-base text side within fixed capacity and the compatible text-output interface;
- **pooling:** change how valid token states become one caption representation;
- **projection:** change trainable mapping, normalization, or bottleneck geometry before alignment;
- **token interaction:** alter contextualization or controlled token-level interaction on the text side;
- **alignment loss:** change the declared symmetric cross-modal training objective without changing evaluation;
- **negative handling:** reweight, mine, debias, or mask negatives using only the fixed global batch.

Pooling and projection belong to one hypothesis family in the experiment design; their interface knobs are listed separately here. Candidates may replace BERT-base architecture, but trainable parameters and training FLOPs must remain within the sealed ceilings and verifier-compatible text-output contract.

## Required artifacts

- `artifacts/model.npz`: NumPy NPZ containing numeric arrays only; no object dtype, pickle, executable metadata, paths, cached metrics, or evaluator state. The artifact verifier enforces allowlisted prefixes, shapes, dtypes, finite-value rules, and image-parameter identity.
- `artifacts/config.json`: UTF-8 canonical JSON matching the frozen schema and the actual resolved training configuration.

Both artifact hashes are bound to candidate source, fixed configuration, data/initializer/source versions, counters, and run ledger. The clean verifier never trusts candidate-reported scores.

## Outside the editable surface

Image architecture/initializer/parameters/preprocessing; tokenization resources; data/splits; outer optimizer/schedule; batch/steps/exposure; gradient/distributed behavior; seeds; capacity counters; checkpoint writer; evaluator; metric/gate; mounts/network; accelerators/wall time; and lineage enforcement are immutable harness responsibilities.
