# Verifier Contract

The executable verifier implements the core policy, artifact, lineage, capacity, clean-evaluation, retention, and score gates. This table also records deployment-level isolation/resource cases enforced by Harbor or calibrated with the genuine baseline. Every rejection produces structured evidence and reward zero.

| Test ID | Setup | Expected result | Required evidence |
|---|---|---|---|
| `valid_numeric_artifacts` | Complete allowlisted numeric NPZ plus canonical schema-valid JSON | accept artifact boundary and continue | parsed keys/shapes/dtypes, finite check, hashes |
| `arbitrary_object` | NPZ/object/pickle or executable configuration object | reject safely without object loading | parser class, object marker, artifact hash |
| `missing_array` | Remove one required model array | reject before reconstruction | missing allowlisted key |
| `wrong_shape_or_dtype` | Change required array shape or dtype | reject before evaluation | expected/actual schema |
| `nan_or_inf` | Insert non-finite parameter | reject before evaluation | key/index and finite-check report |
| `incompatible_config` | Unknown key, invalid value, or config inconsistent with arrays | reject | schema errors and config/artifact hashes |
| `image_parameter_mutation` | Change one `img/.*` byte or image configuration | reject | initialized/final byte hashes and first mismatch |
| `parameter_ceiling_breach` | Candidate exceeds frozen trainable-parameter ceiling | reject before training or scoring | versioned counter result and ceiling |
| `flop_ceiling_breach` | Candidate exceeds frozen training-FLOP ceiling | reject before scoring | versioned counter result and ceiling |
| `data_or_preprocess_change` | Change COCO bytes/order policy, tokenizer, or image preprocessing | reject before training | expected/actual manifest/config diff |
| `exposure_change` | Change global batch, step count, or 20.48M exposure | reject run | measured batch/update/exposure ledger |
| `optimizer_or_schedule_change` | Change optimizer, LR, WD, warmup, schedule, clipping, or precision | reject before training | reconstructed locked-config diff |
| `hash_mismatch` | Alter source, dependency, data, initializer, config, checkpoint, evaluator, or result byte | reject at affected stage | expected/actual hash and lineage node |
| `runtime_network` | Any candidate/trainer/verifier process opens a network connection | terminate and reject | process identity and isolation event |
| `final_asset_access` | Candidate reads final COCO/ImageNet asset, label, template, or seed | deny and reject | mount/access audit |
| `external_path_access` | Candidate reads an undeclared host path | deny and reject | attempted path and process identity |
| `candidate_evaluator` | Candidate supplies evaluator/metric code | reject attempt; clean verifier remains authoritative | candidate artifact list and verifier image hash |
| `cached_score_injection` | Prepopulate score or evaluator cache | injected value cannot affect clean recomputation; reject attempt | clean scratch manifest and fresh result lineage |
| `missing_r1_direction` | Only I2T or T2I R@1 is produced | reject aggregation | component presence report |
| `missing_retention` | ImageNet retention metric or gate decision is absent | reject | retention evidence report |
| `failed_rank` | One distributed rank fails or omits state | reject without partial score | all-rank status ledger |
| `missing_cross_rank_reduction` | Rank-local embeddings/metrics are not globally reduced | reject | collective/reduction proof |
| `deterministic_rerun` | Repeat identical source/data/config/seed | satisfy frozen deterministic or measured-tolerance rule | paired lineage records and numeric diff |
| `lineage_binding` | Substitute checkpoint/config from another candidate run | reject before evaluation | broken lineage edge and hashes |
| `wall_time_breach` | Complete run exceeds frozen limit | terminate or invalidate | verifier-controlled monotonic timing ledger |
| `accelerator_breach` | Request/use more than frozen GPU topology | deny or invalidate | scheduler allocation and device inventory |
| `valid_complete_run` | Valid artifacts, both retrieval directions, retention pass, all gates and lineage complete | emit hash-bound scalar record | full lineage, diagnostics, gate decisions, resource ledger |

The verifier fails closed: a missing measurement, rank, hash, counter, directional metric, or gate is not interpreted as passing.
