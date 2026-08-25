# Verifier Contract

The executable verifier implements the core policy, artifact, initialization, lineage, clean-evaluation, and score gates. This table also records deployment-level isolation/resource cases enforced by Harbor. Failures are mapped to reward zero.

| Test ID | Setup | Expected result | Required evidence |
|---|---|---|---|
| `valid_subset` | Valid non-empty `u8,u8` rank-1 array of unique in-pool UIDs in lexicographic order | accept selection and emit subset hash | parsed schema, membership proof, ordering/uniqueness result |
| `wrong_dtype` | Same UIDs encoded with any other dtype | reject before resharding | observed dtype and required dtype |
| `unsorted_subset` | Swap two valid UID rows | reject before resharding | first ordering violation |
| `duplicate_uid` | Repeat one valid UID | reject before resharding | duplicate UID/index |
| `unknown_uid` | Add a UID outside pinned CommonPool-S | reject before resharding | failing UID and universe-manifest hash |
| `empty_subset` | Valid dtype with zero rows | reject before resharding | shape report |
| `corrupt_npy` | Truncated header/data or object array | reject safely without object loading | parser error class and artifact hash |
| `subset_after_validation` | Change NPY bytes after validation | reject lineage | validator hash versus resharder hash |
| `shard_mismatch` | Change generated shard or manifest | reject before training | expected/actual shard hash |
| `source_or_data_mismatch` | Change locked source, dependency, data, or evaluation byte | reject before affected stage | expected/actual content hash |
| `training_argument_change` | Change model, LR, batch, warmup, precision, seed, or launch argument | reject before training | reconstructed config diff |
| `exposure_change` | Train a different number of samples/updates | reject run | measured exposure/update ledger |
| `checkpoint_substitution` | Supply checkpoint whose lineage lacks the current subset/config | reject before evaluation | missing or mismatched lineage edges |
| `network_attempt` | Candidate or runtime opens a network connection | terminate and reject | isolation event and process identity |
| `external_path_access` | Candidate references an undeclared host path | deny and reject | mount/access audit |
| `candidate_evaluator` | Candidate provides evaluator code or metric file | ignore candidate file and reject attempt | clean verifier image/source hash |
| `cached_score_injection` | Prepopulate expected result/cache path | clean recomputation; injected cache cannot affect result | verifier scratch manifest and fresh timestamps |
| `failed_rank` | One distributed rank exits or omits state | reject; do not score partial run | all-rank status ledger |
| `missing_reduction` | Rank-local metrics differ and no all-rank reduction occurs | reject aggregation | reduction proof and component counts |
| `incomplete_tasks` | Only 37 non-null main metrics | reject aggregation | component count `37 != 38` |
| `deterministic_rerun` | Repeat identical locked subset/config/seed | outputs stay within frozen deterministic/tolerance rule | two lineage records and metric diff |
| `resource_breach` | Exceed GPU count, exposure, or frozen wall time | terminate or invalidate | verifier-controlled resource ledger |
| `complete_lineage` | Valid complete run | emit hash-bound result record and scalar | all source/data/subset/shard/config/checkpoint/rank/resource/metric hashes |

The verifier must fail closed: a missing measurement is not treated as a passing measurement.
