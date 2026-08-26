# Validation ledger

Last updated: 2026-08-09.

## Current evidence

- CPU contract suite: 35 tests passed locally with `python3 -m unittest discover -s autolab/tasks/slime_search_r1_algorithm/tests -p 'test_*.py' -v`.
- Python syntax: run locally with `python3 -m compileall -q autolab/tasks/slime_search_r1_algorithm`.
- Shell syntax: run locally with `bash -n` over environment, verifier, and solution scripts.
- Whitespace/patch integrity: run locally with `git diff --check`.
- Harbor manifest: parsed successfully with repository-locked `harbor==0.3.0`.

## Release gates requiring the target environment

- `UNVERIFIED`: build the digest-pinned Docker image and confirm all Hugging Face snapshots and the 5 GB compressed wiki-18 corpus fit the configured storage.
- `UNVERIFIED`: start the Java/Pyserini CPU BM25 server and verify a known query returns exactly top-k 3 documents.
- `UNVERIFIED`: launch the root supervisor with the declared `SYS_ADMIN`/`NET_ADMIN` capabilities, confirm `unshare --net` and private-loopback startup succeed, and prove the ordinary agent namespace cannot reach the privileged Ray control plane.
- `UNVERIFIED`: run a short infrastructure-only Slime multi-GPU smoke to validate native HF-to-Megatron actor/reference loading, Ray/SGLang routing, built-in algorithm selection, exact checkpoint naming, and HF export.
- `UNVERIFIED`: run the unchanged full 1,000-round baseline on 8 H100s and record wall time, peak memory, checkpoint size, and aggregate score.
- `UNVERIFIED`: run the complete Harbor agent-to-root-verifier flow and confirm candidate/base/released-anchor evaluation plus numeric `reward.json`.

The task must not be marked released until every target-environment gate above has fresh evidence. The P0 implementation intentionally does not infer GPU or Docker success from static checks.
