# Official asset bootstrap

Run from the copied task directory:

```bash
./setup/prepare_assets.sh --dry-run
./setup/prepare_assets.sh
```

The setup container is the only networked phase. It checks out the pinned official DataComp commit, runs its `download_upstream.py` and `download_evalsets.py` entrypoints, downloads the pinned official released control, normalizes the output under `.assets/`, builds the successful-UID universe, hashes every runtime/verifier asset, and emits `.assets/contracts/baseline_contract.json`.

This is a very large acquisition: CommonPool-S materialization contacts the URLs listed by the official dataset and therefore requires substantial disk, bandwidth, file descriptors, and time. The candidate, trainer, and verifier do not inherit setup networking.

`DATACOMP_ASSETS_ROOT=/cluster/path` may place the generated asset tree on cluster storage. The path is resolved by `run_harbor.sh`; no source edit is required.
