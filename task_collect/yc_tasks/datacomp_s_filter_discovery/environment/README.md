# Harbor Environment

The image builds the two pinned upstream repositories, verifies their commits and trees, applies only the immutable initialization-attestation patch, and installs the locked runtime. The candidate workspace is `/app/project`; immutable code is under `/task-tools`, `/opt/datacomp`, `/opt/open_clip`, and `/opt/contracts`.

```text
read-only metadata/features -> candidate selector twice -> UID validation
  -> pinned resharder -> fixed four-rank learner -> checkpoint + lineage
  -> separate verifier -> 38-component mean reward
```

The candidate container mounts CommonPool-S read-only and never mounts final evaluation assets. `selector` validates and fingerprints a candidate; `full` additionally requires the matching asset manifest, a truthful experiments ledger, a fresh training directory, and the complete `repository_anchored_demo_v1` baseline contract.

The separate verifier uses a pristine candidate-tree reference and verifier-only evaluation mount. It never trusts a candidate-reported score. Candidate, trainer, and verifier networking is disabled; the task-owned official acquisition phase is described in `../setup/README.md` and `../../OPERATOR_STAGING.md`.
