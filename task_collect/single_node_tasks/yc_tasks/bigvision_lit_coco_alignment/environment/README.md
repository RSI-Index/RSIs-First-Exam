# Harbor Environment

The image checks out Big Vision `8921d5141504390a8a4f7b2dacb3b3c042237290`, verifies the locked tree and core files, installs the explicit H100-oriented dependency lock, and exposes the candidate workspace at `/app/project`. Immutable adapters live under `/task-tools` and `/opt/big_vision`.

```text
candidate three-file package -> closed config + policy/capacity checks
  -> fixed eight-GPU historical contrastive trainer
  -> numeric-only model/config + frozen-image/provenance records
  -> separate clean COCO/ImageNet verifier
  -> retention gate -> bidirectional R@1 geometric-mean reward
```

`validate` parses and fingerprints the candidate. `full` additionally requires the matching asset manifest, a truthful experiments ledger, capacity compliance, a fresh work directory, exact final image-parameter identity, and the complete `repository_anchored_demo_v1` baseline contract.

Candidate training sees only COCO train plus declared initializers and vocabulary. The separate verifier receives final COCO/ImageNet assets. Candidate, trainer, and verifier networking is disabled; the task-owned official acquisition phase is described in `../setup/README.md` and `../../OPERATOR_STAGING.md`.
