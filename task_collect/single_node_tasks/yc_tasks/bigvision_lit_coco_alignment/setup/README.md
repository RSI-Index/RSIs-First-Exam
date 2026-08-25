# Official asset bootstrap

COCO Captions and all Big Vision initializers/checkpoints have official automatic sources. ImageNet requires an account and license acceptance, so obtain these two official archives and place them in one directory:

- `ILSVRC2012_img_train.tar`
- `ILSVRC2012_img_val.tar`

Then run from the copied task directory:

```bash
./setup/prepare_assets.sh --dry-run
IMAGENET_MANUAL_DIR=/cluster/licensed/imagenet ./setup/prepare_assets.sh
```

The setup container downloads public objects from the exact historical Google Cloud Storage paths, checks their recorded sizes and MD5 digests, asks the pinned TFDS builders to prepare COCO Captions and ImageNet, hashes the resulting tree, and emits `.assets/initializers/harbor_manifest.json` plus `.assets/contracts/baseline_contract.json`.

Only setup uses the network. Candidate, trainer, and verifier remain offline. Set `BIGVISION_ASSETS_ROOT=/cluster/path` to keep the generated assets on cluster storage; `run_harbor.sh` resolves that path automatically. No unofficial ImageNet mirror is used.
