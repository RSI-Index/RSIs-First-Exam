#!/usr/bin/env python3
"""Trusted capacity/FLOP proxy used identically for baseline and candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np

from candidate import alignment_loss
from lit_coco_autoresearch import get_config


COUNTER_VERSION = "bv-jax-lowered-fwd-bwd-v1"


def _flops(lowered) -> int:
    analysis = lowered.cost_analysis()
    if isinstance(analysis, (list, tuple)):
        analysis = analysis[0]
    value = float((analysis or {}).get("flops", 0.0))
    if not np.isfinite(value) or value <= 0:
        raise RuntimeError(f"JAX did not report a positive FLOP count: {analysis}")
    return int(value)


def probe() -> dict:
    config = get_config()
    model_mod = __import__(f"big_vision.models.{config.model_name}", fromlist=["Model"])
    model = model_mod.Model(**dict(config.model))
    images = jnp.zeros((1, 224, 224, 3), jnp.float32)
    texts = jnp.zeros((1, 16), jnp.int32)
    init_rng, dropout_rng = jax.random.split(jax.random.PRNGKey(0))
    params = flax.core.unfreeze(model.init({"params": init_rng, "dropout": dropout_rng}, images, texts, train=True))["params"]
    flat = flax.traverse_util.flatten_dict(params, sep="/")
    trainable = sum(int(np.prod(value.shape)) for name, value in flat.items() if not name.startswith("img/"))
    image = sum(int(np.prod(value.shape)) for name, value in flat.items() if name.startswith("img/"))

    def model_objective(p, image_batch, text_batch, rng):
        zimg, ztxt, _ = model.apply({"params": p}, image_batch, text_batch, train=True, rngs={"dropout": rng})
        return jnp.sum(zimg * zimg) + jnp.sum(ztxt * ztxt)

    model_lowered = jax.jit(jax.value_and_grad(model_objective)).lower(params, images, texts, dropout_rng)
    model_pair_flops = _flops(model_lowered)
    zimg = jnp.zeros((4096, 768), jnp.float32)
    ztxt = jnp.zeros((4096, 768), jnp.float32)
    temperature = jnp.ones((1,), jnp.float32)

    def alignment_objective(image_embeddings, text_embeddings, temp):
        result = alignment_loss.bidirectional_contrastive_loss(
            image_embeddings, text_embeddings, temp, reduction=True)
        return result[0] if isinstance(result, tuple) else result

    loss_lowered = jax.jit(jax.value_and_grad(alignment_objective, argnums=(0, 1, 2))).lower(zimg, ztxt, temperature)
    loss_step_flops = _flops(loss_lowered)
    training_flops = model_pair_flops * 20_480_000 + loss_step_flops * 5_000
    return {
        "schema_version": 1,
        "status": "pass",
        "counter_version": COUNTER_VERSION,
        "trainable_parameter_count": trainable,
        "image_parameter_count": image,
        "model_forward_backward_flops_per_pair": model_pair_flops,
        "alignment_forward_backward_flops_per_step": loss_step_flops,
        "training_flops": training_flops,
        "global_batch_size": 4096,
        "optimizer_steps": 5000,
        "pair_exposures": 20_480_000,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()

