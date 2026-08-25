# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Trusted adapter from the root transfer recipe to the pinned AdamH runtime."""

from __future__ import annotations

import importlib.util
import math
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch
import marin_adamh
from marin_adamh import build_megatron_optimizer as build_adamh_optimizer
from marin_adamh import parameter_layout


_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_RECIPE_PATH = _PROJECT_ROOT / "parameterization_transfer.py"
_BASE_MODEL = SimpleNamespace(layers=12, hidden_size=1152, ffn_hidden_size=4608)
_RECIPE: Any | None = None
_STEP_PATCHED = False


def _positive(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive, got {result}")
    return result


def _model_shape() -> SimpleNamespace:
    return SimpleNamespace(
        layers=int(os.environ["MODEL_NUM_LAYERS"]),
        hidden_size=int(os.environ["MODEL_HIDDEN_SIZE"]),
        ffn_hidden_size=int(os.environ["MODEL_FFN_HIDDEN_SIZE"]),
    )


def _tensor_shape(tensor: torch.Tensor) -> SimpleNamespace:
    dimensions = tuple(int(value) for value in tensor.shape)
    fan_in = dimensions[-1] if dimensions else 1
    fan_out = dimensions[0] if len(dimensions) > 1 else fan_in
    return SimpleNamespace(dimensions=dimensions, fan_in=fan_in, fan_out=fan_out)


def _load_recipe() -> Any:
    global _RECIPE
    if _RECIPE is not None:
        return _RECIPE
    module_name = "parameterization_transfer_candidate_recipe"
    spec = importlib.util.spec_from_file_location(module_name, _RECIPE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load candidate recipe: {_RECIPE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    recipe = module.build_recipe()
    _positive(getattr(recipe, "base_lr", None), "recipe.base_lr")
    for method in ("init_scale", "forward_scale", "lr_scale"):
        if not callable(getattr(recipe, method, None)):
            raise TypeError(f"candidate recipe must define {method}()")
    _RECIPE = recipe
    return recipe


def _init_spec(role: str, tensor: torch.Tensor) -> dict[str, Any]:
    recipe = _load_recipe()
    raw = recipe.init_scale(role, _tensor_shape(tensor), _model_shape(), _BASE_MODEL)
    if isinstance(raw, dict):
        result = dict(raw)
    else:
        result = {"std": raw}
    distribution = str(result.get("distribution", "truncated_normal"))
    if distribution == "constant":
        value = float(result.get("value", 1.0))
        if not math.isfinite(value):
            raise ValueError(f"{role} initializer constant must be finite")
        return {"distribution": distribution, "value": value}
    result["std"] = _positive(result.get("std"), f"{role} initializer std")
    if distribution not in {"truncated_normal", "normal", "uniform"}:
        raise ValueError(f"unsupported {role} initializer distribution: {distribution}")
    result["distribution"] = distribution
    return result


def _apply_initializer(tensor: torch.Tensor, role: str) -> None:
    spec = _init_spec(role, tensor)
    distribution = spec["distribution"]
    if distribution == "constant":
        torch.nn.init.constant_(tensor, float(spec["value"]))
        return
    std = float(spec["std"])
    if distribution == "truncated_normal":
        truncation = _positive(spec.get("truncation", 3.0), f"{role} truncation")
        torch.nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-truncation * std, b=truncation * std)
    elif distribution == "normal":
        torch.nn.init.normal_(tensor, mean=0.0, std=std)
    else:
        bound = math.sqrt(3.0) * std
        torch.nn.init.uniform_(tensor, -bound, bound)


def initializer_methods() -> tuple[Callable[[torch.Tensor], None], Callable[[torch.Tensor], None], Callable[[torch.Tensor], None]]:
    """Return trusted callbacks for hidden, readout, and embedding initialization."""

    return (
        lambda tensor: _apply_initializer(tensor, "hidden_matrix"),
        lambda tensor: _apply_initializer(tensor, "readout"),
        lambda tensor: _apply_initializer(tensor, "embedding"),
    )


def _matrix_role(name: str, layout: str) -> str:
    lowered = name.lower()
    if lowered.endswith("linear_qkv.weight"):
        return "mha_qkv"
    if lowered.endswith("linear_proj.weight"):
        return "attention_output"
    if lowered.endswith("linear_fc1.weight"):
        return "mlp_gate_up"
    if lowered.endswith("linear_fc2.weight"):
        return "mlp_output"
    if "output_layer" in lowered:
        return "readout"
    return layout if layout != "matrix" else "hidden_matrix"


def _scale_output(value: Any, multiplier: float) -> Any:
    if torch.is_tensor(value):
        return value * multiplier
    if isinstance(value, tuple):
        return tuple(_scale_output(item, multiplier) for item in value)
    if isinstance(value, list):
        return [_scale_output(item, multiplier) for item in value]
    return value


def _forward_site(name: str) -> str | None:
    lowered = name.lower()
    if lowered.endswith("self_attention.linear_proj"):
        return "attention_output"
    if lowered.endswith("mlp.linear_fc2"):
        return "mlp_output"
    if lowered.endswith("output_layer"):
        return "readout"
    if lowered.endswith("embedding"):
        return "embedding"
    if re.search(r"(?:^|\.)decoder\.layers\.\d+$", lowered):
        return "residual"
    return None


def _install_model_recipe(model_chunks: Any) -> list[float]:
    recipe = _load_recipe()
    model = _model_shape()
    seen: set[int] = set()
    adamh_lr_scales: list[float] = []
    for model_chunk in model_chunks:
        for name, parameter in model_chunk.named_parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            algorithm, layout = parameter_layout(name, parameter)
            if algorithm == "adamh":
                role = _matrix_role(name, layout)
                multiplier = _positive(
                    recipe.lr_scale(role, _tensor_shape(parameter), model, _BASE_MODEL),
                    f"{role} LR multiplier",
                )
                setattr(parameter, "_parameterization_lr_scale", multiplier)
                adamh_lr_scales.append(multiplier)
            elif parameter.ndim == 1 and ("norm" in name.lower() or "layernorm" in name.lower()):
                _apply_initializer(parameter, "normalization")
        for name, module in model_chunk.named_modules():
            site = _forward_site(name)
            if site is None or getattr(module, "_parameterization_hook_installed", False):
                continue
            multiplier = _positive(
                recipe.forward_scale(site, model, _BASE_MODEL),
                f"{site} forward multiplier",
            )
            if multiplier != 1.0:
                module.register_forward_hook(
                    lambda _module, _inputs, output, scale=multiplier: _scale_output(output, scale)
                )
            setattr(module, "_parameterization_hook_installed", True)
    return adamh_lr_scales


def _install_scaled_adamh_step() -> None:
    global _STEP_PATCHED
    if _STEP_PATCHED:
        return

    @torch.no_grad()
    def step(self: marin_adamh.MarinAdamH, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            self._clip_group(group["params"])
        for group in self.param_groups:
            group_lr = float(group["lr"])
            algorithm = group["algorithm"]
            layouts = group.get("logical_layouts", ["vector_or_embedding"] * len(group["params"]))
            lr_scales = group.get("parameterization_lr_scales", [1.0] * len(group["params"]))
            for parameter, layout, lr_scale in zip(group["params"], layouts, lr_scales):
                if parameter.grad is None:
                    continue
                direction = self._adam_direction(parameter, self.state[parameter])
                if algorithm == "adam":
                    parameter.add_(direction, alpha=-group_lr)
                    continue
                learning_rate = group_lr * float(lr_scale)
                parameter_views = marin_adamh._logical_matrices(parameter, layout, self.tensor_parallel_size)
                direction_views = marin_adamh._logical_matrices(direction, layout, self.tensor_parallel_size)
                for parameter_view, direction_view in zip(parameter_views, direction_views):
                    self._project(parameter_view, direction_view, learning_rate)
        return loss

    marin_adamh.MarinAdamH.step = step
    _STEP_PATCHED = True


def install_runtime() -> None:
    """Install the scale-general AdamH LR-multiplier hook."""
    recipe = _load_recipe()
    marin_adamh.ADAMH_LR = _positive(recipe.base_lr, "recipe.base_lr")
    _install_scaled_adamh_step()


def build_megatron_optimizer(config: Any, model_chunks: Any, **kwargs: Any) -> Any:
    """Materialize the frozen recipe and construct the locked AdamH/Adam split."""
    install_runtime()
    adamh_lr_scales = _install_model_recipe(model_chunks)
    optimizer = build_adamh_optimizer(config, model_chunks, **kwargs)
    raw_optimizer = getattr(optimizer, "optimizer", None)
    if not isinstance(raw_optimizer, marin_adamh.MarinAdamH):
        raise TypeError("locked AdamH wrapper did not expose its raw optimizer")
    adamh_group = next(group for group in raw_optimizer.param_groups if group["algorithm"] == "adamh")
    if len(adamh_group["params"]) != len(adamh_lr_scales):
        raise RuntimeError("AdamH parameter/LR-multiplier materialization length mismatch")
    adamh_group["parameterization_lr_scales"] = adamh_lr_scales
    return optimizer


__all__ = ["build_megatron_optimizer", "initializer_methods", "install_runtime"]
