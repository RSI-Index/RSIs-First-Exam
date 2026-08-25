# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Candidate-owned positional-mechanism hooks.

The trusted pretraining entrypoint imports this module for both training and
verifier checkpoint reload. AutoResearch candidates may edit this file without
changing the locked launcher, optimizer, data path, or evaluators.
"""

from __future__ import annotations

from typing import Any


def install_runtime() -> None:
    """Install optional process-wide positional runtime behavior."""


def configure_model(model_cfg: Any) -> None:
    """Configure the baseline Llama-3-scaled RoPE model in place."""

