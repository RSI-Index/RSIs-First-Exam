"""Candidate-visible deterministic parameterization hook surface."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TransferRecipe:
    base_lr: float = 3.011461785505323e-3

    def init_scale(self, role, shape, model, base):
        if role == "embedding":
            return {"distribution": "truncated_normal", "std": float(model.hidden_size ** -1)}
        if role == "normalization":
            return {"distribution": "constant", "value": 1.0}
        fan_in = getattr(shape, "fan_in", getattr(model, "hidden_size", 1152))
        return {"distribution": "truncated_normal", "std": float(fan_in ** -0.5)}

    def forward_scale(self, site, model, base):
        return 1.0

    def lr_scale(self, role, shape, model, base):
        return 1.0


RECIPE = TransferRecipe()


def build_recipe():
    return RECIPE
