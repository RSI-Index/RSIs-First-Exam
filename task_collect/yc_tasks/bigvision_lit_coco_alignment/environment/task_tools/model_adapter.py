"""Immutable bridge from Big Vision's module loader to the candidate tower."""

from candidate.text_tower import Model as Model
from candidate.text_tower import load as load

__all__ = ["Model", "load"]

