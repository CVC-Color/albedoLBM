"""Albedo Estimation via Latent Bridge Matching (CIC 2026)."""

from .inference import evaluate, get_model
from .pipeline import IntrinsicDecomposer

__all__ = ["IntrinsicDecomposer", "evaluate", "get_model"]
__version__ = "1.0.0"
