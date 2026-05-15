"""Core abstractions: BasePIELM, feature maps, and linear solvers.

Implemented across Steps 2 (core) and 5 (models).
"""

from __future__ import annotations

from .base import BasePIELM, PIELMConfig
from .feature_maps import AutogradFeatureMap, FourierFeatureMap, RandomFeatureMap
from .solver import BayesianSolveResult, bayesian_solve, ridge_solve, rrqr_solve, tikhonov_solve

__all__ = [
    # Base
    "BasePIELM",
    "PIELMConfig",
    # Feature maps
    "RandomFeatureMap",
    "FourierFeatureMap",
    "AutogradFeatureMap",
    # Solvers
    "ridge_solve",
    "rrqr_solve",
    "bayesian_solve",
    "tikhonov_solve",
    "BayesianSolveResult",
]
