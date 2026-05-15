"""Model library: all PIELM variants, PINN baselines, and the model registry.

Public surface::

    from pypielm.models import (
        CorePIELM, VanillaPIELM, BayesianPIELM, GFFPIELM,
        DPIELM, LocELM, DDELMCoarse,
        CurriculumPIELM,
        NullSpacePIELM, EigPIELM, LSEELM, StefanPIELM,
        NormalEquationELM, ParameterRetentionELM, PiecewiseELM, DELM,
        FPIELM, SGEPIELM, RINN, RaNNPIELM, XPIELM,
        PIELMRVDS, TSPIELM, KAPIELM, SoftPartitionKAPIELM,
        VanillaPINN, AdaptivePINN, FourierPINN, MuonPINN,
        ResidualAdaptivePINN,
        get_model, MODEL_REGISTRY,
    )
"""

from __future__ import annotations

# Core PIELM variants
from .vanilla import CorePIELM, VanillaPIELM
from .bayesian import BayesianPIELM
from .fourier import GFFPIELM
from .domain import DPIELM, LocELM, DDELMCoarse
from .curriculum import CurriculumPIELM
from .constrained import (
    NullSpacePIELM,
    EigPIELM,
    LSEELM,
    StefanPIELM,
    NormalEquationELM,
    ParameterRetentionELM,
    PiecewiseELM,
    DELM,
    FPIELM,
    SGEPIELM,
    RINN,
    RaNNPIELM,
    XPIELM,
    PIELMRVDS,
    TSPIELM,
    KAPIELM,
    SoftPartitionKAPIELM,
)

# Gradient-based PINN baselines
from .pinn import (
    VanillaPINN,
    AdaptivePINN,
    FourierPINN,
    MuonPINN,
    ResidualAdaptivePINN,
)

# Registry helpers
from .registry import MODEL_REGISTRY, get_model, register

__all__ = [
    # PIELM
    "VanillaPIELM",
    "CorePIELM",
    "BayesianPIELM",
    "GFFPIELM",
    "DPIELM",
    "LocELM",
    "DDELMCoarse",
    "CurriculumPIELM",
    "NullSpacePIELM",
    "EigPIELM",
    "LSEELM",
    "StefanPIELM",
    "NormalEquationELM",
    "ParameterRetentionELM",
    "PiecewiseELM",
    "DELM",
    "FPIELM",
    "SGEPIELM",
    "RINN",
    "RaNNPIELM",
    "XPIELM",
    "PIELMRVDS",
    "TSPIELM",
    "KAPIELM",
    "SoftPartitionKAPIELM",
    # PINN
    "VanillaPINN",
    "AdaptivePINN",
    "FourierPINN",
    "MuonPINN",
    "ResidualAdaptivePINN",
    # Registry
    "get_model",
    "register",
    "MODEL_REGISTRY",
]
