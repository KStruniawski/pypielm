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

from .bayesian import BayesianPIELM
from .constrained import (
    DELM,
    FPIELM,
    KAPIELM,
    LSEELM,
    PIELMRVDS,
    RINN,
    SGEPIELM,
    TSPIELM,
    XPIELM,
    EigPIELM,
    NormalEquationELM,
    NullSpacePIELM,
    ParameterRetentionELM,
    PiecewiseELM,
    RaNNPIELM,
    SoftPartitionKAPIELM,
    StefanPIELM,
)
from .curriculum import CurriculumPIELM
from .domain import DPIELM, DDELMCoarse, LocELM
from .fourier import GFFPIELM

# Gradient-based PINN baselines
from .pinn import (
    AdaptivePINN,
    FourierPINN,
    MuonPINN,
    ResidualAdaptivePINN,
    VanillaPINN,
)

# Registry helpers
from .registry import MODEL_REGISTRY, get_model, register

# Core PIELM variants
from .vanilla import CorePIELM, VanillaPIELM

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
