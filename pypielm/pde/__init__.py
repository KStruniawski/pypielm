"""PDE building blocks: operators, collocation, and constraints.

Public surface::

    from pypielm.pde import (
        # Operators
        gradient, laplacian, divergence, advection_term, AnalyticLaplacian,
        # Domain + samplers
        BoxDomain, UniformSampler, LHSSampler, AdaptiveSampler, GridSampler,
        # Conditions
        DirichletBC, NeumannBC, InitialCondition, PeriodicBC,
    )
"""

from __future__ import annotations

from .operators import (
    AnalyticLaplacian,
    advection_term,
    divergence,
    gradient,
    laplacian,
)
from .collocation import (
    AdaptiveSampler,
    BoxDomain,
    GridSampler,
    LHSSampler,
    UniformSampler,
)
from .constraints import (
    DirichletBC,
    InitialCondition,
    NeumannBC,
    PeriodicBC,
)

__all__ = [
    "gradient",
    "laplacian",
    "divergence",
    "advection_term",
    "AnalyticLaplacian",
    "BoxDomain",
    "UniformSampler",
    "LHSSampler",
    "AdaptiveSampler",
    "GridSampler",
    "DirichletBC",
    "NeumannBC",
    "InitialCondition",
    "PeriodicBC",
]
