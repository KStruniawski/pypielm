"""PyPIELM: A Unified and Reproducible Framework for Physics-Informed Extreme Learning Machines.

Physics-Informed Extreme Learning Machines (PIELMs) solve PDEs by embedding
differential operators into the ELM training objective. The hidden-layer weights
are sampled randomly and frozen; only the output weights are determined
analytically (ridge regression, RRQR, or Bayesian solve), making training
orders of magnitude faster than gradient-based PINNs.

This package provides:

- 26+ PIELM variants and 4 PINN baselines under a unified ``fit/predict/score`` API.
- PyTorch-native implementation with autograd PDE operators and GPU support.
- Universal data adapters (CSV, NPZ, PINNacle, PDEBench, ``torch.utils.data.Dataset``).
- YAML-driven reproducible experiment configs and a CLI entry point.
- Publication-quality visualisation helpers.

Example::

    from pypielm.data import auto_load
    from pypielm.models import CorePIELM
    from pypielm.pde.operators import AnalyticLaplacian

    ds    = auto_load("poisson.dat", source="pinnacle")
    model = CorePIELM(hidden_dim=300, ridge_lambda=1e-8)
    model.fit(ds, pde_operator=AnalyticLaplacian())
    print(model.score(ds.X_test, ds.y_test))  # relative L²
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Krzysztof Struniawski"
__license__ = "MIT"

__all__ = [
    "__version__",
    "__author__",
    "__license__",
]
