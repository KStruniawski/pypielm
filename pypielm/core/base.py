"""Abstract base class and shared configuration for all PIELM models.

All PIELM and PINN variants in PyPIELM inherit from :class:`BasePIELM`, which
is itself a :class:`torch.nn.Module`.  The ELM paradigm mandates that hidden-layer
weights are **randomly sampled and frozen**; only the output layer weights
``beta`` are determined during :meth:`BasePIELM.fit` (analytically, not via
gradient descent).
"""

from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from pypielm.core.solver import WeightedLinearSystem
    from pypielm.data.dataset import PIELMDataset

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Tensor = torch.Tensor
Array = np.ndarray | torch.Tensor


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PIELMConfig:
    """Shared hyper-parameter configuration for PIELM models.

    Args:
        hidden_dim: Number of hidden neurons (random feature dimension H).
        activation: Activation function name. Supported: ``'tanh'``, ``'sin'``,
            ``'relu'``, ``'sigmoid'``, ``'softplus'``.
        ridge_lambda: Tikhonov regularisation coefficient λ used in the
            output-weight solve.  Typically in [1e-12, 1e-4].
        seed: Integer seed for reproducible hidden-layer weight sampling.
        device: PyTorch device string (e.g. ``'cpu'``, ``'cuda'``, ``'cuda:0'``).
        dtype: Floating-point dtype.  ``torch.float64`` (double precision) is
            the default because PDE residuals require high numerical accuracy.

    Example::

        cfg = PIELMConfig(hidden_dim=500, activation="tanh", ridge_lambda=1e-10)
    """

    hidden_dim: int = 200
    activation: str = "tanh"
    ridge_lambda: float = 1e-8
    seed: int = 42
    device: str = "cpu"
    dtype: torch.dtype = field(default_factory=lambda: torch.float64)


# ---------------------------------------------------------------------------
# Abstract base model
# ---------------------------------------------------------------------------

class BasePIELM(nn.Module):
    """Abstract base for all PIELM and PINN variants in PyPIELM.

    Subclasses must implement :meth:`fit`, :meth:`predict`, :meth:`score`, and
    :meth:`get_feature_matrix`.  Hidden-layer weights are **frozen** (ELM
    paradigm): they are initialised in ``__init__`` and never updated by a
    gradient step.  Output weights ``beta`` are solved analytically in
    :meth:`fit`.

    The class inherits from :class:`torch.nn.Module` to enable:

    * ``model.to(device)`` — move all parameters to a device.
    * ``torch.jit.script`` / ``torch.jit.trace`` — TorchScript export.
    * ``torch.onnx.export`` — ONNX export.
    * ``torch.compile`` — optional kernel fusion.

    Scikit-learn compatibility:

    * ``fit(dataset, ...)`` returns ``self`` (fluent chaining).
    * ``predict(X)`` returns a :class:`torch.Tensor`.
    * ``score(X, y)`` returns a scalar ``float``.
    """

    @abstractmethod
    def fit(
        self,
        dataset: PIELMDataset,
        *,
        pde_operator: Any | None = None,
        bcs: list[Any] | None = None,
        ics: list[Any] | None = None,
        collocation_sampler: Any | None = None,
    ) -> BasePIELM:
        """Solve for output weights analytically.

        Args:
            dataset: A :class:`~pypielm.data.PIELMDataset` containing
                ``X_train``, ``y_train``, and (optionally) validation splits.
            pde_operator: Optional differential operator applied at collocation
                points to build the physics residual block in the linear system.
                If ``None``, the model falls back to pure data regression.
            bcs: List of boundary condition objects
                (:class:`~pypielm.pde.constraints.DirichletBC`, etc.).
            ics: List of initial condition objects.
            collocation_sampler: Overrides the default collocation sampler used
                to generate interior PDE points.

        Returns:
            ``self`` — enables fluent chaining:
            ``model.fit(ds, pde_operator=op).predict(X_test)``.
        """
        ...

    @abstractmethod
    def predict(self, X: Array) -> Tensor:
        """Evaluate the surrogate solution at input coordinates X.

        Args:
            X: Input coordinates of shape ``(N, d)``.  Accepts both
                :class:`torch.Tensor` and :class:`numpy.ndarray`; the result
                is always a :class:`torch.Tensor`.

        Returns:
            Predicted values of shape ``(N, 1)`` or ``(N, out_dim)``.
        """
        ...

    @abstractmethod
    def score(
        self,
        X: Array,
        y: Array,
        metric: str = "relative_l2",
    ) -> float:
        """Compute a scalar accuracy metric on a held-out set.

        Args:
            X: Input coordinates of shape ``(N, d)``.
            y: Reference values of shape ``(N,)`` or ``(N, out_dim)``.
            metric: One of ``'relative_l2'``, ``'rmse'``, ``'mae'``, ``'r2'``,
                ``'max_error'``.

        Returns:
            Scalar metric value.  For error metrics (relative_l2, rmse, mae,
            max_error) lower is better; for R² higher is better.
        """
        ...

    @abstractmethod
    def get_feature_matrix(self, X: Array) -> Tensor:
        """Return the hidden-layer activation matrix Φ(X).

        Args:
            X: Input coordinates of shape ``(N, d)``.

        Returns:
            Feature matrix H of shape ``(N, hidden_dim)``.
        """
        ...


# ---------------------------------------------------------------------------
# Shared utility functions used by all model implementations
# ---------------------------------------------------------------------------

def _ensure_tensor(
    x: Array,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Convert *x* (numpy or tensor) to a ``torch.Tensor`` on *device*."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.to(dtype=dtype, device=device)


def _ensure_2d(t: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is at least 2-D: ``(N,)`` → ``(N, 1)``."""
    return t.unsqueeze(1) if t.ndim == 1 else t


def _compute_metric(
    pred: torch.Tensor,
    ref: torch.Tensor,
    metric: str,
) -> float:
    """Compute a scalar accuracy metric between *pred* and *ref*."""
    pred = pred.double()
    ref = ref.double()
    if pred.shape != ref.shape:
        ref = ref.view_as(pred)
    if metric == "relative_l2":
        denom = ref.norm()
        if denom < 1e-30:
            denom = torch.ones(1, dtype=ref.dtype, device=ref.device)
        return ((pred - ref).norm() / denom).item()
    if metric == "rmse":
        return ((pred - ref).pow(2).mean().sqrt()).item()
    if metric == "mae":
        return (pred - ref).abs().mean().item()
    if metric == "r2":
        ss_res = (pred - ref).pow(2).sum()
        ss_tot = (ref - ref.mean()).pow(2).sum()
        if ss_tot < 1e-30:
            return 1.0 if ss_res < 1e-30 else 0.0
        return (1.0 - ss_res / ss_tot).item()
    if metric == "max_error":
        return (pred - ref).abs().max().item()
    raise ValueError(f"Unknown metric '{metric}'. Choose from: relative_l2, rmse, mae, r2, max_error.")


def _stack_blocks(blocks: list[WeightedLinearSystem]) -> tuple[Tensor, Tensor]:
    """Stack weighted :class:`~pypielm.core.solver.WeightedLinearSystem` blocks.

    Multiplies each row block by ``sqrt(weight)`` to incorporate the precision
    weighting into a single overdetermined least-squares system.

    Args:
        blocks: Non-empty list of :class:`~pypielm.core.solver.WeightedLinearSystem`.

    Returns:
        ``(H_full, y_full)`` tensors ready for ``ridge_solve`` / ``rrqr_solve``.
    """
    if not blocks:
        raise ValueError("blocks must be non-empty.")
    H_parts: list[torch.Tensor] = []
    y_parts: list[torch.Tensor] = []
    for blk in blocks:
        w_sqrt = math.sqrt(float(blk.weight))
        H_parts.append(w_sqrt * blk.H)
        y = _ensure_2d(blk.y)
        y_parts.append(w_sqrt * y)
    return torch.cat(H_parts, dim=0), torch.cat(y_parts, dim=0)
