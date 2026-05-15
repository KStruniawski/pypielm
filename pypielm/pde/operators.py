"""PDE differential operators (autograd + analytic fast paths).

Public API::

    from pypielm.pde.operators import (
        gradient, laplacian, divergence, advection_term, AnalyticLaplacian
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pypielm.core.feature_maps import RandomFeatureMap


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_2d(t: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is 2-D (N, 1) if it is 1-D (N,)."""
    if t.ndim == 1:
        return t.unsqueeze(1)
    return t


def _autograd_grad(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
    *,
    create_graph: bool = False,
    allow_unused: bool = False,
) -> torch.Tensor:
    """Wrapper around torch.autograd.grad that returns a (N, d) gradient."""
    (grad,) = torch.autograd.grad(
        outputs.sum(),
        inputs,
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=allow_unused,
    )
    if grad is None:
        grad = torch.zeros_like(inputs)
    return grad  # (N, d)


# ---------------------------------------------------------------------------
# Autograd operators
# ---------------------------------------------------------------------------

def gradient(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute the gradient ∇u w.r.t. x via autograd.

    .. math:: \\nabla u = \\left(\\frac{\\partial u}{\\partial x_1}, \\ldots,
              \\frac{\\partial u}{\\partial x_d}\\right)

    Args:
        u: Scalar field tensor, shape ``(N,)`` or ``(N, 1)``.  Must have
            ``requires_grad=True`` set on the computation graph leading to it.
        x: Input coordinates, shape ``(N, d)``, with ``requires_grad=True``.

    Returns:
        Gradient tensor of shape ``(N, d)``.
    """
    u = _ensure_2d(u)
    return _autograd_grad(u, x)


def laplacian(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute the Laplacian Δu = Σᵢ ∂²u/∂xᵢ² via autograd.

    .. math:: \\Delta u = \\sum_{i=1}^{d} \\frac{\\partial^2 u}{\\partial x_i^2}

    Args:
        u: Scalar field, shape ``(N,)`` or ``(N, 1)``.
        x: Input coordinates, shape ``(N, d)``, with ``requires_grad=True``.

    Returns:
        Laplacian values, shape ``(N, 1)``.
    """
    u = _ensure_2d(u)
    # First gradient — keep graph so we can differentiate again
    grad_u = _autograd_grad(u, x, create_graph=True)  # (N, d)

    lap = torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
    d = x.shape[1]
    for i in range(d):
        # ∂²u/∂xᵢ² = ∂(∂u/∂xᵢ)/∂xᵢ
        gi = grad_u[:, i : i + 1]
        if not gi.requires_grad:
            # The i-th partial is constant w.r.t. x → ∂²u/∂xᵢ² = 0
            continue
        g2 = _autograd_grad(gi, x, allow_unused=True)
        lap = lap + g2[:, i : i + 1]
    return lap


def divergence(flux: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute the divergence ∇ · F of a vector field F.

    .. math:: \\nabla \\cdot \\mathbf{F} = \\sum_{i=1}^{d} \\frac{\\partial F_i}{\\partial x_i}

    Args:
        flux: Vector field tensor, shape ``(N, d)``.
        x: Input coordinates, shape ``(N, d)``, with ``requires_grad=True``.

    Returns:
        Divergence values, shape ``(N, 1)``.
    """
    d = flux.shape[1]
    div = torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
    for i in range(d):
        gi = _autograd_grad(flux[:, i : i + 1], x)  # (N, d)
        div = div + gi[:, i : i + 1]
    return div


def advection_term(
    u: torch.Tensor,
    v: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Compute the advection term v · ∇u.

    .. math:: \\mathbf{v} \\cdot \\nabla u = \\sum_{i=1}^{d} v_i \\frac{\\partial u}{\\partial x_i}

    Args:
        u: Scalar field, shape ``(N,)`` or ``(N, 1)``.
        v: Advection velocity, shape ``(N, d)``.
        x: Input coordinates, shape ``(N, d)``, with ``requires_grad=True``.

    Returns:
        Advection term, shape ``(N, 1)``.
    """
    u = _ensure_2d(u)
    grad_u = _autograd_grad(u, x)  # (N, d)
    return (v * grad_u).sum(dim=1, keepdim=True)  # (N, 1)


# ---------------------------------------------------------------------------
# Analytic fast path
# ---------------------------------------------------------------------------

class AnalyticLaplacian:
    """Fast Laplacian operator using precomputed analytic second derivatives.

    Avoids the O(N · H · d) autograd overhead by using the analytic
    :meth:`~pypielm.core.feature_maps.RandomFeatureMap.d2` (or
    :meth:`~pypielm.core.feature_maps.RandomFeatureMap.laplacian`) method
    of the feature map directly.

    When ``feature_map`` is provided and exposes a ``laplacian`` method,
    that is used directly.  Otherwise the :meth:`d2` method is summed over
    all spatial dimensions.

    .. math::

        \\Delta [H \\boldsymbol{\\beta}](x) =
            \\left(\\sum_{i=1}^{d} \\mathbf{H}^{(2,i)}(x)\\right) \\boldsymbol{\\beta}

    where :math:`\\mathbf{H}^{(2,i)}` is the second partial derivative of H
    w.r.t. coordinate :math:`x_i`.

    Args:
        feature_map: A :class:`~pypielm.core.feature_maps.RandomFeatureMap`
            or compatible feature map instance (must expose ``d2`` or
            ``laplacian``).
        input_dim: Number of spatial dimensions to sum over.  Inferred from
            ``feature_map.input_dim`` if ``feature_map`` is provided.
    """

    def __init__(
        self,
        feature_map: "RandomFeatureMap | None" = None,
        input_dim: int | None = None,
    ) -> None:
        self.feature_map = feature_map
        if input_dim is None and feature_map is not None:
            input_dim = feature_map.input_dim
        self.input_dim = input_dim

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the Laplacian feature matrix Σᵢ ∂²H/∂xᵢ².

        Args:
            X: Input coordinates, shape ``(N, d)``.

        Returns:
            Laplacian feature matrix, shape ``(N, H)``.

        Raises:
            ValueError: If no feature map was provided.
        """
        if self.feature_map is None:
            raise ValueError(
                "AnalyticLaplacian requires a feature_map to be provided at "
                "construction time."
            )
        fm = self.feature_map
        # Use the built-in laplacian method if available (RandomFeatureMap)
        if hasattr(fm, "laplacian"):
            dims = (
                list(range(self.input_dim)) if self.input_dim is not None else None
            )
            return fm.laplacian(X, dims=dims)

        # Fallback: sum d2 over all axes
        d = self.input_dim if self.input_dim is not None else X.shape[1]
        lap = fm.d2(X, 0)
        for i in range(1, d):
            lap = lap + fm.d2(X, i)
        return lap
