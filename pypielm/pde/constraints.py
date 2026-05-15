"""Boundary and initial condition helpers.

Each condition class evaluates the constraint at given boundary/IC points and
returns a :class:`~pypielm.core.solver.WeightedLinearSystem` tuple
``(H_bc, y_bc, weight)`` that can be stacked into the global linear system
assembled during model training.

Public API::

    from pypielm.pde.constraints import (
        DirichletBC, NeumannBC, InitialCondition, PeriodicBC
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

if TYPE_CHECKING:
    from pypielm.core.feature_maps import RandomFeatureMap
    from pypielm.core.solver import WeightedLinearSystem


class DirichletBC:
    """Hard Dirichlet boundary condition: u(x) = g(x) on ∂Ω.

    Args:
        boundary_fn: Callable ``g(x: Tensor) → Tensor`` returning the
            prescribed values at boundary points, shape ``(N_bc,)`` or
            ``(N_bc, 1)``.
        points: Boundary collocation points, shape ``(N_bc, d)``.
        weight: Observation precision for this BC block.
    """

    def __init__(
        self,
        boundary_fn: Callable[[torch.Tensor], torch.Tensor],
        points: torch.Tensor,
        weight: float = 1.0,
    ) -> None:
        self.boundary_fn = boundary_fn
        self.points = points
        self.weight = float(weight)

    def assemble(
        self, feature_map: "RandomFeatureMap"
    ) -> "WeightedLinearSystem":
        """Evaluate BC and return the linear system block.

        Args:
            feature_map: The model's hidden-layer feature map.

        Returns:
            :class:`~pypielm.core.solver.WeightedLinearSystem` with
            ``H = feature_map(points)``, ``y = boundary_fn(points)``,
            ``weight = self.weight``.
        """
        from pypielm.core.solver import WeightedLinearSystem

        H = feature_map(self.points)  # (N_bc, hidden_dim)
        y = self.boundary_fn(self.points)
        if y.ndim == 1:
            y = y.unsqueeze(1)
        return WeightedLinearSystem(H=H, y=y, weight=self.weight)


class NeumannBC:
    """Neumann boundary condition: ∂u/∂n = h(x) on ∂Ω.

    The feature-matrix contribution uses the outward unit normal ``n`` to form
    the directional derivative: ∂H/∂n = Σᵢ nᵢ · (∂H/∂xᵢ).

    Args:
        flux_fn: Callable returning the prescribed normal flux, shape
            ``(N_bc,)`` or ``(N_bc, 1)``.
        normal: Outward unit normal vectors, shape ``(N_bc, d)``.
        points: Boundary collocation points, shape ``(N_bc, d)``.
        weight: Observation precision.
    """

    def __init__(
        self,
        flux_fn: Callable[[torch.Tensor], torch.Tensor],
        normal: torch.Tensor,
        points: torch.Tensor,
        weight: float = 1.0,
    ) -> None:
        self.flux_fn = flux_fn
        self.normal = normal
        self.points = points
        self.weight = float(weight)

    def assemble(
        self, feature_map: "RandomFeatureMap"
    ) -> "WeightedLinearSystem":
        """Evaluate flux BC and return the linear system block.

        Builds the directional-derivative feature matrix::

            H_n[i, j] = Σ_k  n[i, k] * (∂H/∂x_k)[i, j]
        """
        from pypielm.core.solver import WeightedLinearSystem

        d = self.points.shape[1]
        # ∂H/∂n = Σ_k n_k * d1(X, k)
        H_n = torch.zeros(
            self.points.shape[0],
            feature_map.hidden_dim,
            dtype=self.points.dtype,
            device=self.points.device,
        )
        for k in range(d):
            dH_k = feature_map.d1(self.points, k)  # (N_bc, H)
            H_n = H_n + self.normal[:, k : k + 1] * dH_k

        y = self.flux_fn(self.points)
        if y.ndim == 1:
            y = y.unsqueeze(1)
        return WeightedLinearSystem(H=H_n, y=y, weight=self.weight)


class InitialCondition:
    """Initial condition: u(x, t=0) = u₀(x).

    Args:
        ic_fn: Callable returning prescribed initial values, shape
            ``(N_ic,)`` or ``(N_ic, 1)``.
        points: Initial condition points (with t=0 already embedded),
            shape ``(N_ic, d)``.
        weight: Observation precision.
    """

    def __init__(
        self,
        ic_fn: Callable[[torch.Tensor], torch.Tensor],
        points: torch.Tensor,
        weight: float = 1.0,
    ) -> None:
        self.ic_fn = ic_fn
        self.points = points
        self.weight = float(weight)

    def assemble(
        self, feature_map: "RandomFeatureMap"
    ) -> "WeightedLinearSystem":
        """Evaluate IC and return the linear system block."""
        from pypielm.core.solver import WeightedLinearSystem

        H = feature_map(self.points)
        y = self.ic_fn(self.points)
        if y.ndim == 1:
            y = y.unsqueeze(1)
        return WeightedLinearSystem(H=H, y=y, weight=self.weight)


class PeriodicBC:
    """Periodic boundary condition along a specified axis.

    Pairs boundary points ``x_left`` and ``x_right`` and enforces
    u(x_left) = u(x_right) by adding the penalty rows
    ``H(x_left) - H(x_right)`` with target ``y = 0`` to the linear system.

    Args:
        axis: Axis along which periodicity is imposed (informational only;
            the caller is responsible for pairing points correctly).
        points_left: Left boundary points, shape ``(N_bc, d)``.
        points_right: Right boundary points, shape ``(N_bc, d)``.
        weight: Observation precision for the pairing rows.
    """

    def __init__(
        self,
        axis: int,
        points_left: torch.Tensor,
        points_right: torch.Tensor,
        weight: float = 1.0,
    ) -> None:
        self.axis = int(axis)
        self.points_left = points_left
        self.points_right = points_right
        self.weight = float(weight)

    def assemble(
        self, feature_map: "RandomFeatureMap"
    ) -> "WeightedLinearSystem":
        """Assemble the pairing penalty block.

        Returns rows ``H(x_left) - H(x_right)`` with target zero, so the
        solver enforces u(x_left) ≈ u(x_right).
        """
        from pypielm.core.solver import WeightedLinearSystem

        H_left = feature_map(self.points_left)    # (N, H)
        H_right = feature_map(self.points_right)  # (N, H)
        H_diff = H_left - H_right                 # (N, H)
        y = torch.zeros(H_diff.shape[0], 1, dtype=H_diff.dtype, device=H_diff.device)
        return WeightedLinearSystem(H=H_diff, y=y, weight=self.weight)


