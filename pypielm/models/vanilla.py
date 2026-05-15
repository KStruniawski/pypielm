"""Vanilla and Core PIELM models.

* :class:`VanillaPIELM` — ELM with random features and ridge regression.
  No physics information; pure data-driven regression.  Useful as a
  performance lower-bound.

* :class:`CorePIELM` — the standard Physics-Informed ELM formulation.
  Assembles collocation blocks for PDE interior, boundary, and initial
  conditions into one augmented linear system and solves with ridge or RRQR.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from pypielm.core.base import (
    Array,
    BasePIELM,
    Tensor,
    _compute_metric,
    _ensure_2d,
    _ensure_tensor,
    _stack_blocks,
)
from pypielm.core.feature_maps import RandomFeatureMap
from pypielm.core.solver import WeightedLinearSystem, ridge_solve, rrqr_solve
from pypielm.models.registry import register

if TYPE_CHECKING:
    from pypielm.data.dataset import PIELMDataset


# ---------------------------------------------------------------------------
# Internal helper: collect all observation blocks from dataset + explicit args
# ---------------------------------------------------------------------------

def _collect_blocks(
    fm: RandomFeatureMap,
    dataset: PIELMDataset,
    pde_operator: Any | None,
    bcs: list[Any] | None,
    ics: list[Any] | None,
    w_pde: float,
    w_bc: float,
    w_ic: float,
    dtype: torch.dtype,
    device: torch.device,
) -> list[WeightedLinearSystem]:
    blocks: list[WeightedLinearSystem] = []

    # data block — if X_data is absent but y_data is provided, observations
    # are assumed to sit at the collocation points (natural for CSV/array inputs).
    # Guard: only use the fallback when sizes actually match to prevent silent
    # shape mismatches (e.g. when a caller subsamples X_colloc independently).
    if dataset.y_data is not None:
        if dataset.X_data is not None:
            X_d = _ensure_tensor(dataset.X_data, dtype, device)
        elif dataset.X_colloc.shape[0] == dataset.y_data.shape[0]:
            X_d = _ensure_tensor(dataset.X_colloc, dtype, device)
        else:
            X_d = None
        if X_d is not None:
            y_d = _ensure_2d(_ensure_tensor(dataset.y_data, dtype, device))
            blocks.append(WeightedLinearSystem(fm(X_d), y_d, 1.0))

    # PDE interior block
    if pde_operator is not None:
        X_c = _ensure_tensor(dataset.X_colloc, dtype, device)
        blk = pde_operator(fm, X_c)
        blocks.append(WeightedLinearSystem(blk.H, blk.y, w_pde * float(blk.weight)))

    # boundary conditions
    if bcs:
        for bc in bcs:
            blk = bc.assemble(fm)
            blocks.append(WeightedLinearSystem(blk.H, blk.y, w_bc * float(blk.weight)))
    elif dataset.X_bc is not None and dataset.y_bc is not None:
        X_bc = _ensure_tensor(dataset.X_bc, dtype, device)
        y_bc = _ensure_2d(_ensure_tensor(dataset.y_bc, dtype, device))
        blocks.append(WeightedLinearSystem(fm(X_bc), y_bc, w_bc))

    # initial conditions
    if ics:
        for ic in ics:
            blk = ic.assemble(fm)
            blocks.append(WeightedLinearSystem(blk.H, blk.y, w_ic * float(blk.weight)))
    elif dataset.X_ic is not None and dataset.y_ic is not None:
        X_ic = _ensure_tensor(dataset.X_ic, dtype, device)
        y_ic = _ensure_2d(_ensure_tensor(dataset.y_ic, dtype, device))
        blocks.append(WeightedLinearSystem(fm(X_ic), y_ic, w_ic))

    return blocks


# ---------------------------------------------------------------------------
# VanillaPIELM
# ---------------------------------------------------------------------------

@register("vanilla_pielm")
class VanillaPIELM(BasePIELM):
    """ELM regression with random features and ridge solve — no physics.

    Args:
        hidden_dim: Number of random neurons.
        ridge_lambda: Ridge regularisation lambda.
        activation: Activation function name.
        seed: Random seed for hidden-layer weights.
        device: PyTorch device.
        dtype: Floating-point dtype.
    """

    def __init__(
        self,
        hidden_dim: int = 200,
        ridge_lambda: float = 1e-8,
        activation: str = "tanh",
        seed: int = 42,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ridge_lambda = ridge_lambda
        self.activation = activation
        self.seed = seed
        self.dtype = dtype
        self._device = torch.device(device) if isinstance(device, str) else device
        self._fm: RandomFeatureMap | None = None
        self.register_buffer("_beta", None)

    def _build_fm(self, input_dim: int) -> RandomFeatureMap:
        return RandomFeatureMap(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            activation=self.activation,
            seed=self.seed,
            device=self._device,
            dtype=self.dtype,
        )

    def fit(self, dataset: PIELMDataset, **kwargs: Any) -> VanillaPIELM:
        X = dataset.X_data if dataset.X_data is not None else dataset.X_colloc
        y = dataset.y_data
        if y is None:
            raise ValueError("VanillaPIELM requires dataset.y_data.")
        X = _ensure_tensor(X, self.dtype, self._device)
        y = _ensure_2d(_ensure_tensor(y, self.dtype, self._device))
        self._fm = self._build_fm(X.shape[1])
        beta = ridge_solve(self._fm(X), y, self.ridge_lambda)
        self.register_buffer("_beta", beta)
        return self

    def predict(self, X: Array) -> Tensor:
        if self._fm is None or self._beta is None:
            raise RuntimeError("Call fit() before predict().")
        return self._fm(_ensure_tensor(X, self.dtype, self._device)) @ self._beta

    def score(self, X: Array, y: Array, metric: str = "relative_l2") -> float:
        return _compute_metric(
            self.predict(X),
            _ensure_2d(_ensure_tensor(y, self.dtype, self._device)),
            metric,
        )

    def get_feature_matrix(self, X: Array) -> Tensor:
        if self._fm is None:
            raise RuntimeError("Call fit() before get_feature_matrix().")
        return self._fm(_ensure_tensor(X, self.dtype, self._device))

    def forward(self, X: Tensor) -> Tensor:  # nn.Module interface for tracing
        return self.predict(X)

    def __repr__(self) -> str:
        return (
            f"VanillaPIELM(hidden_dim={self.hidden_dim}, "
            f"ridge_lambda={self.ridge_lambda}, "
            f"activation='{self.activation}')"
        )


# ---------------------------------------------------------------------------
# CorePIELM
# ---------------------------------------------------------------------------

@register("core_pielm")
class CorePIELM(BasePIELM):
    """Physics-Informed ELM: random features + physics-augmented linear system.

    Assembles an augmented weighted least-squares system from PDE collocation
    blocks, boundary/initial condition blocks, and optionally observed data,
    then solves analytically via ridge regression or RRQR.

    The ``pde_operator`` argument is a callable with signature::

        pde_operator(feature_map, X_colloc) -> WeightedLinearSystem

    where ``WeightedLinearSystem.H`` is the PDE operator applied to the feature
    matrix (e.g. the Laplacian feature matrix) and ``WeightedLinearSystem.y``
    is the RHS of the PDE evaluated at the collocation points.

    Args:
        hidden_dim: Number of random neurons.
        ridge_lambda: Regularisation strength.
        activation: Activation function.
        w_pde: Weight on PDE residual rows.
        w_bc: Weight on boundary condition rows.
        w_ic: Weight on initial condition rows.
        solver: ``'ridge'`` or ``'rrqr'``.
        seed: Random seed.
        device: PyTorch device.
        dtype: Floating-point dtype.
    """

    def __init__(
        self,
        hidden_dim: int = 200,
        ridge_lambda: float = 1e-8,
        activation: str = "tanh",
        w_pde: float = 1.0,
        w_bc: float = 1.0,
        w_ic: float = 1.0,
        solver: str = "ridge",
        seed: int = 42,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ridge_lambda = ridge_lambda
        self.activation = activation
        self.w_pde = w_pde
        self.w_bc = w_bc
        self.w_ic = w_ic
        if solver not in ("ridge", "rrqr"):
            raise ValueError(f"solver must be 'ridge' or 'rrqr', got '{solver}'.")
        self.solver = solver
        self.seed = seed
        self.dtype = dtype
        self._device = torch.device(device) if isinstance(device, str) else device
        self._fm: RandomFeatureMap | None = None
        self.register_buffer("_beta", None)

    def _build_fm(self, input_dim: int) -> RandomFeatureMap:
        return RandomFeatureMap(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            activation=self.activation,
            seed=self.seed,
            device=self._device,
            dtype=self.dtype,
        )

    def fit(
        self,
        dataset: PIELMDataset,
        *,
        pde_operator: Any | None = None,
        bcs: list[Any] | None = None,
        ics: list[Any] | None = None,
        collocation_sampler: Any | None = None,
    ) -> CorePIELM:
        input_dim = dataset.X_colloc.shape[1]
        if self._fm is None or self._fm.input_dim != input_dim:
            self._fm = self._build_fm(input_dim)

        blocks = _collect_blocks(
            self._fm, dataset, pde_operator, bcs, ics,
            self.w_pde, self.w_bc, self.w_ic, self.dtype, self._device,
        )
        if not blocks:
            raise ValueError(
                "No observation blocks assembled. Provide pde_operator, bcs, "
                "ics, or dataset.y_data."
            )

        H_full, y_full = _stack_blocks(blocks)
        if self.solver == "ridge":
            beta = ridge_solve(H_full, y_full, self.ridge_lambda)
        else:
            beta = rrqr_solve(H_full, y_full)

        self.register_buffer("_beta", beta)
        return self

    def predict(self, X: Array) -> Tensor:
        if self._fm is None or self._beta is None:
            raise RuntimeError("Call fit() before predict().")
        return self._fm(_ensure_tensor(X, self.dtype, self._device)) @ self._beta

    def score(self, X: Array, y: Array, metric: str = "relative_l2") -> float:
        return _compute_metric(
            self.predict(X),
            _ensure_2d(_ensure_tensor(y, self.dtype, self._device)),
            metric,
        )

    def get_feature_matrix(self, X: Array) -> Tensor:
        if self._fm is None:
            raise RuntimeError("Call fit() before get_feature_matrix().")
        return self._fm(_ensure_tensor(X, self.dtype, self._device))

    def forward(self, X: Tensor) -> Tensor:  # nn.Module interface for tracing
        return self.predict(X)

    def __repr__(self) -> str:
        return (
            f"CorePIELM(hidden_dim={self.hidden_dim}, "
            f"ridge_lambda={self.ridge_lambda}, "
            f"solver='{self.solver}', "
            f"activation='{self.activation}')"
        )
