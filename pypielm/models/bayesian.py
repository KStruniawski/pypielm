"""Bayesian PIELM model.

Port of ``BPIELM/bpielm.py`` to a PyTorch-native, GPU-aware implementation.
Uses sequential Bayesian linear regression over weighted observation blocks
(PDE interior, BCs, ICs, data) rather than a single ridge solve, providing
posterior uncertainty estimates for the output weights.
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
)
from pypielm.core.feature_maps import RandomFeatureMap
from pypielm.core.solver import BayesianSolveResult, WeightedLinearSystem, bayesian_solve
from pypielm.models.registry import register
from pypielm.models.vanilla import _collect_blocks

if TYPE_CHECKING:
    from pypielm.data.dataset import PIELMDataset


@register("bayesian_pielm")
class BayesianPIELM(BasePIELM):
    """Physics-Informed ELM with Bayesian output-weight estimation.

    Instead of a single ridge solve, this model computes the full posterior
    distribution over output weights β via sequential Bayesian updates:

    .. math::

        p(\\boldsymbol{\\beta} \\mid \\text{data}, \\text{PDE})
        = \\mathcal{N}(\\boldsymbol{\\mu}_{\\text{post}},
                       \\boldsymbol{\\Lambda}_{\\text{post}}^{-1})

    Prediction is the posterior mean; uncertainty is propagated through the
    output layer giving pointwise confidence intervals on the PDE solution.

    Args:
        hidden_dim: Number of random neurons.
        activation: Activation function name.
        prior_precision: Precision α of the isotropic Gaussian prior on β.
        w_pde: Observation precision for PDE collocation blocks.
        w_bc: Observation precision for boundary condition blocks.
        w_ic: Observation precision for initial condition blocks.
        w_data: Observation precision for data-fit block.
        seed: Random seed.
        device: PyTorch device.
        dtype: Floating-point dtype.
    """

    def __init__(
        self,
        hidden_dim: int = 200,
        activation: str = "tanh",
        prior_precision: float = 1e-4,
        w_pde: float = 1.0,
        w_bc: float = 1.0,
        w_ic: float = 1.0,
        w_data: float = 1.0,
        seed: int = 42,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.prior_precision = prior_precision
        self.w_pde = w_pde
        self.w_bc = w_bc
        self.w_ic = w_ic
        self.w_data = w_data
        self.seed = seed
        self.dtype = dtype
        self._device = torch.device(device) if isinstance(device, str) else device
        self._fm: RandomFeatureMap | None = None
        # Posterior: beta_mean (H, out_dim), beta_precision (H, H)
        self.register_buffer("_beta", None)
        self.register_buffer("_beta_precision", None)

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
    ) -> BayesianPIELM:
        input_dim = dataset.X_colloc.shape[1]
        if self._fm is None or self._fm.input_dim != input_dim:
            self._fm = self._build_fm(input_dim)

        # Build blocks (w_data=1.0 by default; override data block weight separately)
        blocks = _collect_blocks(
            self._fm, dataset, pde_operator, bcs, ics,
            self.w_pde, self.w_bc, self.w_ic, self.dtype, self._device,
        )
        # Override weight of the data block (first block if X_data present)
        if (
            blocks
            and dataset.X_data is not None
            and dataset.y_data is not None
            and self.w_data != 1.0
        ):
            blk = blocks[0]
            blocks[0] = WeightedLinearSystem(blk.H, blk.y, self.w_data)

        if not blocks:
            raise ValueError(
                "No observation blocks assembled. Provide pde_operator, bcs, "
                "ics, or dataset.y_data."
            )

        result: BayesianSolveResult = bayesian_solve(blocks, self.prior_precision)
        self.register_buffer("_beta", result.beta_mean)
        self.register_buffer("_beta_precision", result.beta_cov)
        return self

    def predict(self, X: Array) -> Tensor:
        if self._fm is None or self._beta is None:
            raise RuntimeError("Call fit() before predict().")
        return self._fm(_ensure_tensor(X, self.dtype, self._device)) @ self._beta

    def predict_with_uncertainty(self, X: Array) -> tuple[Tensor, Tensor]:
        """Return (posterior mean, posterior std) at input points X.

        Args:
            X: Input coordinates, shape ``(N, d)``.

        Returns:
            Tuple ``(mean, std)`` each of shape ``(N, out_dim)``.
        """
        if self._fm is None or self._beta is None or self._beta_precision is None:
            raise RuntimeError("Call fit() before predict_with_uncertainty().")
        X_t = _ensure_tensor(X, self.dtype, self._device)
        H = self._fm(X_t)  # (N, H)
        mean = H @ self._beta  # (N, out_dim)

        # Posterior covariance of predictions: diag(H Λ⁻¹ Hᵀ)
        # Solve Λ @ V = Hᵀ  →  V = Λ⁻¹ Hᵀ
        try:
            L = torch.linalg.cholesky(self._beta_precision)
            V = torch.cholesky_solve(H.T, L)  # (H, N)
        except torch.linalg.LinAlgError:
            V = torch.linalg.solve(self._beta_precision, H.T)
        pred_var = (H * V.T).sum(dim=1, keepdim=True)  # (N, 1)
        std = pred_var.clamp(min=0.0).sqrt()  # (N, 1)
        if mean.shape[1] > 1:
            std = std.expand_as(mean)
        return mean, std

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

    def __repr__(self) -> str:
        return (
            f"BayesianPIELM(hidden_dim={self.hidden_dim}, "
            f"prior_precision={self.prior_precision}, "
            f"activation='{self.activation}')"
        )
