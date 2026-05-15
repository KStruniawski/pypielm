"""GFF-PIELM: Generalised Fourier Feature Physics-Informed ELM.

Uses :class:`~pypielm.core.feature_maps.FourierFeatureMap` instead of a
standard random feature map.  The multi-scale frequency set enables accurate
approximation of high-frequency PDE solutions that standard random-activation
ELMs fail to capture.
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
from pypielm.core.feature_maps import FourierFeatureMap, FrequencyInit
from pypielm.core.solver import ridge_solve, rrqr_solve
from pypielm.models.registry import register
from pypielm.models.vanilla import _collect_blocks  # reuse block-collection logic

if TYPE_CHECKING:
    from pypielm.data.dataset import PIELMDataset


@register("gff_pielm")
class GFFPIELM(BasePIELM):
    """Generalised Fourier Feature PIELM (GFF-PIELM).

    Port of ``GFF-PIELM/gff_pielm.py`` to PyTorch with GPU support.

    Each hidden neuron computes:

    .. math::

        \\phi_j(\\mathbf{x}) =
            \\sqrt{2} \\cos\\!\\left(
                \\omega_j \\, \\mathbf{w}_j^\\top \\mathbf{x} + b_j
            \\right)

    The analytic second derivative w.r.t. each input dimension is used
    directly (no autograd overhead).

    Args:
        hidden_dim: Number of Fourier neurons.
        freq_init: Frequency initialisation strategy (``'log_uniform'``,
            ``'uniform'``).
        freq_min: Minimum frequency value.
        freq_max: Maximum frequency value.
        ridge_lambda: Output-weight regularisation.
        w_pde: Weight for PDE residual block.
        w_bc: Weight for BC block.
        w_ic: Weight for IC block.
        solver: ``'ridge'`` or ``'rrqr'``.
        seed: Random seed.
        device: Target device.
        dtype: Floating-point dtype.
    """

    def __init__(
        self,
        hidden_dim: int = 200,
        freq_init: FrequencyInit = "log_uniform",
        freq_min: float = 1.0,
        freq_max: float = 100.0,
        ridge_lambda: float = 1e-8,
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
        self.freq_init = freq_init
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.ridge_lambda = ridge_lambda
        self.w_pde = w_pde
        self.w_bc = w_bc
        self.w_ic = w_ic
        if solver not in ("ridge", "rrqr"):
            raise ValueError(f"solver must be 'ridge' or 'rrqr', got '{solver}'.")
        self.solver = solver
        self.seed = seed
        self.dtype = dtype
        self._device = torch.device(device) if isinstance(device, str) else device
        self._fm: FourierFeatureMap | None = None
        self.register_buffer("_beta", None)

    def _build_fm(self, input_dim: int) -> FourierFeatureMap:
        return FourierFeatureMap(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            freq_init=self.freq_init,
            freq_min=self.freq_min,
            freq_max=self.freq_max,
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
    ) -> GFFPIELM:
        input_dim = dataset.X_colloc.shape[1]
        if self._fm is None or self._fm.input_dim != input_dim:
            self._fm = self._build_fm(input_dim)

        blocks = _collect_blocks(
            self._fm, dataset, pde_operator, bcs, ics,  # type: ignore[arg-type]
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

    def __repr__(self) -> str:
        return (
            f"GFFPIELM(hidden_dim={self.hidden_dim}, "
            f"freq_init='{self.freq_init}', "
            f"freq_min={self.freq_min}, freq_max={self.freq_max}, "
            f"ridge_lambda={self.ridge_lambda})"
        )
