"""Domain-decomposition PIELM variants.

* :class:`DPIELM`      — Distributed PIELM: fixed uniform decomposition.
* :class:`LocELM`      — Localised ELM: each subdomain has its own feature map.
* :class:`DDELMCoarse` — DD-ELM with a coarse global correction layer.
"""

from __future__ import annotations

from dataclasses import dataclass
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
from pypielm.models.registry import register

if TYPE_CHECKING:
    from pypielm.data.dataset import PIELMDataset


# ---------------------------------------------------------------------------
# Internal subdomain bookkeeping
# ---------------------------------------------------------------------------

@dataclass
class _SubdomainModel:
    center: torch.Tensor     # (d,)  centroid of training points
    x_min: torch.Tensor      # (d,)  bounding box lower-left
    x_max: torch.Tensor      # (d,)  bounding box upper-right
    beta: torch.Tensor       # (H, out_dim)
    fm: Any                  # RandomFeatureMap


def _quantile_indices(
    X: torch.Tensor,
    sub_id: int,
    n_sub: int,
    overlap: float,
    axis: int = 0,
) -> torch.Tensor:
    """Return boolean mask for points belonging to subdomain *sub_id*."""
    vals = X[:, axis]
    q0 = max(0.0, sub_id / n_sub - overlap / n_sub)
    q1 = min(1.0, (sub_id + 1) / n_sub + overlap / n_sub)
    # torch.quantile requires CPU on some backends (e.g. MPS lacks lerp)
    vals_cpu = vals.float().cpu()
    lo = torch.quantile(vals_cpu, q0).to(dtype=vals.dtype, device=vals.device)
    hi = torch.quantile(vals_cpu, q1).to(dtype=vals.dtype, device=vals.device)
    return (vals >= lo) & (vals <= hi)


# ---------------------------------------------------------------------------
# Base class shared by DPIELM and LocELM
# ---------------------------------------------------------------------------

class _DomainDecompositionBase(BasePIELM):
    """Shared fit/predict logic for nearest-centroid domain decomposition."""

    _variant_name: str = "DomainDecomposition"

    def __init__(
        self,
        n_subdomains: int = 4,
        overlap: float = 0.1,
        hidden_dim: int = 128,
        ridge_lambda: float = 1e-8,
        activation: str = "tanh",
        seed: int = 42,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
        *,
        local_seeds: bool = True,
    ) -> None:
        super().__init__()
        self.n_subdomains = max(1, int(n_subdomains))
        self.overlap = max(0.0, float(overlap))
        self.hidden_dim = hidden_dim
        self.ridge_lambda = ridge_lambda
        self.activation = activation
        self.seed = seed
        self.dtype = dtype
        self._device = torch.device(device) if isinstance(device, str) else device
        self._local_seeds = local_seeds
        self._submodels: list[_SubdomainModel] = []

    def _build_fm(self, input_dim: int, sub_seed: int):
        from pypielm.core.feature_maps import RandomFeatureMap
        return RandomFeatureMap(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            activation=self.activation,
            seed=sub_seed,
            device=self._device,
            dtype=self.dtype,
        )

    def _fit_subdomain(
        self, X_sub: torch.Tensor, y_sub: torch.Tensor, sub_id: int
    ) -> _SubdomainModel:
        from pypielm.core.solver import ridge_solve
        sub_seed = self.seed + sub_id if self._local_seeds else self.seed
        fm = self._build_fm(X_sub.shape[1], sub_seed)
        H = fm(X_sub)
        beta = ridge_solve(H, y_sub, self.ridge_lambda)
        center = X_sub.mean(dim=0)
        x_min = X_sub.min(dim=0).values
        x_max = X_sub.max(dim=0).values
        return _SubdomainModel(center=center, x_min=x_min, x_max=x_max,
                               beta=beta, fm=fm)

    def fit(self, dataset: PIELMDataset, **kwargs: Any) -> _DomainDecompositionBase:
        X = dataset.X_data if dataset.X_data is not None else dataset.X_colloc
        y = dataset.y_data
        if y is None:
            raise ValueError(f"{self.__class__.__name__} requires dataset.y_data.")
        X = _ensure_tensor(X, self.dtype, self._device)
        y = _ensure_2d(_ensure_tensor(y, self.dtype, self._device))

        self._submodels = []
        for sub_id in range(self.n_subdomains):
            mask = _quantile_indices(X, sub_id, self.n_subdomains, self.overlap)
            X_sub = X[mask]
            y_sub = y[mask]
            min_pts = max(6, X_sub.shape[1] + 2)
            if X_sub.shape[0] < min_pts:
                continue
            self._submodels.append(self._fit_subdomain(X_sub, y_sub, sub_id))

        if not self._submodels:
            # Fallback: single model on full dataset
            self._submodels.append(self._fit_subdomain(X, y, 0))

        return self

    def _nearest_subdomain_weights(self, X: torch.Tensor) -> torch.Tensor:
        """Return (N, n_sub) one-hot weight matrix for nearest-centroid assignment."""
        centers = torch.stack([m.center for m in self._submodels], dim=0)  # (S, d)
        d2 = ((X.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(dim=2)  # (N, S)
        idx = d2.argmin(dim=1)  # (N,)
        W = torch.zeros(X.shape[0], len(self._submodels),
                        dtype=X.dtype, device=X.device)
        W.scatter_(1, idx.unsqueeze(1), 1.0)
        return W

    def predict(self, X: Array) -> Tensor:
        if not self._submodels:
            raise RuntimeError("Call fit() before predict().")
        X_t = _ensure_tensor(X, self.dtype, self._device)
        W = self._nearest_subdomain_weights(X_t)  # (N, S)
        preds = []
        for sm in self._submodels:
            H = sm.fm(X_t)
            preds.append(H @ sm.beta)  # (N, out_dim)
        pred_stack = torch.stack(preds, dim=2)  # (N, out_dim, S)
        return (pred_stack * W.unsqueeze(1)).sum(dim=2)  # (N, out_dim)

    def score(self, X: Array, y: Array, metric: str = "relative_l2") -> float:
        return _compute_metric(
            self.predict(X),
            _ensure_2d(_ensure_tensor(y, self.dtype, self._device)),
            metric,
        )

    def get_feature_matrix(self, X: Array) -> Tensor:
        """Return feature matrix from the nearest subdomain's feature map."""
        if not self._submodels:
            raise RuntimeError("Call fit() before get_feature_matrix().")
        X_t = _ensure_tensor(X, self.dtype, self._device)
        W = self._nearest_subdomain_weights(X_t)  # (N, S)
        preds = [sm.fm(X_t) for sm in self._submodels]  # each (N, H)
        stacked = torch.stack(preds, dim=2)  # (N, H, S)
        return (stacked * W.unsqueeze(1)).sum(dim=2)  # (N, H)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_subdomains={self.n_subdomains}, "
            f"overlap={self.overlap}, "
            f"hidden_dim={self.hidden_dim})"
        )


# ---------------------------------------------------------------------------
# DPIELM
# ---------------------------------------------------------------------------

@register("dpielm")
class DPIELM(_DomainDecompositionBase):
    """Distributed PIELM with fixed uniform domain decomposition.

    The spatial domain is partitioned into ``n_subdomains`` regions along the
    first spatial axis.  Each subdomain trains an independent ELM.  Predictions
    are assembled by assigning each query point to its containing subdomain.

    Args:
        n_subdomains: Number of subdomains.
        overlap: Fractional overlap between adjacent subdomains (0 = none).
        hidden_dim: Hidden neurons per subdomain.
        ridge_lambda: Regularisation per subdomain.
        activation: Activation function.
        seed: Random seed.
        device: Target device.
        dtype: Floating-point dtype.
    """

    _variant_name = "DPIELM"

    def __init__(
        self,
        n_subdomains: int = 4,
        overlap: float = 0.1,
        hidden_dim: int = 128,
        ridge_lambda: float = 1e-8,
        activation: str = "tanh",
        seed: int = 42,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__(
            n_subdomains=n_subdomains, overlap=overlap, hidden_dim=hidden_dim,
            ridge_lambda=ridge_lambda, activation=activation, seed=seed,
            device=device, dtype=dtype, local_seeds=False,
        )


# ---------------------------------------------------------------------------
# LocELM
# ---------------------------------------------------------------------------

@register("locelm")
class LocELM(_DomainDecompositionBase):
    """Localised ELM (LocELM): independent local feature maps per subdomain.

    Similar to :class:`DPIELM` but each subdomain has its own independently
    initialised random feature map (different seed per subdomain).

    Args:
        n_subdomains: Number of subdomains.
        overlap: Fractional overlap between adjacent subdomains.
        hidden_dim: Hidden neurons per subdomain.
        ridge_lambda: Regularisation per subdomain.
        activation: Activation function.
        seed: Base random seed; each subdomain gets ``seed + sub_id``.
        device: Target device.
        dtype: Floating-point dtype.
    """

    _variant_name = "LocELM"

    def __init__(
        self,
        n_subdomains: int = 6,
        overlap: float = 0.25,
        hidden_dim: int = 160,
        ridge_lambda: float = 1e-8,
        activation: str = "tanh",
        seed: int = 42,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__(
            n_subdomains=n_subdomains, overlap=overlap, hidden_dim=hidden_dim,
            ridge_lambda=ridge_lambda, activation=activation, seed=seed,
            device=device, dtype=dtype, local_seeds=True,
        )


# ---------------------------------------------------------------------------
# DDELMCoarse
# ---------------------------------------------------------------------------

@register("ddelm_coarse")
class DDELMCoarse(_DomainDecompositionBase):
    """Domain-Decomposition ELM with a coarse global correction layer.

    Combines a local domain-decomposition solve (like :class:`DPIELM`) with a
    single global ELM trained on the full dataset, blending the two predictions:

    .. math::

        \\hat{u}(x) = (1 - \\alpha_{\\text{coarse}})\\, \\hat{u}_{\\text{local}}(x)
                    + \\alpha_{\\text{coarse}}\\, \\hat{u}_{\\text{coarse}}(x)

    Args:
        n_subdomains: Number of local subdomains.
        overlap: Subdomain overlap fraction.
        hidden_dim: Hidden neurons per subdomain.
        coarse_hidden_dim: Hidden neurons in the global correction ELM.
        coarse_alpha: Blending weight for the coarse model (default 0.2).
        ridge_lambda: Regularisation.
        activation: Activation function.
        seed: Random seed.
        device: Target device.
        dtype: Floating-point dtype.
    """

    _variant_name = "DDELMCoarse"

    def __init__(
        self,
        n_subdomains: int = 6,
        overlap: float = 0.15,
        hidden_dim: int = 128,
        coarse_hidden_dim: int = 64,
        coarse_alpha: float = 0.2,
        ridge_lambda: float = 1e-8,
        activation: str = "tanh",
        seed: int = 42,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__(
            n_subdomains=n_subdomains, overlap=overlap, hidden_dim=hidden_dim,
            ridge_lambda=ridge_lambda, activation=activation, seed=seed,
            device=device, dtype=dtype, local_seeds=True,
        )
        self.coarse_hidden_dim = coarse_hidden_dim
        self.coarse_alpha = coarse_alpha
        # Coarse global correction model
        self._coarse_sub: _SubdomainModel | None = None

    def fit(self, dataset: PIELMDataset, **kwargs: Any) -> DDELMCoarse:
        # Fit local subdomains
        super().fit(dataset, **kwargs)

        # Fit global correction model
        X = dataset.X_data if dataset.X_data is not None else dataset.X_colloc
        y = dataset.y_data
        if y is not None:
            X_t = _ensure_tensor(X, self.dtype, self._device)
            y_t = _ensure_2d(_ensure_tensor(y, self.dtype, self._device))
            # Use a large seed offset for the coarse model
            fm = self._build_fm_coarse(X_t.shape[1])
            from pypielm.core.solver import ridge_solve
            H = fm(X_t)
            beta = ridge_solve(H, y_t, self.ridge_lambda)
            self._coarse_sub = _SubdomainModel(
                center=X_t.mean(dim=0),
                x_min=X_t.min(dim=0).values,
                x_max=X_t.max(dim=0).values,
                beta=beta,
                fm=fm,
            )
        return self

    def _build_fm_coarse(self, input_dim: int):
        from pypielm.core.feature_maps import RandomFeatureMap
        return RandomFeatureMap(
            input_dim=input_dim,
            hidden_dim=self.coarse_hidden_dim,
            activation=self.activation,
            seed=self.seed + 999,
            device=self._device,
            dtype=self.dtype,
        )

    def predict(self, X: Array) -> Tensor:
        local = super().predict(X)
        if self._coarse_sub is None:
            return local
        X_t = _ensure_tensor(X, self.dtype, self._device)
        coarse = self._coarse_sub.fm(X_t) @ self._coarse_sub.beta
        return (1.0 - self.coarse_alpha) * local + self.coarse_alpha * coarse

    def __repr__(self) -> str:
        return (
            f"DDELMCoarse(n_subdomains={self.n_subdomains}, "
            f"hidden_dim={self.hidden_dim}, "
            f"coarse_hidden_dim={self.coarse_hidden_dim}, "
            f"coarse_alpha={self.coarse_alpha})"
        )
