"""Curriculum PIELM: residual-adaptive collocation resampling.

:class:`CurriculumPIELM` iteratively refines the set of collocation points
by concentrating new samples in high-residual regions, progressively improving
accuracy for solutions with localised features (shocks, steep gradients).
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
from pypielm.core.solver import WeightedLinearSystem, ridge_solve
from pypielm.models.registry import register
from pypielm.models.vanilla import _collect_blocks

if TYPE_CHECKING:
    from pypielm.data.dataset import PIELMDataset


@register("curriculum_pielm")
class CurriculumPIELM(BasePIELM):
    """Physics-Informed ELM with curriculum (residual-adaptive) collocation.

    Training proceeds in ``n_stages`` rounds.  In each round:

    1. Solve for β on the current collocation set (ridge solve).
    2. Evaluate PDE residual ``|H_pde @ β − f|`` at a dense candidate set.
    3. Replace ``refine_ratio`` fraction of collocation points with points
       sampled from the high-residual tail of the candidate distribution.
    4. Repeat until ``n_stages`` is reached.

    Args:
        hidden_dim: Number of random neurons.
        ridge_lambda: Regularisation strength.
        activation: Activation name.
        n_stages: Number of curriculum refinement rounds.
        n_collocation: Number of collocation points per stage.
        n_candidates: Number of dense candidate points used for residual evaluation.
        refine_ratio: Fraction of collocation points replaced each stage.
        seed: Random seed.
        device: Target device.
        dtype: Floating-point dtype.
    """

    def __init__(
        self,
        hidden_dim: int = 200,
        ridge_lambda: float = 1e-8,
        activation: str = "tanh",
        n_stages: int = 5,
        n_collocation: int = 1000,
        n_candidates: int = 5000,
        refine_ratio: float = 0.5,
        seed: int = 42,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ridge_lambda = ridge_lambda
        self.activation = activation
        self.n_stages = n_stages
        self.n_collocation = n_collocation
        self.n_candidates = n_candidates
        self.refine_ratio = refine_ratio
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
        dataset: "PIELMDataset",
        *,
        pde_operator: Any | None = None,
        bcs: list[Any] | None = None,
        ics: list[Any] | None = None,
        collocation_sampler: Any | None = None,
    ) -> "CurriculumPIELM":
        input_dim = dataset.X_colloc.shape[1]
        if self._fm is None or self._fm.input_dim != input_dim:
            self._fm = self._build_fm(input_dim)

        # Current collocation points (start from dataset.X_colloc)
        X_coll = _ensure_tensor(dataset.X_colloc, self.dtype, self._device)

        # Infer domain bounds from collocation points for candidate generation
        x_lo = X_coll.min(dim=0).values  # (d,)
        x_hi = X_coll.max(dim=0).values  # (d,)

        # Truncate / pad initial collocation set to n_collocation
        if X_coll.shape[0] > self.n_collocation:
            idx = torch.randperm(X_coll.shape[0], device=self._device)[:self.n_collocation]
            X_coll = X_coll[idx]

        rng = torch.Generator(device=self._device)
        rng.manual_seed(self.seed)

        for stage in range(self.n_stages):
            # --- build a temporary dataset for this stage ---
            # When X_data is absent the caller used X_colloc as observation
            # locations.  Carry the *original* full X_colloc as X_data so
            # _collect_blocks can build the data block even when X_coll has
            # been sub-sampled for the PDE collocation step.
            from pypielm.data.dataset import PIELMDataset as DS
            x_data_stage = (
                dataset.X_data
                if dataset.X_data is not None
                else (dataset.X_colloc if dataset.y_data is not None else None)
            )
            stage_ds = DS(
                X_colloc=X_coll,
                X_bc=dataset.X_bc,
                y_bc=dataset.y_bc,
                X_ic=dataset.X_ic,
                y_ic=dataset.y_ic,
                X_data=x_data_stage,
                y_data=dataset.y_data,
            )

            blocks = _collect_blocks(
                self._fm, stage_ds, pde_operator, bcs, ics,
                1.0, 1.0, 1.0, self.dtype, self._device,
            )
            if not blocks:
                raise ValueError(
                    "No observation blocks assembled in CurriculumPIELM. "
                    "Provide pde_operator, bcs, ics, or dataset.y_data."
                )

            H_full, y_full = _stack_blocks(blocks)
            beta = ridge_solve(H_full, y_full, self.ridge_lambda)
            self.register_buffer("_beta", beta)

            # --- compute residual at candidate points ---
            if pde_operator is None or stage == self.n_stages - 1:
                # No PDE operator → can't compute PDE residual; skip refinement
                break

            d = input_dim
            # Uniform random candidates in [x_lo, x_hi]
            eps = torch.rand(self.n_candidates, d, generator=rng,
                             device=self._device, dtype=self.dtype)
            X_cand = x_lo + eps * (x_hi - x_lo)

            blk_cand = pde_operator(self._fm, X_cand)
            H_pde_cand = blk_cand.H  # (n_cand, H)
            y_pde_cand = _ensure_2d(blk_cand.y.to(dtype=self.dtype, device=self._device))
            res = (H_pde_cand @ beta - y_pde_cand).abs().mean(dim=1)  # (n_cand,)

            # Replace refine_ratio fraction of collocation points with
            # points sampled proportional to residual magnitude
            n_replace = max(1, int(self.refine_ratio * X_coll.shape[0]))
            probs = res / (res.sum() + 1e-30)
            chosen = torch.multinomial(probs, n_replace, replacement=True, generator=rng)
            X_new = X_cand[chosen]

            # Keep (1 - refine_ratio) of previous collocation points
            n_keep = X_coll.shape[0] - n_replace
            keep_idx = torch.randperm(X_coll.shape[0], generator=rng,
                                      device=self._device)[:n_keep]
            X_coll = torch.cat([X_coll[keep_idx], X_new], dim=0)

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
            f"CurriculumPIELM(hidden_dim={self.hidden_dim}, "
            f"n_stages={self.n_stages}, "
            f"n_collocation={self.n_collocation}, "
            f"refine_ratio={self.refine_ratio})"
        )
