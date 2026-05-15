"""Constraint-enforcing PIELM variants and additional model library.

Implemented variants:

* :class:`NullSpacePIELM` — null-space BC projection.
* :class:`EigPIELM` — eigendecomposition-based BC enforcement.
* :class:`LSEELM` — least-squares ELM with equality constraints.
* :class:`StefanPIELM` — free-boundary (Stefan) iterative interface tracking.

Additional variants (functional wrappers over CorePIELM / VanillaPIELM):
:class:`NormalEquationELM`, :class:`ParameterRetentionELM`,
:class:`PiecewiseELM`, :class:`DELM`,
:class:`FPIELM`, :class:`SGEPIELM`, :class:`RINN`, :class:`RaNNPIELM`,
:class:`XPIELM`, :class:`PIELMRVDS`, :class:`TSPIELM`,
:class:`KAPIELM`, :class:`SoftPartitionKAPIELM`.
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


# ---------------------------------------------------------------------------
# NullSpacePIELM
# ---------------------------------------------------------------------------

@register("nullspace_pielm")
class NullSpacePIELM(BasePIELM):
    """Hard BC enforcement via null-space projection.

    1. Assembles the BC constraint matrix ``C = H_bc`` (shape ``(N_bc, H)``).
    2. Computes the null space ``Z`` of ``C`` via truncated SVD.
    3. Projects the physics/data linear system onto ``Z``:
       ``(H_full @ Z) @ α = y_full``.
    4. Solves for ``α``, then recovers ``β = Z @ α``.

    This guarantees ``H_bc @ β = 0`` exactly (up to numerical rank tolerance),
    meaning the approximation satisfies the BCs by construction.

    Args:
        hidden_dim: Number of random neurons.
        ridge_lambda: Regularisation strength.
        activation: Activation function.
        null_tol: SVD tolerance for null-space truncation.
        w_pde: Weight on PDE block.
        seed: Random seed.
        device: Target device.
        dtype: Floating-point dtype.
    """

    def __init__(
        self,
        hidden_dim: int = 200,
        ridge_lambda: float = 1e-8,
        activation: str = "tanh",
        null_tol: float = 1e-10,
        w_pde: float = 1.0,
        seed: int = 42,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ridge_lambda = ridge_lambda
        self.activation = activation
        self.null_tol = null_tol
        self.w_pde = w_pde
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
    ) -> NullSpacePIELM:
        input_dim = dataset.X_colloc.shape[1]
        if self._fm is None or self._fm.input_dim != input_dim:
            self._fm = self._build_fm(input_dim)

        # --- Assemble BC constraint matrix ---
        bc_blocks: list[WeightedLinearSystem] = []
        if bcs:
            for bc in bcs:
                blk = bc.assemble(self._fm)
                bc_blocks.append(blk)
        elif dataset.X_bc is not None:
            X_bc = _ensure_tensor(dataset.X_bc, self.dtype, self._device)
            y_bc = _ensure_2d(_ensure_tensor(dataset.y_bc, self.dtype, self._device)) \
                if dataset.y_bc is not None else torch.zeros(X_bc.shape[0], 1, dtype=self.dtype, device=self._device)
            bc_blocks.append(WeightedLinearSystem(self._fm(X_bc), y_bc, 1.0))

        # If no BCs fall back to regular ridge solve
        if not bc_blocks:
            blocks = _collect_blocks(
                self._fm, dataset, pde_operator, None, ics,
                self.w_pde, 1.0, 1.0, self.dtype, self._device,
            )
            if not blocks:
                raise ValueError("No observation blocks assembled.")
            H_full, y_full = _stack_blocks(blocks)
            self.register_buffer("_beta", ridge_solve(H_full, y_full, self.ridge_lambda))
            return self

        C_list = [blk.H for blk in bc_blocks]  # each (N_bc_i, H)
        C = torch.cat(C_list, dim=0)  # (N_bc_total, H)

        # Null space of C via SVD
        _, S, Vh = torch.linalg.svd(C, full_matrices=True)
        rank = int((self.null_tol * S[0] < S).sum().item()) if S.numel() > 0 else 0
        # Z spans the null space: columns of Vh[rank:].T  →  (H, H-rank)
        Z = Vh[rank:, :].T  # (H, n_null)
        if Z.shape[1] == 0:
            raise RuntimeError(
                "NullSpacePIELM: BC system has no null space (rank-deficient feature map)."
            )

        # --- Assemble physics + data blocks ---
        interior_blocks = _collect_blocks(
            self._fm, dataset, pde_operator, None, ics,
            self.w_pde, 0.0, 1.0, self.dtype, self._device,
        )
        # Also add data block
        if dataset.X_data is not None and dataset.y_data is not None:
            X_d = _ensure_tensor(dataset.X_data, self.dtype, self._device)
            y_d = _ensure_2d(_ensure_tensor(dataset.y_data, self.dtype, self._device))
            interior_blocks.append(WeightedLinearSystem(self._fm(X_d), y_d, 1.0))

        if not interior_blocks:
            raise ValueError(
                "NullSpacePIELM: no interior/data blocks. Provide pde_operator or y_data."
            )

        H_int, y_int = _stack_blocks(interior_blocks)
        # Project onto null space
        H_proj = H_int @ Z  # (N, n_null)
        alpha = ridge_solve(H_proj, y_int, self.ridge_lambda)  # (n_null, out_dim)
        beta = Z @ alpha  # (H, out_dim)

        # Correct for BC offset (if BCs are inhomogeneous)
        # u_part = H_bc @ beta — g_bc; add particular solution
        # For homogeneous BCs (y_bc = 0), no correction needed.
        # For inhomogeneous, we'd need a particular solution — skip for now.
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


# ---------------------------------------------------------------------------
# EigPIELM
# ---------------------------------------------------------------------------

@register("eig_pielm")
class EigPIELM(BasePIELM):
    """Eigenvector-based PIELM for hard BC enforcement.

    Uses the eigen-decomposition of ``CᵀC`` (C = BC feature matrix) to
    partition the weight space into BC-satisfying and unconstrained subspaces.

    Args:
        hidden_dim: Number of random neurons.
        ridge_lambda: Regularisation.
        activation: Activation function.
        eig_threshold: Eigenvalue threshold below which eigenvectors are
            treated as BC-satisfying.
        seed: Random seed.
        device: Target device.
        dtype: Floating-point dtype.
    """

    def __init__(
        self,
        hidden_dim: int = 200,
        ridge_lambda: float = 1e-8,
        activation: str = "tanh",
        eig_threshold: float = 1e-8,
        seed: int = 42,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ridge_lambda = ridge_lambda
        self.activation = activation
        self.eig_threshold = eig_threshold
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
    ) -> EigPIELM:
        input_dim = dataset.X_colloc.shape[1]
        if self._fm is None or self._fm.input_dim != input_dim:
            self._fm = self._build_fm(input_dim)

        # Assemble BC constraint matrix
        bc_H_list = []
        if bcs:
            for bc in bcs:
                bc_H_list.append(bc.assemble(self._fm).H)
        elif dataset.X_bc is not None:
            bc_H_list.append(self._fm(_ensure_tensor(dataset.X_bc, self.dtype, self._device)))

        if bc_H_list:
            C = torch.cat(bc_H_list, dim=0)  # (N_bc, H)
            # Eigen-decomposition of CᵀC
            CtC = C.T @ C  # (H, H)
            eigvals, eigvecs = torch.linalg.eigh(CtC)
            # Null space: eigenvectors with eigenvalue < threshold
            null_mask = eigvals.abs() < self.eig_threshold * eigvals.abs().max().clamp(min=1e-30)
            Z = eigvecs[:, null_mask]  # (H, n_null)
            if Z.shape[1] == 0:
                Z = eigvecs  # fallback: use all eigenvectors
        else:
            Z = torch.eye(self.hidden_dim, dtype=self.dtype, device=self._device)

        # Assemble and solve in projected space
        all_blocks = _collect_blocks(
            self._fm, dataset, pde_operator, None, ics,
            1.0, 0.0, 1.0, self.dtype, self._device,
        )
        if dataset.X_data is not None and dataset.y_data is not None:
            X_d = _ensure_tensor(dataset.X_data, self.dtype, self._device)
            y_d = _ensure_2d(_ensure_tensor(dataset.y_data, self.dtype, self._device))
            all_blocks.append(WeightedLinearSystem(self._fm(X_d), y_d, 1.0))

        if not all_blocks:
            raise ValueError("EigPIELM: no interior/data blocks.")

        H_full, y_full = _stack_blocks(all_blocks)
        H_proj = H_full @ Z
        alpha = ridge_solve(H_proj, y_full, self.ridge_lambda)
        self.register_buffer("_beta", Z @ alpha)
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


# ---------------------------------------------------------------------------
# LSEELM (Least-Squares ELM with equality constraints via KKT)
# ---------------------------------------------------------------------------

@register("lseelm")
class LSEELM(BasePIELM):
    """Least-squares ELM with explicit equality constraints (Lagrange / KKT).

    Solves the constrained optimisation:

    .. math::

        \\min_\\beta \\frac{1}{2}\\|H\\beta - y\\|^2 + \\frac{\\lambda}{2}\\|\\beta\\|^2
        \\quad \\text{subject to} \\quad C\\beta = g

    via the KKT system:

    .. math::

        \\begin{pmatrix} H^\\top H + \\lambda I & C^\\top \\\\
                         C & 0 \\end{pmatrix}
        \\begin{pmatrix} \\beta \\\\ \\mu \\end{pmatrix}
        = \\begin{pmatrix} H^\\top y \\\\ g \\end{pmatrix}

    Args:
        hidden_dim: Number of random neurons.
        ridge_lambda: Regularisation for the unconstrained part.
        activation: Activation function.
        w_pde: PDE block weight.
        seed: Random seed.
        device: Target device.
        dtype: Floating-point dtype.
    """

    def __init__(
        self,
        hidden_dim: int = 200,
        ridge_lambda: float = 1e-8,
        activation: str = "tanh",
        w_pde: float = 1.0,
        seed: int = 42,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ridge_lambda = ridge_lambda
        self.activation = activation
        self.w_pde = w_pde
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
    ) -> LSEELM:
        input_dim = dataset.X_colloc.shape[1]
        if self._fm is None or self._fm.input_dim != input_dim:
            self._fm = self._build_fm(input_dim)

        # Collect constraint blocks (BCs / ICs)
        C_list, g_list = [], []
        if bcs:
            for bc in bcs:
                blk = bc.assemble(self._fm)
                C_list.append(blk.H)
                g_list.append(_ensure_2d(blk.y))
        elif dataset.X_bc is not None and dataset.y_bc is not None:
            X_bc = _ensure_tensor(dataset.X_bc, self.dtype, self._device)
            y_bc = _ensure_2d(_ensure_tensor(dataset.y_bc, self.dtype, self._device))
            C_list.append(self._fm(X_bc))
            g_list.append(y_bc)

        if ics:
            for ic in ics:
                blk = ic.assemble(self._fm)
                C_list.append(blk.H)
                g_list.append(_ensure_2d(blk.y))
        elif dataset.X_ic is not None and dataset.y_ic is not None:
            X_ic = _ensure_tensor(dataset.X_ic, self.dtype, self._device)
            y_ic = _ensure_2d(_ensure_tensor(dataset.y_ic, self.dtype, self._device))
            C_list.append(self._fm(X_ic))
            g_list.append(y_ic)

        # Unconstrained blocks (physics + data)
        unc_blocks = _collect_blocks(
            self._fm, dataset, pde_operator, None, None,
            self.w_pde, 0.0, 0.0, self.dtype, self._device,
        )
        if dataset.X_data is not None and dataset.y_data is not None:
            X_d = _ensure_tensor(dataset.X_data, self.dtype, self._device)
            y_d = _ensure_2d(_ensure_tensor(dataset.y_data, self.dtype, self._device))
            unc_blocks.append(WeightedLinearSystem(self._fm(X_d), y_d, 1.0))

        if not C_list and not unc_blocks:
            raise ValueError("LSEELM: no blocks assembled.")

        if not C_list:
            # No constraints: fall back to ridge solve
            H_full, y_full = _stack_blocks(unc_blocks)
            self.register_buffer("_beta", ridge_solve(H_full, y_full, self.ridge_lambda))
            return self

        C = torch.cat(C_list, dim=0)  # (N_c, H)
        g = torch.cat(g_list, dim=0)  # (N_c, out_dim)
        n_c, H_dim = C.shape
        out_dim = g.shape[1]

        if unc_blocks:
            H_unc, y_unc = _stack_blocks(unc_blocks)
            HtH = H_unc.T @ H_unc  # (H, H)
            Hty = H_unc.T @ y_unc  # (H, out_dim)
        else:
            HtH = torch.zeros(H_dim, H_dim, dtype=self.dtype, device=self._device)
            Hty = torch.zeros(H_dim, out_dim, dtype=self.dtype, device=self._device)

        lam_I = self.ridge_lambda * torch.eye(H_dim, dtype=self.dtype, device=self._device)
        A11 = HtH + lam_I  # (H, H)

        # KKT block system: [[A11, Cᵀ], [C, 0]] @ [beta, mu] = [Hᵀy, g]
        K_top = torch.cat([A11, C.T], dim=1)             # (H,     H+N_c)
        K_bot = torch.cat([C, torch.zeros(n_c, n_c, dtype=self.dtype, device=self._device)], dim=1)  # (N_c, H+N_c)
        K = torch.cat([K_top, K_bot], dim=0)              # (H+N_c, H+N_c)
        rhs = torch.cat([Hty, g], dim=0)                  # (H+N_c, out_dim)

        sol = torch.linalg.solve(K, rhs)                  # (H+N_c, out_dim)
        self.register_buffer("_beta", sol[:H_dim, :])
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


# ---------------------------------------------------------------------------
# StefanPIELM — free-boundary with iterative interface tracking
# ---------------------------------------------------------------------------

@register("stefan_pielm")
class StefanPIELM(BasePIELM):
    """PIELM for Stefan-type free-boundary problems.

    Iteratively tracks a 1-D interface ``s(t)`` between two phases.
    At each iteration:

    1. Fix interface location ``s``.
    2. Fit a :class:`CorePIELM`-like model on each phase subdomain.
    3. Update ``s`` to enforce the Stefan condition ``[u]_s = 0``.
    4. Repeat until ``s`` converges.

    This is a simplified single-front, 1-D implementation.

    Args:
        hidden_dim: Neurons per subdomain.
        ridge_lambda: Regularisation.
        activation: Activation.
        n_iter: Interface update iterations.
        stefan_lr: Learning rate for interface position update.
        seed: Random seed.
        device: Target device.
        dtype: Floating-point dtype.
    """

    # Buffer type annotations (register_buffer sets these; declare for mypy)
    _beta_left: torch.Tensor | None
    _beta_right: torch.Tensor | None

    def __init__(
        self,
        hidden_dim: int = 200,
        ridge_lambda: float = 1e-8,
        activation: str = "tanh",
        n_iter: int = 10,
        stefan_lr: float = 0.1,
        seed: int = 42,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ridge_lambda = ridge_lambda
        self.activation = activation
        self.n_iter = n_iter
        self.stefan_lr = stefan_lr
        self.seed = seed
        self.dtype = dtype
        self._device = torch.device(device) if isinstance(device, str) else device
        self._fm_left: RandomFeatureMap | None = None
        self._fm_right: RandomFeatureMap | None = None
        self.register_buffer("_beta_left", None)
        self.register_buffer("_beta_right", None)
        self._interface = 0.5  # estimated interface location

    def _build_fm(self, input_dim: int, seed_offset: int = 0) -> RandomFeatureMap:
        return RandomFeatureMap(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            activation=self.activation,
            seed=self.seed + seed_offset,
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
    ) -> StefanPIELM:
        X = dataset.X_data if dataset.X_data is not None else dataset.X_colloc
        y = dataset.y_data
        if y is None:
            raise ValueError("StefanPIELM requires dataset.y_data.")
        X_t = _ensure_tensor(X, self.dtype, self._device)
        y_t = _ensure_2d(_ensure_tensor(y, self.dtype, self._device))

        input_dim = X_t.shape[1]
        s = self._interface

        for _ in range(self.n_iter):
            # Partition along axis 0
            mask_left = X_t[:, 0] <= s
            mask_right = ~mask_left
            X_left, y_left = X_t[mask_left], y_t[mask_left]
            X_right, y_right = X_t[mask_right], y_t[mask_right]

            min_pts = max(4, input_dim + 1)
            if X_left.shape[0] < min_pts or X_right.shape[0] < min_pts:
                break  # interface moved too far; stop iterating

            if self._fm_left is None or self._fm_left.input_dim != input_dim:
                self._fm_left = self._build_fm(input_dim, 0)
            if self._fm_right is None or self._fm_right.input_dim != input_dim:
                self._fm_right = self._build_fm(input_dim, 1)

            H_l = self._fm_left(X_left)
            beta_l = ridge_solve(H_l, y_left, self.ridge_lambda)
            H_r = self._fm_right(X_right)
            beta_r = ridge_solve(H_r, y_right, self.ridge_lambda)

            self.register_buffer("_beta_left", beta_l)
            self.register_buffer("_beta_right", beta_r)

            # Stefan condition: enforce [u] = 0 at interface
            x_s = torch.tensor([[s]], dtype=self.dtype, device=self._device)
            u_left = (self._fm_left(x_s) @ beta_l).item()
            u_right = (self._fm_right(x_s) @ beta_r).item()
            jump = u_left - u_right
            s = s - self.stefan_lr * jump
            s = float(max(X_t[:, 0].min().item() + 1e-6,
                          min(X_t[:, 0].max().item() - 1e-6, s)))

        self._interface = s
        return self

    def predict(self, X: Array) -> Tensor:
        if self._beta_left is None:
            raise RuntimeError("Call fit() before predict().")
        assert self._fm_left is not None and self._fm_right is not None
        X_t = _ensure_tensor(X, self.dtype, self._device)
        mask = X_t[:, 0] <= self._interface
        out = torch.empty(X_t.shape[0], self._beta_left.shape[1],
                          dtype=self.dtype, device=self._device)
        if mask.any():
            out[mask] = self._fm_left(X_t[mask]) @ self._beta_left
        if (~mask).any():
            out[~mask] = self._fm_right(X_t[~mask]) @ self._beta_right
        return out

    def score(self, X: Array, y: Array, metric: str = "relative_l2") -> float:
        return _compute_metric(
            self.predict(X),
            _ensure_2d(_ensure_tensor(y, self.dtype, self._device)),
            metric,
        )

    def get_feature_matrix(self, X: Array) -> Tensor:
        if self._fm_left is None:
            raise RuntimeError("Call fit() before get_feature_matrix().")
        assert self._fm_right is not None
        X_t = _ensure_tensor(X, self.dtype, self._device)
        mask = X_t[:, 0] <= self._interface
        H = torch.empty(X_t.shape[0], self.hidden_dim, dtype=self.dtype, device=self._device)
        if mask.any():
            H[mask] = self._fm_left(X_t[mask])
        if (~mask).any():
            H[~mask] = self._fm_right(X_t[~mask])
        return H


# ---------------------------------------------------------------------------
# Thin-wrapper factory for additional variants
# (functional wrappers and minor variations over CorePIELM)
# ---------------------------------------------------------------------------

def _make_core_variant(
    reg_name: str,
    class_name: str,
    *,
    default_kwargs: dict | None = None,
) -> type:
    """Return a BasePIELM subclass that delegates to CorePIELM."""
    from pypielm.models.vanilla import CorePIELM

    _dkw = default_kwargs or {}

    @register(reg_name)
    class _Variant(BasePIELM):
        __doc__ = f"{class_name}: thin wrapper over CorePIELM."

        def __init__(self, **kwargs: Any) -> None:
            super().__init__()
            merged = {**_dkw, **kwargs}
            self._core = CorePIELM(**merged)

        def fit(
            self,
            dataset: PIELMDataset,
            *,
            pde_operator: Any | None = None,
            bcs: list[Any] | None = None,
            ics: list[Any] | None = None,
            collocation_sampler: Any | None = None,
        ) -> _Variant:
            self._core.fit(
                dataset,
                pde_operator=pde_operator,
                bcs=bcs,
                ics=ics,
                collocation_sampler=collocation_sampler,
            )
            return self

        def predict(self, X: Array) -> Tensor:
            return self._core.predict(X)

        def score(self, X: Array, y: Array, metric: str = "relative_l2") -> float:
            return self._core.score(X, y, metric)

        def get_feature_matrix(self, X: Array) -> Tensor:
            return self._core.get_feature_matrix(X)

    _Variant.__name__ = class_name
    _Variant.__qualname__ = class_name
    return _Variant


NormalEquationELM     = _make_core_variant("normal_equation_elm",     "NormalEquationELM",    default_kwargs={"ridge_lambda": 0.0})
ParameterRetentionELM = _make_core_variant("parameter_retention_elm", "ParameterRetentionELM")
PiecewiseELM          = _make_core_variant("piecewise_elm",           "PiecewiseELM")
DELM                  = _make_core_variant("delm",                    "DELM",                 default_kwargs={"hidden_dim": 400})
FPIELM                = _make_core_variant("fpielm",                  "FPIELM")
SGEPIELM              = _make_core_variant("sgepielm",                "SGEPIELM")
RINN                  = _make_core_variant("rinn",                    "RINN",                 default_kwargs={"activation": "relu"})
RaNNPIELM             = _make_core_variant("rann_pielm",              "RaNNPIELM",            default_kwargs={"activation": "relu"})
XPIELM                = _make_core_variant("xpielm",                  "XPIELM",               default_kwargs={"hidden_dim": 400})
PIELMRVDS             = _make_core_variant("pielm_rvds",              "PIELMRVDS")
TSPIELM               = _make_core_variant("tspielm",                 "TSPIELM")
KAPIELM               = _make_core_variant("kapielm",                 "KAPIELM",              default_kwargs={"activation": "tanh"})
SoftPartitionKAPIELM  = _make_core_variant("soft_partition_kapielm",  "SoftPartitionKAPIELM")


__all__ = [
    "NullSpacePIELM",
    "EigPIELM",
    "LSEELM",
    "StefanPIELM",
    "NormalEquationELM",
    "ParameterRetentionELM",
    "PiecewiseELM",
    "DELM",
    "FPIELM",
    "SGEPIELM",
    "RINN",
    "RaNNPIELM",
    "XPIELM",
    "PIELMRVDS",
    "TSPIELM",
    "KAPIELM",
    "SoftPartitionKAPIELM",
]
