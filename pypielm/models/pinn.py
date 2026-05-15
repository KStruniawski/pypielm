"""Gradient-based PINN variants (baselines for PIELM comparison).

All variants expose the same ``fit / predict / score`` API as the PIELM
models and inherit from :class:`~pypielm.core.base.BasePIELM` for interface
consistency.

* :class:`VanillaPINN`           — Standard MLP with Adam / L-BFGS.
* :class:`AdaptivePINN`          — Residual-importance-weighted collocation.
* :class:`FourierPINN`           — Fourier input encoding (Tancik et al., 2020).
* :class:`MuonPINN`              — Muon (momentum-based orthogonal update) optimizer.
* :class:`ResidualAdaptivePINN`  — ResNet backbone + adaptive sampling.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

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
# Activation factory
# ---------------------------------------------------------------------------

_ACTIVATIONS: dict[str, type[nn.Module] | None] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "sin": None,  # handled specially below
    "softplus": nn.Softplus,
    "sigmoid": nn.Sigmoid,
    "elu": nn.ELU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}


class _Sin(nn.Module):
    """Element-wise sin activation."""

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(x)


def _make_activation(name: str) -> nn.Module:
    if name == "sin":
        return _Sin()
    cls = _ACTIVATIONS.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Choose from: {list(_ACTIVATIONS.keys())}."
        )
    return cls()


# ---------------------------------------------------------------------------
# MLP backbone
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    """Fully-connected MLP with configurable hidden layers.

    Args:
        input_dim: Input dimension ``d``.
        layer_dims: Width of each hidden layer.
        output_dim: Output dimension (typically 1 for scalar PDEs).
        activation: Hidden-layer activation name.
        dtype: Parameter dtype.
    """

    def __init__(
        self,
        input_dim: int,
        layer_dims: list[int],
        output_dim: int,
        activation: str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        act_fn = _make_activation(activation)
        dims = [input_dim] + list(layer_dims) + [output_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], dtype=dtype))
            if i < len(dims) - 2:
                layers.append(act_fn if i == 0 else _make_activation(activation))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# ResNet block for ResidualAdaptivePINN
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    """Single residual block: ``y = x + act(W2 act(W1 x))``."""

    def __init__(self, width: int, activation: str, dtype: torch.dtype) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, width, dtype=dtype)
        self.fc2 = nn.Linear(width, width, dtype=dtype)
        self.act = _make_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fc2(self.act(self.fc1(x)))


class _ResNet(nn.Module):
    """ResNet backbone: lift → N residual blocks → project."""

    def __init__(
        self,
        input_dim: int,
        width: int,
        n_blocks: int,
        output_dim: int,
        activation: str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.lift = nn.Linear(input_dim, width, dtype=dtype)
        self.act = _make_activation(activation)
        self.blocks = nn.ModuleList(
            [_ResBlock(width, activation, dtype) for _ in range(n_blocks)]
        )
        self.proj = nn.Linear(width, output_dim, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        h = self.act(self.lift(x))
        for blk in self.blocks:
            h = blk(h)
        return self.proj(h)


# ---------------------------------------------------------------------------
# Shared loss computation
# ---------------------------------------------------------------------------

def _pinn_loss(
    net: nn.Module,
    dataset: PIELMDataset,
    pde_operator: Any | None,
    bcs: list[Any] | None,
    ics: list[Any] | None,
    w_pde: float,
    w_bc: float,
    w_ic: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, dict[str, float]]:
    """Compute total PINN loss and per-term breakdown.

    PDE residual uses :func:`torch.autograd.grad` on the network output directly.
    BC/IC terms use supervised MSE on the network predictions at boundary points.

    Returns:
        ``(total_loss, {'pde': ..., 'bc': ..., 'ic': ..., 'data': ...})``
    """
    loss_terms: dict[str, float] = {}
    total = torch.zeros(1, dtype=dtype, device=device)

    # ---- PDE interior residual ----
    if pde_operator is not None and dataset.X_colloc is not None:
        X_c = _ensure_tensor(dataset.X_colloc, dtype, device).requires_grad_(True)
        u_pred = net(X_c)

        # Build a feature-map-compatible wrapper so pde_operator(fm, X) still works.
        # PINNs use an ad-hoc wrapper that computes d1/d2 via autograd.
        class _AutogradFM:
            hidden_dim = u_pred.shape[-1]
            input_dim = X_c.shape[-1]

            def __call__(self, X: Tensor) -> Tensor:  # noqa: N803
                return net(X)

            def forward(self, X: Tensor) -> Tensor:  # noqa: N803
                return net(X)

            def d1(self, X: Tensor, axis: int) -> Tensor:
                X = X.requires_grad_(True)
                u = net(X)
                grad = torch.autograd.grad(
                    u.sum(), X, create_graph=True
                )[0]
                return grad[:, axis : axis + 1]

            def d2(self, X: Tensor, axis: int) -> Tensor:
                X = X.requires_grad_(True)
                u = net(X)
                g1 = torch.autograd.grad(u.sum(), X, create_graph=True)[0]
                g2 = torch.autograd.grad(
                    g1[:, axis].sum(), X, create_graph=True
                )[0]
                return g2[:, axis : axis + 1]

            def laplacian(self, X: Tensor, dims: list[int] | None = None) -> Tensor:
                if dims is None:
                    dims = list(range(X.shape[-1]))
                lap = torch.zeros(X.shape[0], 1, dtype=X.dtype, device=X.device)
                for ax in dims:
                    lap = lap + self.d2(X, ax)
                return lap

        blk = pde_operator(_AutogradFM(), X_c)
        # blk.H = L[u](X_c), blk.y = f(X_c)  (both shape (N, 1) or broadcast)
        # PDE residual loss: MSE(L[u] - f)
        pde_loss = w_pde * blk.weight * nn.functional.mse_loss(
            blk.H, blk.y.expand_as(blk.H)
        )
        total = total + pde_loss
        loss_terms["pde"] = pde_loss.item()

    _has_bc = bool(bcs) or (dataset.X_bc is not None and dataset.y_bc is not None)
    _has_ic = bool(ics) or (dataset.X_ic is not None and dataset.y_ic is not None)

    # ---- Boundary conditions ----
    bc_loss = torch.zeros(1, dtype=dtype, device=device)
    if bcs:
        for bc in bcs:
            X_bc = _ensure_tensor(bc.points, dtype, device)
            u_bc = net(X_bc)
            y_bc = _ensure_tensor(bc.values, dtype, device)
            bc_loss = bc_loss + nn.functional.mse_loss(u_bc, _ensure_2d(y_bc))
    elif dataset.X_bc is not None and dataset.y_bc is not None:
        X_bc = _ensure_tensor(dataset.X_bc, dtype, device)
        y_bc = _ensure_2d(_ensure_tensor(dataset.y_bc, dtype, device))
        u_bc = net(X_bc)
        bc_loss = bc_loss + nn.functional.mse_loss(u_bc, y_bc)
    if _has_bc:
        bc_loss = w_bc * bc_loss
        total = total + bc_loss
        loss_terms["bc"] = bc_loss.item()

    # ---- Initial conditions ----
    ic_loss = torch.zeros(1, dtype=dtype, device=device)
    if ics:
        for ic in ics:
            X_ic = _ensure_tensor(ic.points, dtype, device)
            u_ic = net(X_ic)
            y_ic = _ensure_tensor(ic.values, dtype, device)
            ic_loss = ic_loss + nn.functional.mse_loss(u_ic, _ensure_2d(y_ic))
    elif dataset.X_ic is not None and dataset.y_ic is not None:
        X_ic = _ensure_tensor(dataset.X_ic, dtype, device)
        y_ic = _ensure_2d(_ensure_tensor(dataset.y_ic, dtype, device))
        u_ic = net(X_ic)
        ic_loss = ic_loss + nn.functional.mse_loss(u_ic, y_ic)
    if _has_ic:
        ic_loss = w_ic * ic_loss
        total = total + ic_loss
        loss_terms["ic"] = ic_loss.item()

    # ---- Observation data ----
    if dataset.X_data is not None and dataset.y_data is not None:
        X_d = _ensure_tensor(dataset.X_data, dtype, device)
        y_d = _ensure_2d(_ensure_tensor(dataset.y_data, dtype, device))
        data_loss = nn.functional.mse_loss(net(X_d), y_d)
        total = total + data_loss
        loss_terms["data"] = data_loss.item()

    return total, loss_terms


# ---------------------------------------------------------------------------
# Base gradient-PINN mixin: shared fit / predict / score logic
# ---------------------------------------------------------------------------

class _GradPINNBase(BasePIELM):
    """Internal mixin that provides the shared training loop for all PINN variants.

    Subclasses must:
    1. Set ``self._net`` (an ``nn.Module``) before calling ``_train``.
    2. Optionally override ``_build_optimizer`` or ``_pre_step_hook``.
    """

    # training bookkeeping
    _loss_history: list[float]
    _net: nn.Module | None  # set lazily in subclass __init__ / fit()

    def _build_optimizer(self, lr: float, optimizer_name: str) -> torch.optim.Optimizer:
        assert self._net is not None
        if optimizer_name == "lbfgs":
            return torch.optim.LBFGS(
                self._net.parameters(),
                lr=lr,
                max_iter=20,
                tolerance_grad=1e-9,
                tolerance_change=1e-12,
                history_size=50,
                line_search_fn="strong_wolfe",
            )
        return torch.optim.Adam(self._net.parameters(), lr=lr)

    def _train(
        self,
        dataset: PIELMDataset,
        pde_operator: Any | None,
        bcs: list[Any] | None,
        ics: list[Any] | None,
        optimizer_name: str,
        lr: float,
        max_epochs: int,
        w_pde: float,
        w_bc: float,
        w_ic: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Core training loop (Adam or L-BFGS)."""
        assert self._net is not None
        self._net.to(device=device, dtype=dtype)
        opt = self._build_optimizer(lr, optimizer_name)
        self._loss_history = []

        if optimizer_name == "lbfgs":
            # L-BFGS requires closure
            for _ in range(max_epochs):
                def closure() -> float:
                    net = self._net
                    assert net is not None
                    opt.zero_grad()
                    loss, _ = _pinn_loss(
                        net, dataset, pde_operator, bcs, ics,
                        w_pde, w_bc, w_ic, dtype, device,
                    )
                    loss.backward()
                    return float(loss)

                loss_val = opt.step(closure)  # type: ignore[arg-type]
                self._loss_history.append(
                    float(loss_val) if loss_val is not None else float("nan")
                )
        else:
            for _ in range(max_epochs):
                opt.zero_grad()
                loss, _ = _pinn_loss(
                    self._net, dataset, pde_operator, bcs, ics,
                    w_pde, w_bc, w_ic, dtype, device,
                )
                loss.backward()
                opt.step()
                self._loss_history.append(loss.item())

    def predict(self, X: Array) -> Tensor:
        assert self._net is not None
        device = next(self._net.parameters()).device
        dtype = next(self._net.parameters()).dtype
        X_t = _ensure_tensor(X, dtype, device)
        if X_t.ndim == 1:
            X_t = X_t.unsqueeze(1)
        self._net.eval()
        with torch.no_grad():
            return self._net(X_t)

    def score(self, X: Array, y: Array, metric: str = "relative_l2") -> float:
        assert self._net is not None
        device = next(self._net.parameters()).device
        dtype = next(self._net.parameters()).dtype
        X_t = _ensure_tensor(X, dtype, device)
        y_t = _ensure_2d(_ensure_tensor(y, dtype, device))
        pred = self.predict(X_t)
        return _compute_metric(pred, y_t, metric)

    def get_feature_matrix(self, X: Array) -> Tensor:
        """Return the last hidden-layer activations as the feature matrix."""
        assert self._net is not None
        device = next(self._net.parameters()).device
        dtype = next(self._net.parameters()).dtype
        X_t = _ensure_tensor(X, dtype, device)
        if X_t.ndim == 1:
            X_t = X_t.unsqueeze(1)
        self._net.eval()
        # Walk up to the penultimate layer
        layers = list(self._net.modules())  # type: ignore[arg-type]
        # For MLP: collect all but last Linear
        linears = [m for m in layers if isinstance(m, nn.Linear)]
        if len(linears) < 2:
            return self.predict(X_t)
        # Run up to but not including the final linear
        with torch.no_grad():
            h = X_t
            for m in self._net.children():
                if hasattr(m, "__iter__"):
                    sub = list(m)  # type: ignore[call-overload]  # m is nn.Sequential
                    for layer in sub[:-1]:
                        h = layer(h)
                else:
                    h = m(h)
                break
        return h


# ---------------------------------------------------------------------------
# VanillaPINN
# ---------------------------------------------------------------------------

@register("vanilla_pinn")
class VanillaPINN(_GradPINNBase):
    """Standard Physics-Informed Neural Network (MLP backbone).

    Trains via **Adam** (default) or **L-BFGS** by minimising a weighted sum of:

    .. math::
        \\mathcal{L} = w_{\\text{pde}}\\,\\mathcal{L}_{\\text{pde}}
                     + w_{\\text{bc}}\\,\\mathcal{L}_{\\text{bc}}
                     + w_{\\text{ic}}\\,\\mathcal{L}_{\\text{ic}}
                     + \\mathcal{L}_{\\text{data}}

    Args:
        layer_dims: Width of each hidden layer, e.g. ``[50, 50, 50]``.
        activation: Hidden activation (``'tanh'``, ``'sin'``, ``'relu'``,
            ``'softplus'``).
        optimizer: ``'adam'`` or ``'lbfgs'``.
        lr: Learning rate for Adam (L-BFGS ignores this; uses line search).
        max_epochs: Maximum number of training epochs / outer L-BFGS iterations.
        w_pde: Weight on PDE residual loss term.
        w_bc: Weight on BC loss term.
        w_ic: Weight on IC loss term.
        seed: Random seed for weight initialisation.
        device: Target device (``'cpu'``, ``'cuda'``, ``'mps'``).
        dtype: Floating-point dtype (``torch.float64`` default).

    Example::

        from pypielm.models import VanillaPINN
        model = VanillaPINN(layer_dims=[64, 64], max_epochs=5000)
        model.fit(dataset, pde_operator=laplacian_op)
        u_pred = model.predict(X_test)
    """

    def __init__(
        self,
        layer_dims: list[int] | None = None,
        activation: str = "tanh",
        optimizer: str = "adam",
        lr: float = 1e-3,
        max_epochs: int = 10_000,
        w_pde: float = 1.0,
        w_bc: float = 1.0,
        w_ic: float = 1.0,
        seed: int = 42,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.layer_dims = layer_dims or [50, 50, 50]
        self.activation = activation
        self.optimizer_name = optimizer
        self.lr = lr
        self.max_epochs = max_epochs
        self.w_pde = w_pde
        self.w_bc = w_bc
        self.w_ic = w_ic
        self.seed = seed
        self._device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self._input_dim: int | None = None
        self._loss_history: list[float] = []
        # _net will be built lazily in fit() once input_dim is known.
        self._net: nn.Module | None = None  # type: ignore[assignment]

    def _build_net(self, input_dim: int) -> nn.Module:
        torch.manual_seed(self.seed)
        return _MLP(
            input_dim=input_dim,
            layer_dims=self.layer_dims,
            output_dim=1,
            activation=self.activation,
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
    ) -> VanillaPINN:
        """Train the PINN on *dataset*.

        Args:
            dataset: :class:`~pypielm.data.PIELMDataset` with collocation,
                boundary, and optionally observation points.
            pde_operator: Callable ``(fm, X_colloc) → WeightedLinearSystem``
                used to evaluate PDE residuals.  When provided, the loss
                includes a PDE term.
            bcs: Explicit boundary condition objects (optional; falls back to
                ``dataset.X_bc / y_bc``).
            ics: Explicit initial condition objects (optional).
            collocation_sampler: Not used by gradient-based PINN (reserved for
                future adaptive variants).

        Returns:
            ``self``
        """
        input_dim = int(dataset.X_colloc.shape[-1])
        self._net = self._build_net(input_dim)
        self._input_dim = input_dim
        self._train(
            dataset, pde_operator, bcs, ics,
            self.optimizer_name, self.lr, self.max_epochs,
            self.w_pde, self.w_bc, self.w_ic,
            self.dtype, self._device,
        )
        return self

    def predict(self, X: Array) -> Tensor:
        if self._net is None:
            raise RuntimeError("Call fit() before predict().")
        return super().predict(X)

    def score(self, X: Array, y: Array, metric: str = "relative_l2") -> float:
        if self._net is None:
            raise RuntimeError("Call fit() before score().")
        return super().score(X, y, metric)

    def get_feature_matrix(self, X: Array) -> Tensor:
        if self._net is None:
            raise RuntimeError("Call fit() before get_feature_matrix().")
        return super().get_feature_matrix(X)

    def __repr__(self) -> str:
        return (
            f"VanillaPINN(layer_dims={self.layer_dims}, "
            f"activation='{self.activation}', "
            f"optimizer='{self.optimizer_name}', "
            f"max_epochs={self.max_epochs})"
        )


# ---------------------------------------------------------------------------
# AdaptivePINN: residual-importance collocation reweighting
# ---------------------------------------------------------------------------

@register("adaptive_pinn")
class AdaptivePINN(VanillaPINN):
    """PINN with residual-based importance weighting on collocation points.

    After every ``update_every`` Adam steps, collocation points are re-sampled
    from ``n_candidates`` candidates by drawing ``n_colloc`` points with
    probability proportional to the squared PDE residual (Anagnostopoulos et al.,
    2024; Lu et al., 2021 RAR).

    Args:
        n_colloc: Number of collocation points to keep each iteration.
        n_candidates: Candidate pool for residual evaluation.
        update_every: Resampling interval (epochs).
        domain_lb: Lower bound of the sampling domain (tensor or list).
        domain_ub: Upper bound of the sampling domain (tensor or list).
        resample_ratio: Fraction of points replaced at each update.
        **kwargs: Forwarded to :class:`VanillaPINN`.

    Example::

        model = AdaptivePINN(
            n_colloc=500, domain_lb=[0.0], domain_ub=[1.0],
            update_every=100, layer_dims=[64, 64],
        )
        model.fit(dataset, pde_operator=laplacian_op)
    """

    def __init__(
        self,
        *,
        n_colloc: int = 500,
        n_candidates: int = 2000,
        update_every: int = 100,
        domain_lb: list[float] | None = None,
        domain_ub: list[float] | None = None,
        resample_ratio: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n_colloc = n_colloc
        self.n_candidates = n_candidates
        self.update_every = update_every
        self.domain_lb = domain_lb
        self.domain_ub = domain_ub
        self.resample_ratio = resample_ratio

    def fit(
        self,
        dataset: PIELMDataset,
        *,
        pde_operator: Any | None = None,
        bcs: list[Any] | None = None,
        ics: list[Any] | None = None,
        collocation_sampler: Any | None = None,
    ) -> AdaptivePINN:
        """Train with periodic adaptive collocation resampling."""
        input_dim = int(dataset.X_colloc.shape[-1])
        self._net = self._build_net(input_dim)
        self._input_dim = input_dim
        device = self._device
        dtype = self.dtype

        # Infer domain bounds from dataset if not provided
        lb = self.domain_lb
        ub = self.domain_ub
        if lb is None or ub is None:
            X_c = dataset.X_colloc
            lb_t = X_c.min(0).values
            ub_t = X_c.max(0).values
            lb = lb_t.tolist()
            ub = ub_t.tolist()

        lb_t = torch.tensor(lb, dtype=dtype, device=device)
        ub_t = torch.tensor(ub, dtype=dtype, device=device)

        self._net.to(device=device, dtype=dtype)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        self._loss_history = []

        import copy
        current_dataset = copy.copy(dataset)

        for epoch in range(self.max_epochs):
            # Periodic resampling
            if epoch > 0 and epoch % self.update_every == 0 and pde_operator is not None:
                current_dataset = self._resample(
                    current_dataset, pde_operator, lb_t, ub_t, device, dtype
                )

            opt.zero_grad()
            loss, _ = _pinn_loss(
                self._net, current_dataset, pde_operator, bcs, ics,
                self.w_pde, self.w_bc, self.w_ic, dtype, device,
            )
            loss.backward()
            opt.step()
            self._loss_history.append(loss.item())

        return self

    def _resample(
        self,
        dataset: PIELMDataset,
        pde_operator: Any,
        lb: Tensor,
        ub: Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> PIELMDataset:
        """Resample collocation from high-residual regions."""
        import copy


        # Sample candidates
        d = lb.shape[0]
        X_cand = lb + (ub - lb) * torch.rand(
            self.n_candidates, d, dtype=dtype, device=device
        )

        # Evaluate PDE residual magnitude via autograd
        X_cand.requires_grad_(True)
        assert self._net is not None
        self._net.eval()

        # Use the PINN's own network to evaluate the residual
        net = self._net

        class _AutoFM2:
            def __init__(self) -> None:
                self.hidden_dim = 1
                self.input_dim = d

            def __call__(self, X: Tensor) -> Tensor:
                return net(X)

            def forward(self, X: Tensor) -> Tensor:
                return net(X)

            def d1(self, X: Tensor, axis: int) -> Tensor:
                X = X.requires_grad_(True)
                u = net(X)
                g = torch.autograd.grad(u.sum(), X, create_graph=False)[0]
                return g[:, axis : axis + 1]

            def d2(self, X: Tensor, axis: int) -> Tensor:
                X = X.requires_grad_(True)
                u = net(X)
                g1 = torch.autograd.grad(u.sum(), X, create_graph=True)[0]
                g2 = torch.autograd.grad(
                    g1[:, axis].sum(), X, create_graph=False
                )[0]
                return g2[:, axis : axis + 1]

            def laplacian(self, X: Tensor, dims: list[int] | None = None) -> Tensor:
                if dims is None:
                    dims = list(range(d))
                lap = torch.zeros(X.shape[0], 1, dtype=X.dtype, device=X.device)
                for ax in dims:
                    lap = lap + self.d2(X, ax)
                return lap

        with torch.enable_grad():
            blk = pde_operator(_AutoFM2(), X_cand)
            res = (blk.H - blk.y.expand_as(blk.H)).pow(2).sum(1).detach()

        # Importance-weighted resampling
        probs = res / (res.sum() + 1e-30)
        probs_np = probs.cpu().float().numpy()
        import numpy as np
        idx = np.random.choice(self.n_candidates, size=self.n_colloc, replace=False, p=probs_np)
        X_new = X_cand[idx].detach()  # type: ignore[index]

        new_ds = copy.copy(dataset)
        object.__setattr__(new_ds, "X_colloc", X_new)
        return new_ds

    def __repr__(self) -> str:
        return (
            f"AdaptivePINN(layer_dims={self.layer_dims}, "
            f"n_colloc={self.n_colloc}, update_every={self.update_every}, "
            f"max_epochs={self.max_epochs})"
        )


# ---------------------------------------------------------------------------
# FourierPINN: Fourier input encoding (Tancik et al., 2020)
# ---------------------------------------------------------------------------

@register("fourier_pinn")
class FourierPINN(VanillaPINN):
    """PINN with Fourier input encoding (Tancik et al., 2020).

    Replaces the raw coordinate input with a random Fourier feature encoding:

    .. math::
        \\gamma(\\mathbf{x}) = [\\cos(2\\pi\\mathbf{B}\\mathbf{x}),
                                 \\sin(2\\pi\\mathbf{B}\\mathbf{x})]

    where each entry of ``B`` is drawn from
    :math:`\\mathcal{N}(0, \\sigma^2)`.  This lifts the input into a
    :math:`2m`-dimensional space and mitigates spectral bias.

    Args:
        sigma: Standard deviation of the Gaussian frequency matrix.
        n_fourier: Number of Fourier features ``m`` (output dim = ``2m``).
        **kwargs: Forwarded to :class:`VanillaPINN`.

    Example::

        model = FourierPINN(sigma=10.0, n_fourier=64, layer_dims=[64, 64])
        model.fit(dataset, pde_operator=laplacian_op)
    """

    def __init__(
        self,
        *,
        sigma: float = 10.0,
        n_fourier: int = 64,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.sigma = sigma
        self.n_fourier = n_fourier
        self._B: torch.Tensor | None = None

    def _build_net(self, input_dim: int) -> nn.Module:
        torch.manual_seed(self.seed)
        # Fourier lifting: input_dim → 2 * n_fourier
        fourier_dim = 2 * self.n_fourier
        return _MLP(
            input_dim=fourier_dim,
            layer_dims=self.layer_dims,
            output_dim=1,
            activation=self.activation,
            dtype=self.dtype,
        )

    def _fourier_encode(self, X: Tensor) -> Tensor:
        """Apply Fourier feature encoding to X."""
        if self._B is None:
            raise RuntimeError("B matrix not initialised; call fit() first.")
        # X: (N, d), B: (d, n_fourier) → (N, n_fourier)
        proj = 2.0 * math.pi * (X @ self._B)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

    def fit(
        self,
        dataset: PIELMDataset,
        *,
        pde_operator: Any | None = None,
        bcs: list[Any] | None = None,
        ics: list[Any] | None = None,
        collocation_sampler: Any | None = None,
    ) -> FourierPINN:
        """Train FourierPINN — encodes inputs before building the MLP."""
        input_dim = int(dataset.X_colloc.shape[-1])
        torch.manual_seed(self.seed)
        self._B = torch.randn(input_dim, self.n_fourier, dtype=self.dtype) * self.sigma
        self._B = self._B.to(self._device)
        self._net = self._build_net(input_dim)
        self._input_dim = input_dim

        # Wrap the net so it applies Fourier encoding first
        B = self._B  # capture
        net_inner = self._net

        # Wrap inner MLP in a proper nn.Module that registers it as a child
        # so that parameters() is non-empty.
        class _FourierNetModule(nn.Module):
            def __init__(self, inner: nn.Module, B: Tensor) -> None:
                super().__init__()
                self.inner = inner
                # B is a fixed buffer, not a learnable parameter
                self.register_buffer("B", B)

            def forward(self, x: Tensor) -> Tensor:
                proj = 2.0 * math.pi * (x @ self.B)
                h = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
                return self.inner(h)

        self._net = _FourierNetModule(net_inner, B)  # type: ignore[assignment]

        self._train(
            dataset, pde_operator, bcs, ics,
            self.optimizer_name, self.lr, self.max_epochs,
            self.w_pde, self.w_bc, self.w_ic,
            self.dtype, self._device,
        )
        return self

    def __repr__(self) -> str:
        return (
            f"FourierPINN(sigma={self.sigma}, n_fourier={self.n_fourier}, "
            f"layer_dims={self.layer_dims}, max_epochs={self.max_epochs})"
        )


# ---------------------------------------------------------------------------
# MuonPINN: Muon (momentum orthogonal update) optimizer
# ---------------------------------------------------------------------------

class _MuonOptimizer(torch.optim.Optimizer):
    """Simplified Muon optimizer.

    Muon applies Nesterov momentum in the gradient direction, then projects the
    update onto the orthogonal complement of the current weight matrix via
    Newton-Schulz iteration.  This preserves weight "directionality" and has
    been shown to improve training stability for deep networks.

    Reference: Kosson et al. (2024), Bernstein et al. (2024).

    Args:
        params: Iterable of parameters to optimise.
        lr: Learning rate.
        momentum: Nesterov momentum coefficient.
        nesterov: Whether to use Nesterov update (default ``True``).
        ns_steps: Number of Newton-Schulz iterations for orthogonalisation.
    """

    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        defaults = {"lr": lr, "momentum": momentum, "nesterov": nesterov, "ns_steps": ns_steps}
        super().__init__(params, defaults)

    @staticmethod
    def _zeropower_via_newtonschulz(G: Tensor, steps: int = 5) -> Tensor:
        """Approximate G / ||G||_op via Newton-Schulz iteration.

        Converges to the orthogonal factor of G's polar decomposition.
        """
        assert G.ndim == 2
        a, b, c = (3.4445, -4.7750, 2.0315)
        # Normalise to unit spectral norm
        X = G / (G.norm() + 1e-7)
        if G.shape[0] > G.shape[1]:
            X = X.T
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * (A @ A)
            X = a * X + B @ X
        if G.shape[0] > G.shape[1]:
            X = X.T
        return X

    @torch.no_grad()
    def step(self, closure: Any = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g + momentum * buf if nesterov else buf

                if g.ndim == 2:
                    # Orthogonalise only matrix parameters
                    g = self._zeropower_via_newtonschulz(g, steps=ns_steps)
                    scale = max(1, g.shape[0] / g.shape[1]) ** 0.5
                    p.data.add_(g.to(p.dtype), alpha=-lr * scale)
                else:
                    # Bias and 1-D params: plain SGD step
                    p.data.add_(g.to(p.dtype), alpha=-lr)

        return loss  # type: ignore[return-value]  # Tensor at runtime, declared float | None


@register("muon_pinn")
class MuonPINN(VanillaPINN):
    """PINN trained with the Muon (orthogonal momentum) optimizer.

    Muon orthogonalises parameter updates via Newton-Schulz iteration,
    which improves conditioning and reduces loss of rank in weight matrices.

    Args:
        momentum: Nesterov momentum coefficient (default ``0.95``).
        ns_steps: Number of Newton-Schulz iterations (default ``5``).
        **kwargs: Forwarded to :class:`VanillaPINN`.

    Example::

        model = MuonPINN(layer_dims=[64, 64], momentum=0.95, max_epochs=5000)
        model.fit(dataset, pde_operator=laplacian_op)
    """

    def __init__(
        self,
        *,
        momentum: float = 0.95,
        ns_steps: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.momentum = momentum
        self.ns_steps = ns_steps

    def _build_optimizer(self, lr: float, optimizer_name: str) -> torch.optim.Optimizer:
        # Ignore optimizer_name — always use Muon
        assert self._net is not None
        return _MuonOptimizer(
            self._net.parameters(),
            lr=lr,
            momentum=self.momentum,
            ns_steps=self.ns_steps,
        )

    def __repr__(self) -> str:
        return (
            f"MuonPINN(layer_dims={self.layer_dims}, "
            f"momentum={self.momentum}, ns_steps={self.ns_steps}, "
            f"max_epochs={self.max_epochs})"
        )


# ---------------------------------------------------------------------------
# ResidualAdaptivePINN: ResNet backbone + adaptive sampling
# ---------------------------------------------------------------------------

@register("residual_adaptive_pinn")
class ResidualAdaptivePINN(_GradPINNBase):
    """ResNet-backbone PINN with adaptive collocation sampling.

    Combines:

    * A **residual network** (skip connections) backbone for improved gradient
      flow in deep networks.
    * **Residual-adaptive collocation** (RAR; Lu et al., 2021): every
      ``update_every`` epochs, ``n_new`` fresh points are added in high-residual
      regions, capped at ``max_colloc`` total collocation points.

    Args:
        width: Hidden-layer width for all residual blocks.
        n_blocks: Number of residual blocks.
        activation: Activation function name.
        optimizer: ``'adam'`` or ``'lbfgs'``.
        lr: Learning rate.
        max_epochs: Maximum training epochs.
        w_pde: PDE loss weight.
        w_bc: BC loss weight.
        w_ic: IC loss weight.
        n_new: Points added per RAR update.
        update_every: RAR update interval (epochs).
        max_colloc: Maximum collocation pool size.
        n_candidates: Candidate pool for RAR evaluation.
        domain_lb: Lower bound of sampling domain.
        domain_ub: Upper bound of sampling domain.
        seed: Random seed.
        device: Target device.
        dtype: Floating-point dtype.

    Example::

        model = ResidualAdaptivePINN(
            width=64, n_blocks=3, max_epochs=5000,
            domain_lb=[0.0], domain_ub=[1.0],
        )
        model.fit(dataset, pde_operator=laplacian_op)
    """

    def __init__(
        self,
        width: int = 64,
        n_blocks: int = 3,
        activation: str = "tanh",
        optimizer: str = "adam",
        lr: float = 1e-3,
        max_epochs: int = 10_000,
        w_pde: float = 1.0,
        w_bc: float = 1.0,
        w_ic: float = 1.0,
        n_new: int = 20,
        update_every: int = 100,
        max_colloc: int = 2000,
        n_candidates: int = 5000,
        domain_lb: list[float] | None = None,
        domain_ub: list[float] | None = None,
        seed: int = 42,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.width = width
        self.n_blocks = n_blocks
        self.activation = activation
        self.optimizer_name = optimizer
        self.lr = lr
        self.max_epochs = max_epochs
        self.w_pde = w_pde
        self.w_bc = w_bc
        self.w_ic = w_ic
        self.n_new = n_new
        self.update_every = update_every
        self.max_colloc = max_colloc
        self.n_candidates = n_candidates
        self.domain_lb = domain_lb
        self.domain_ub = domain_ub
        self.seed = seed
        self._device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self._loss_history: list[float] = []
        self._net: nn.Module | None = None  # type: ignore[assignment]

    def fit(
        self,
        dataset: PIELMDataset,
        *,
        pde_operator: Any | None = None,
        bcs: list[Any] | None = None,
        ics: list[Any] | None = None,
        collocation_sampler: Any | None = None,
    ) -> ResidualAdaptivePINN:
        """Train with ResNet backbone and RAR collocation refinement."""
        import copy


        input_dim = int(dataset.X_colloc.shape[-1])
        torch.manual_seed(self.seed)
        self._net = _ResNet(
            input_dim=input_dim,
            width=self.width,
            n_blocks=self.n_blocks,
            output_dim=1,
            activation=self.activation,
            dtype=self.dtype,
        ).to(device=self._device, dtype=self.dtype)

        device = self._device
        dtype = self.dtype

        # Determine domain bounds
        lb = self.domain_lb
        ub = self.domain_ub
        if lb is None or ub is None:
            X_c = dataset.X_colloc
            lb = X_c.min(0).values.tolist()
            ub = X_c.max(0).values.tolist()
        lb_t = torch.tensor(lb, dtype=dtype, device=device)
        ub_t = torch.tensor(ub, dtype=dtype, device=device)

        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        current_dataset = copy.copy(dataset)
        self._loss_history = []

        for epoch in range(self.max_epochs):
            # RAR: add high-residual points periodically
            if (
                epoch > 0
                and epoch % self.update_every == 0
                and pde_operator is not None
            ):
                current_dataset = self._rar_update(
                    current_dataset, pde_operator, lb_t, ub_t, device, dtype
                )

            opt.zero_grad()
            loss, _ = _pinn_loss(
                self._net, current_dataset, pde_operator, bcs, ics,
                self.w_pde, self.w_bc, self.w_ic, dtype, device,
            )
            loss.backward()
            opt.step()
            self._loss_history.append(loss.item())

        return self

    def _rar_update(
        self,
        dataset: PIELMDataset,
        pde_operator: Any,
        lb: Tensor,
        ub: Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> PIELMDataset:
        """RAR: add ``n_new`` high-residual candidate points to collocation set."""
        import copy


        d = int(lb.shape[0])
        X_cand = lb + (ub - lb) * torch.rand(
            self.n_candidates, d, dtype=dtype, device=device
        )

        assert self._net is not None
        net = self._net

        class _AutoFM:
            def __init__(self) -> None:
                self.hidden_dim = 1
                self.input_dim = d

            def __call__(self, X: Tensor) -> Tensor:
                return net(X)

            def forward(self, X: Tensor) -> Tensor:
                return net(X)

            def d1(self, X: Tensor, axis: int) -> Tensor:
                X = X.requires_grad_(True)
                u = net(X)
                g = torch.autograd.grad(u.sum(), X, create_graph=False)[0]
                return g[:, axis : axis + 1]

            def d2(self, X: Tensor, axis: int) -> Tensor:
                X = X.requires_grad_(True)
                u = net(X)
                g1 = torch.autograd.grad(u.sum(), X, create_graph=True)[0]
                g2 = torch.autograd.grad(
                    g1[:, axis].sum(), X, create_graph=False
                )[0]
                return g2[:, axis : axis + 1]

            def laplacian(self, X: Tensor, dims: list[int] | None = None) -> Tensor:
                if dims is None:
                    dims = list(range(d))
                lap = torch.zeros(X.shape[0], 1, dtype=X.dtype, device=X.device)
                for ax in dims:
                    lap = lap + self.d2(X, ax)
                return lap

        self._net.eval()  # _net is nn.Module (asserted above)
        # enable_grad: autograd operators (d2 via autograd.grad) need the
        # computational graph even during the RAR evaluation step.
        with torch.enable_grad():
            blk = pde_operator(_AutoFM(), X_cand)
            res = (blk.H - blk.y.expand_as(blk.H)).pow(2).sum(1).detach()

        # Pick top-n_new residual points
        n_add = min(self.n_new, self.n_candidates)
        _, top_idx = res.topk(n_add)
        X_add = X_cand[top_idx].detach()

        # Append to existing collocation (capped at max_colloc)
        X_old = _ensure_tensor(dataset.X_colloc, dtype, device)
        X_new_all = torch.cat([X_old, X_add], dim=0)
        if X_new_all.shape[0] > self.max_colloc:
            X_new_all = X_new_all[-self.max_colloc :]

        new_ds = copy.copy(dataset)
        object.__setattr__(new_ds, "X_colloc", X_new_all)
        return new_ds

    def __repr__(self) -> str:
        return (
            f"ResidualAdaptivePINN(width={self.width}, "
            f"n_blocks={self.n_blocks}, max_epochs={self.max_epochs})"
        )
