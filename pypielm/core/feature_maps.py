"""Random and Fourier feature maps for ELM hidden layers (PyTorch).

All feature maps are :class:`torch.nn.Module` subclasses with **frozen**
parameters.  They provide both a ``forward`` pass and analytic first/second
partial derivatives (fast paths for tanh and sin/cos) as well as an autograd
fallback for arbitrary activations.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import torch
import torch.nn as nn

ActivationName = Literal["tanh", "sin", "relu", "sigmoid", "softplus"]
FrequencyInit = Literal["uniform", "log_uniform", "auto"]


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------

def _get_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    table: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "tanh": torch.tanh,
        "sin": torch.sin,
        "relu": torch.relu,
        "sigmoid": torch.sigmoid,
        "softplus": nn.functional.softplus,
    }
    if name not in table:
        raise ValueError(
            f"Unknown activation '{name}'. Choose from: {list(table.keys())}"
        )
    return table[name]


class RandomFeatureMap(nn.Module):
    """Random-weight ELM hidden layer with analytic partial derivatives.

    The hidden representation is:

    .. math::

        H_{ij} = \\sigma\\!\\left(\\sum_k x_{ik} W_{kj} + b_j\\right)

    where :math:`\\mathbf{W} \\in \\mathbb{R}^{d \\times H}` and
    :math:`\\mathbf{b} \\in \\mathbb{R}^H` are sampled once at construction and
    **never updated** (frozen buffers).

    Analytic first- and second-order partial derivatives are provided for
    ``'tanh'``, ``'sin'``, ``'relu'``, ``'sigmoid'``, and ``'softplus'``.

    Args:
        input_dim: Spatial dimension ``d`` of the input coordinates.
        hidden_dim: Number of neurons ``H``.
        activation: Name of the activation function.
        seed: Random seed for weight initialisation.
        w_scale: Multiplicative scale applied to the random weights ``W``.
        device: Target device (``'cpu'``, ``'cuda'``, …).
        dtype: Floating-point precision.  ``torch.float64`` recommended.

    Example::

        fm = RandomFeatureMap(input_dim=2, hidden_dim=200, seed=42)
        H   = fm(X)          # shape (N, 200)
        dH  = fm.d1(X, 0)    # ∂H/∂x₀,   shape (N, 200)
        d2H = fm.d2(X, 0)    # ∂²H/∂x₀², shape (N, 200)
    """

    # Buffer type annotations: register_buffer sets these as Tensors, not Modules
    W: torch.Tensor
    b: torch.Tensor

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation: str = "tanh",
        seed: int = 42,
        w_scale: float = 1.0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.activation_name = activation
        self._activation = _get_activation(activation)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)

        W = torch.empty(input_dim, hidden_dim, dtype=dtype).uniform_(
            -w_scale, w_scale, generator=gen
        )
        b = torch.empty(hidden_dim, dtype=dtype).uniform_(0.0, 1.0, generator=gen)

        # Frozen buffers — moved with .to(device) but not in optimizer
        self.register_buffer("W", W)
        self.register_buffer("b", b)
        self.to(device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_tensor(self, X: torch.Tensor) -> torch.Tensor:
        import numpy as np
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        return X.to(device=self.W.device, dtype=self.W.dtype)

    def _preact(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim == 1:
            X = X.unsqueeze(0)
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"X has {X.shape[1]} input dims, expected {self.input_dim}."
            )
        return X @ self.W + self.b  # (N, H)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Compute hidden-layer activations H = σ(X @ W + b).

        Args:
            X: Input coordinates, shape ``(N, d)``.

        Returns:
            Feature matrix H, shape ``(N, H)``.
        """
        X = self._to_tensor(X)
        return self._activation(self._preact(X))

    # ------------------------------------------------------------------
    # Analytic derivatives
    # ------------------------------------------------------------------

    def _d1_activation(self, Z: torch.Tensor) -> torch.Tensor:
        """σ'(Z) element-wise."""
        name = self.activation_name
        if name == "tanh":
            t = torch.tanh(Z)
            return 1.0 - t * t
        if name == "sin":
            return torch.cos(Z)
        if name == "relu":
            return (Z > 0).to(Z.dtype)
        if name == "sigmoid":
            s = torch.sigmoid(Z)
            return s * (1.0 - s)
        if name == "softplus":
            return torch.sigmoid(Z)
        raise ValueError(f"No analytic d1 for activation '{self.activation_name}'.")

    def _d2_activation(self, Z: torch.Tensor) -> torch.Tensor:
        """σ''(Z) element-wise."""
        name = self.activation_name
        if name == "tanh":
            t = torch.tanh(Z)
            d1 = 1.0 - t * t
            return -2.0 * t * d1
        if name == "sin":
            return -torch.sin(Z)
        if name == "relu":
            return torch.zeros_like(Z)
        if name == "sigmoid":
            s = torch.sigmoid(Z)
            d1 = s * (1.0 - s)
            return d1 * (1.0 - 2.0 * s)
        if name == "softplus":
            s = torch.sigmoid(Z)
            return s * (1.0 - s)
        raise ValueError(f"No analytic d2 for activation '{self.activation_name}'.")

    def d1(self, X: torch.Tensor, axis: int) -> torch.Tensor:
        """Analytic first partial derivative ∂H/∂x_{axis}.

        .. math::

            \\frac{\\partial H_{ij}}{\\partial x_{i,\\text{axis}}}
            = \\sigma'(Z_{ij}) \\cdot W_{\\text{axis},j}

        Args:
            X: Input coordinates, shape ``(N, d)``.
            axis: Spatial dimension index (0-based).

        Returns:
            First-derivative matrix, shape ``(N, H)``.
        """
        X = self._to_tensor(X)
        Z = self._preact(X)
        w = self.W[axis, :]  # (H,)
        return self._d1_activation(Z) * w

    def d2(self, X: torch.Tensor, axis: int) -> torch.Tensor:
        """Analytic second partial derivative ∂²H/∂x_{axis}².

        .. math::

            \\frac{\\partial^2 H_{ij}}{\\partial x_{i,\\text{axis}}^2}
            = \\sigma''(Z_{ij}) \\cdot W_{\\text{axis},j}^2

        Args:
            X: Input coordinates, shape ``(N, d)``.
            axis: Spatial dimension index (0-based).

        Returns:
            Second-derivative matrix, shape ``(N, H)``.
        """
        X = self._to_tensor(X)
        Z = self._preact(X)
        w = self.W[axis, :]  # (H,)
        return self._d2_activation(Z) * (w * w)

    def laplacian(
        self,
        X: torch.Tensor,
        dims: list[int] | None = None,
    ) -> torch.Tensor:
        """Compute the Laplacian feature matrix Σ_i ∂²H/∂x_i².

        Args:
            X: Input coordinates, shape ``(N, d)``.
            dims: Dimensions to sum over.  Defaults to all input dims.

        Returns:
            Laplacian feature matrix, shape ``(N, H)``.
        """
        X = self._to_tensor(X)
        if dims is None:
            dims = list(range(self.input_dim))
        Z = self._preact(X)
        sigma_pp = self._d2_activation(Z)  # (N, H)
        w_sq = (self.W[dims, :] ** 2).sum(dim=0)  # (H,)
        return sigma_pp * w_sq

    def __repr__(self) -> str:
        return (
            f"RandomFeatureMap(input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"activation='{self.activation_name}', "
            f"dtype={self.W.dtype}, device={self.W.device})"
        )


# ---------------------------------------------------------------------------
# FourierFeatureMap
# ---------------------------------------------------------------------------

class FourierFeatureMap(nn.Module):
    """Generalised Fourier Feature ELM hidden layer.

    Each hidden neuron computes:

    .. math::

        \\phi_j(\\mathbf{x}) =
            \\sqrt{2} \\cos\\!\\left(
                \\omega_j \\, \\mathbf{w}_j^\\top \\mathbf{x} + b_j
            \\right)

    where :math:`\\mathbf{w}_j` is a random direction, :math:`b_j \\sim
    \\mathcal{U}(0, 2\\pi)` is a random phase, and :math:`\\omega_j` is a
    per-neuron frequency coefficient.

    Analytic derivatives:

    .. math::

        \\frac{\\partial \\phi_j}{\\partial x_i}
            = -\\sqrt{2} \\, \\omega_j w_{ji} \\sin(A_j)

        \\frac{\\partial^2 \\phi_j}{\\partial x_i^2}
            = -\\sqrt{2} \\, (\\omega_j w_{ji})^2 \\cos(A_j)

    where :math:`A_j = \\omega_j \\mathbf{w}_j^\\top \\mathbf{x} + b_j`.

    Args:
        input_dim: Spatial dimension ``d``.
        hidden_dim: Number of neurons ``H``.
        freq_init: Frequency initialisation (``'log_uniform'``, ``'uniform'``).
        freq_min: Minimum frequency.
        freq_max: Maximum frequency.
        seed: Random seed.
        w_scale: Scale of Gaussian weight draw before normalisation.
        normalize_w: If ``True``, unit-normalise each direction vector.
        device: Target device.
        dtype: Floating-point precision.
    """

    # Buffer type annotations: register_buffer sets these as Tensors, not Modules
    W: torch.Tensor
    b: torch.Tensor
    omega: torch.Tensor

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        freq_init: FrequencyInit = "log_uniform",
        freq_min: float = 1.0,
        freq_max: float = 100.0,
        seed: int = 42,
        w_scale: float = 1.0,
        normalize_w: bool = True,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        nn.Module.__init__(self)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.freq_init = freq_init

        if freq_min <= 0 or freq_max <= 0 or freq_max <= freq_min:
            raise ValueError("Require 0 < freq_min < freq_max.")

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)

        W = torch.empty(input_dim, hidden_dim, dtype=dtype).normal_(
            mean=0.0, std=w_scale, generator=gen
        )
        if normalize_w:
            norms = W.norm(dim=0, keepdim=True).clamp(min=1e-12)
            W = W / norms

        b = torch.empty(hidden_dim, dtype=dtype).uniform_(
            0.0, 2.0 * math.pi, generator=gen
        )
        omega = self._sample_omega(
            hidden_dim, freq_init, freq_min, freq_max, dtype, gen
        )

        self.register_buffer("W", W)
        self.register_buffer("b", b)
        self.register_buffer("omega", omega)
        self.to(device)

    @staticmethod
    def _sample_omega(
        n: int,
        method: str,
        lo: float,
        hi: float,
        dtype: torch.dtype,
        gen: torch.Generator,
    ) -> torch.Tensor:
        if method == "uniform":
            return torch.empty(n, dtype=dtype).uniform_(lo, hi, generator=gen)
        if method in ("log_uniform", "auto"):
            log_lo = math.log(lo)
            log_hi = math.log(hi)
            log_vals = torch.empty(n, dtype=dtype).uniform_(
                log_lo, log_hi, generator=gen
            )
            return log_vals.exp()
        raise ValueError(f"Unknown freq_init='{method}'.")

    def _to_tensor(self, X: torch.Tensor) -> torch.Tensor:
        import numpy as np
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        return X.to(device=self.W.device, dtype=self.W.dtype)

    def _preact(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim == 1:
            X = X.unsqueeze(0)
        Z = X @ self.W  # (N, H)
        return Z * self.omega + self.b  # (N, H)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Compute φ(X) = √2 cos(ω (X @ W) + b).

        Args:
            X: Input coordinates, shape ``(N, d)``.

        Returns:
            Feature matrix, shape ``(N, H)``.
        """
        X = self._to_tensor(X)
        return math.sqrt(2.0) * torch.cos(self._preact(X))

    def d1(self, X: torch.Tensor, axis: int) -> torch.Tensor:
        """Analytic ∂φ/∂x_{axis}.

        Args:
            X: Input coordinates, shape ``(N, d)``.
            axis: Spatial dimension index.

        Returns:
            First-derivative matrix, shape ``(N, H)``.
        """
        X = self._to_tensor(X)
        A = self._preact(X)
        coeff = self.omega * self.W[axis, :]  # (H,)
        return -math.sqrt(2.0) * torch.sin(A) * coeff

    def d2(self, X: torch.Tensor, axis: int) -> torch.Tensor:
        """Analytic ∂²φ/∂x_{axis}².

        Args:
            X: Input coordinates, shape ``(N, d)``.
            axis: Spatial dimension index.

        Returns:
            Second-derivative matrix, shape ``(N, H)``.
        """
        X = self._to_tensor(X)
        A = self._preact(X)
        coeff = (self.omega * self.W[axis, :]) ** 2  # (H,)
        return -math.sqrt(2.0) * torch.cos(A) * coeff

    def laplacian(
        self,
        X: torch.Tensor,
        dims: list[int] | None = None,
    ) -> torch.Tensor:
        """Compute Σ_i ∂²φ/∂x_i².

        Args:
            X: Input coordinates, shape ``(N, d)``.
            dims: Dimensions to include.  Defaults to all input dims.

        Returns:
            Laplacian feature matrix, shape ``(N, H)``.
        """
        X = self._to_tensor(X)
        if dims is None:
            dims = list(range(self.input_dim))
        A = self._preact(X)
        coeff = ((self.omega * self.W[dims, :]) ** 2).sum(dim=0)  # (H,)
        return -math.sqrt(2.0) * torch.cos(A) * coeff

    def __repr__(self) -> str:
        return (
            f"FourierFeatureMap(input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"freq_init='{self.freq_init}', "
            f"dtype={self.W.dtype}, device={self.W.device})"
        )


# ---------------------------------------------------------------------------
# AutogradFeatureMap
# ---------------------------------------------------------------------------

class AutogradFeatureMap(nn.Module):
    """Feature map with arbitrary activation, derivatives via autograd.

    Wraps any user-supplied callable activation.  Derivatives are computed
    via :func:`torch.autograd.grad` — slower than analytic but works for any
    smooth activation.

    Args:
        input_dim: Spatial dimension ``d``.
        hidden_dim: Number of neurons ``H``.
        activation_fn: Any differentiable callable ``f: Tensor → Tensor``.
        seed: Random seed.
        w_scale: Weight scale.
        device: Target device.
        dtype: Floating-point precision.
    """

    # Buffer type annotations: register_buffer sets these as Tensors, not Modules
    W: torch.Tensor
    b: torch.Tensor

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation_fn: Callable[[torch.Tensor], torch.Tensor],
        seed: int = 42,
        w_scale: float = 1.0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self._activation_fn = activation_fn

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)

        W = torch.empty(input_dim, hidden_dim, dtype=dtype).uniform_(
            -w_scale, w_scale, generator=gen
        )
        b = torch.empty(hidden_dim, dtype=dtype).uniform_(0.0, 1.0, generator=gen)

        self.register_buffer("W", W)
        self.register_buffer("b", b)
        self.to(device)

    def _to_tensor(self, X: torch.Tensor) -> torch.Tensor:
        import numpy as np
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        return X.to(device=self.W.device, dtype=self.W.dtype)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """H = activation(X @ W + b)."""
        X = self._to_tensor(X)
        if X.ndim == 1:
            X = X.unsqueeze(0)
        Z = X @ self.W + self.b
        return self._activation_fn(Z)

    def d1(self, X: torch.Tensor, axis: int) -> torch.Tensor:
        """Autograd ∂H/∂x_{axis}, shape ``(N, H)``."""
        X = self._to_tensor(X)
        if X.ndim == 1:
            X = X.unsqueeze(0)
        N, H_dim = X.shape[0], self.hidden_dim
        result = torch.zeros(N, H_dim, dtype=X.dtype, device=X.device)
        for i in range(H_dim):
            X_req = X.clone().detach().requires_grad_(True)
            Z = X_req @ self.W + self.b
            h_i = self._activation_fn(Z[:, i : i + 1]).sum()
            (grad,) = torch.autograd.grad(h_i, X_req)
            result[:, i] = grad[:, axis]
        return result

    def d2(self, X: torch.Tensor, axis: int) -> torch.Tensor:
        """Autograd ∂²H/∂x_{axis}², shape ``(N, H)``."""
        X = self._to_tensor(X)
        if X.ndim == 1:
            X = X.unsqueeze(0)
        N, H_dim = X.shape[0], self.hidden_dim
        result = torch.zeros(N, H_dim, dtype=X.dtype, device=X.device)
        for i in range(H_dim):
            X_req = X.clone().detach().requires_grad_(True)
            Z = X_req @ self.W + self.b
            h_i = self._activation_fn(Z[:, i : i + 1]).sum()
            (grad1,) = torch.autograd.grad(h_i, X_req, create_graph=True)
            d1_sum = grad1[:, axis].sum()
            (grad2,) = torch.autograd.grad(d1_sum, X_req)
            result[:, i] = grad2[:, axis]
        return result

    def __repr__(self) -> str:
        return (
            f"AutogradFeatureMap(input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"dtype={self.W.dtype}, device={self.W.device})"
        )
