"""Linear solvers for ELM output-weight determination (PyTorch).

All solvers accept and return :class:`torch.Tensor` objects and work on both
CPU and CUDA devices.  ``float64`` (double precision) is strongly recommended
for numerically stable PDE solves.
"""

from __future__ import annotations

from typing import NamedTuple, cast

import torch

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

class BayesianSolveResult(NamedTuple):
    """Result of the Bayesian linear solve.

    Attributes:
        beta_mean: Posterior mean of output weights, shape ``(H, out_dim)``.
        beta_cov: Posterior precision matrix (inverse covariance), shape
            ``(H, H)``.  Stored as *precision* (not covariance) for numerical
            stability; invert only when uncertainty quantification is needed.
    """

    beta_mean: torch.Tensor
    beta_cov: torch.Tensor


class WeightedLinearSystem(NamedTuple):
    """One weighted observation block: y ≈ H @ beta + eps.

    Attributes:
        H: Feature / design matrix, shape ``(N, H)``.
        y: Target values, shape ``(N, out_dim)`` or ``(N,)``.
        weight: Observation precision 1/σ².  Rows are scaled by
            ``sqrt(weight)`` before the solve, so higher weight increases the
            influence of this block on the output weights.
    """

    H: torch.Tensor
    y: torch.Tensor
    weight: float = 1.0


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def ridge_solve(
    H: torch.Tensor,
    y: torch.Tensor,
    lam: float = 1e-8,
) -> torch.Tensor:
    """Closed-form ridge regression output weights.

    Solves:

    .. math::

        \\boldsymbol{\\beta} = (\\mathbf{H}^\\top \\mathbf{H} + \\lambda \\mathbf{I})^{-1}
                               \\mathbf{H}^\\top \\mathbf{y}

    using :func:`torch.linalg.solve` for numerical stability.

    Args:
        H: Feature matrix of shape ``(N, H)``.
        y: Target values of shape ``(N, out_dim)`` or ``(N,)``.
        lam: Regularisation strength λ ≥ 0.

    Returns:
        Output weight matrix β of shape ``(H, out_dim)``.
    """
    if y.ndim == 1:
        y = y.unsqueeze(1)
    HtH = H.T @ H  # (H, H)
    Hty = H.T @ y  # (H, out_dim)
    lam_I = lam * torch.eye(HtH.shape[0], dtype=H.dtype, device=H.device)
    A = HtH + lam_I
    # MPS backend lacks aten::_linalg_solve_ex; compute on CPU and move back.
    if A.device.type == "mps":
        A_cpu = A.cpu()
        b_cpu = Hty.cpu()
        result = torch.linalg.solve(A_cpu, b_cpu)
        return cast(torch.Tensor, result.to(device=H.device))
    return cast(torch.Tensor, torch.linalg.solve(A, Hty))


def rrqr_solve(
    H: torch.Tensor,
    y: torch.Tensor,
    tol: float | None = None,
) -> torch.Tensor:
    """Rank-revealing QR (divide-and-conquer SVD) least-squares solve.

    Delegates to :func:`torch.linalg.lstsq` with ``driver='gelsd'``.

    Args:
        H: Feature matrix of shape ``(N, H)``.
        y: Target values of shape ``(N, out_dim)`` or ``(N,)``.
        tol: Rank-truncation tolerance.

    Returns:
        Minimum-norm least-squares solution β of shape ``(H, out_dim)``.
    """
    if y.ndim == 1:
        y = y.unsqueeze(1)
    kwargs: dict = {"driver": "gelsd"}
    if tol is not None:
        kwargs["rcond"] = tol
    result = torch.linalg.lstsq(H, y, **kwargs)
    return cast(torch.Tensor, result.solution)


def bayesian_solve(
    blocks: list[WeightedLinearSystem],
    prior_precision: float = 1e-4,
) -> BayesianSolveResult:
    """Sequential Bayesian update over weighted observation blocks.

    Computes the posterior:

    .. math::

        \\boldsymbol{\\Lambda}_{\\text{post}} =
            \\alpha \\mathbf{I} + \\sum_k \\lambda_k \\mathbf{H}_k^\\top \\mathbf{H}_k

        \\boldsymbol{\\beta}_{\\text{post}} =
            \\boldsymbol{\\Lambda}_{\\text{post}}^{-1}
            \\sum_k \\lambda_k \\mathbf{H}_k^\\top \\mathbf{y}_k

    Args:
        blocks: List of :class:`WeightedLinearSystem` objects.
        prior_precision: Scalar α for the isotropic prior β ~ N(0, α⁻¹ I).

    Returns:
        :class:`BayesianSolveResult` with ``beta_mean`` and ``beta_cov``
        (posterior precision matrix).
    """
    if not blocks:
        raise ValueError("blocks must be a non-empty list.")

    # Infer hidden_dim from first block
    m = blocks[0].H.shape[1]
    device = blocks[0].H.device
    dtype = blocks[0].H.dtype

    precision = prior_precision * torch.eye(m, dtype=dtype, device=device)
    rhs = torch.zeros(m, 1, dtype=dtype, device=device)

    for blk in blocks:
        H = blk.H
        y = blk.y
        if y.ndim == 1:
            y = y.unsqueeze(1)
        w = float(blk.weight)
        precision = precision + w * (H.T @ H)
        rhs = rhs + w * (H.T @ y)

    # Cholesky for SPD precision matrix (more stable).
    # Fall back to torch.linalg.solve on devices that don't support Cholesky (e.g. MPS).
    try:
        L = torch.linalg.cholesky(precision)
        beta_mean = torch.cholesky_solve(rhs, L)
    except (torch.linalg.LinAlgError, NotImplementedError):
        # NotImplementedError: MPS lacks aten::linalg_cholesky_ex; use general solve.
        # LinAlgError: near-singular — add jitter and retry with general solve.
        beta_mean = torch.linalg.solve(precision, rhs)

    return BayesianSolveResult(beta_mean=beta_mean, beta_cov=precision)


def tikhonov_solve(
    H: torch.Tensor,
    y: torch.Tensor,
    L: torch.Tensor,
    lam: float = 1e-8,
) -> torch.Tensor:
    """Generalised Tikhonov regularisation.

    Solves:

    .. math::

        \\boldsymbol{\\beta} =
            (\\mathbf{H}^\\top \\mathbf{H} + \\lambda \\mathbf{L}^\\top \\mathbf{L})^{-1}
            \\mathbf{H}^\\top \\mathbf{y}

    Args:
        H: Feature matrix of shape ``(N, H)``.
        y: Target values of shape ``(N, out_dim)`` or ``(N,)``.
        L: Regularisation matrix of shape ``(r, H)``.
        lam: Regularisation strength λ.

    Returns:
        Output weight matrix β of shape ``(H, out_dim)``.
    """
    if y.ndim == 1:
        y = y.unsqueeze(1)
    HtH = H.T @ H  # (H, H)
    Hty = H.T @ y  # (H, out_dim)
    LtL = L.T @ L  # (H, H)
    reg = HtH + lam * LtL
    return cast(torch.Tensor, torch.linalg.solve(reg, Hty))
