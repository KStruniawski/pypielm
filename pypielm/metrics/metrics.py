"""Evaluation metrics for PDE solution accuracy.

All functions operate on :class:`torch.Tensor` or array-likes and return
plain Python ``float`` values for easy logging and comparison.
"""

from __future__ import annotations

import numpy as np
import torch


def _to_tensor(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x


def rmse(y_pred: torch.Tensor | np.ndarray, y_true: torch.Tensor | np.ndarray) -> float:
    """Root Mean Squared Error.

    .. math:: \\text{RMSE} = \\sqrt{\\frac{1}{N} \\|\\hat{u} - u\\|_2^2}

    Args:
        y_pred: Predicted values, shape ``(N,)`` or ``(N, d)``.
        y_true: Reference values, same shape.

    Returns:
        Scalar RMSE.
    """
    p = _to_tensor(y_pred).float()
    t = _to_tensor(y_true).float()
    return float(torch.sqrt(torch.mean((p - t) ** 2)).item())


def mae(y_pred: torch.Tensor | np.ndarray, y_true: torch.Tensor | np.ndarray) -> float:
    """Mean Absolute Error.

    .. math:: \\text{MAE} = \\frac{1}{N} \\|\\hat{u} - u\\|_1

    Args:
        y_pred: Predicted values.
        y_true: Reference values.

    Returns:
        Scalar MAE.
    """
    p = _to_tensor(y_pred).float()
    t = _to_tensor(y_true).float()
    return float(torch.mean(torch.abs(p - t)).item())


def relative_l2(
    y_pred: torch.Tensor | np.ndarray,
    y_true: torch.Tensor | np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Relative L₂ error (also called normalised RMSE in the benchmark).

    .. math::

        \\epsilon_{L_2} = \\frac{\\|\\hat{u} - u\\|_2}{\\|u\\|_2 + \\epsilon}

    Args:
        y_pred: Predicted values.
        y_true: Reference values.
        eps: Small constant to avoid division by zero.

    Returns:
        Scalar relative L₂ error.
    """
    p = _to_tensor(y_pred).float()
    t = _to_tensor(y_true).float()
    return float((torch.norm(p - t) / (torch.norm(t) + eps)).item())


def max_error(y_pred: torch.Tensor | np.ndarray, y_true: torch.Tensor | np.ndarray) -> float:
    """Maximum absolute pointwise error (L∞ norm).

    .. math:: \\epsilon_{\\infty} = \\max_i |\\hat{u}_i - u_i|

    Args:
        y_pred: Predicted values.
        y_true: Reference values.

    Returns:
        Scalar L∞ error.
    """
    p = _to_tensor(y_pred).float()
    t = _to_tensor(y_true).float()
    return float(torch.max(torch.abs(p - t)).item())


def r2_score(y_pred: torch.Tensor | np.ndarray, y_true: torch.Tensor | np.ndarray) -> float:
    """Coefficient of determination R².

    .. math::

        R^2 = 1 - \\frac{\\text{SS}_\\text{res}}{\\text{SS}_\\text{tot}}
            = 1 - \\frac{\\|u - \\hat{u}\\|_2^2}{\\|u - \\bar{u}\\|_2^2}

    Args:
        y_pred: Predicted values.
        y_true: Reference values.

    Returns:
        Scalar R² (1.0 is perfect, can be negative for bad models).
    """
    p = _to_tensor(y_pred).float()
    t = _to_tensor(y_true).float()
    ss_res = torch.sum((t - p) ** 2)
    ss_tot = torch.sum((t - t.mean()) ** 2)
    return float((1.0 - ss_res / (ss_tot + 1e-12)).item())


class MetricsBundle:
    """Compute and store all standard metrics in one call.

    Args:
        y_pred: Predicted values, shape ``(N,)`` or ``(N, d)``.
        y_true: Ground truth values, same shape.

    Attributes:
        rmse: Root mean squared error.
        mae: Mean absolute error.
        rel_l2: Relative L₂ error.
        max_err: Maximum absolute error.
        r2: Coefficient of determination.

    Example::

        mb = MetricsBundle(model.predict(X_test), y_test)
        print(mb)
    """

    def __init__(self, y_pred: torch.Tensor | np.ndarray, y_true: torch.Tensor | np.ndarray) -> None:
        self.rmse: float = rmse(y_pred, y_true)
        self.mae: float = mae(y_pred, y_true)
        self.rel_l2: float = relative_l2(y_pred, y_true)
        self.max_err: float = max_error(y_pred, y_true)
        self.r2: float = r2_score(y_pred, y_true)

    def to_dict(self) -> dict[str, float]:
        """Return metrics as a plain dictionary."""
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "rel_l2": self.rel_l2,
            "max_err": self.max_err,
            "r2": self.r2,
        }

    def __repr__(self) -> str:
        return (
            f"MetricsBundle("
            f"rmse={self.rmse:.4e}, "
            f"rel_l2={self.rel_l2:.4e}, "
            f"r2={self.r2:.4f})"
        )
