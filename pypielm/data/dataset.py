"""Core dataset container for PyPIELM.

:class:`PIELMDataset` is the canonical input to every ``model.fit()`` call.
It holds collocation, boundary, initial condition, and optional observation
tensors and exposes convenience constructors and device-migration helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


def _to_tensor(
    x: Any,
    dtype: torch.dtype,
    device: str | torch.device,
) -> torch.Tensor:
    """Convert array-like *x* to a 2-D ``(N, d)`` tensor."""
    if isinstance(x, torch.Tensor):
        t = x.to(dtype=dtype, device=device)
    else:
        import numpy as np  # numpy is a hard dep of PyTorch anyway
        t = torch.tensor(np.asarray(x), dtype=dtype, device=device)
    if t.ndim == 1:
        t = t.unsqueeze(1)
    return t


def _maybe_tensor(
    x: Any | None,
    dtype: torch.dtype,
    device: str | torch.device,
) -> torch.Tensor | None:
    return None if x is None else _to_tensor(x, dtype, device)


@dataclass
class PIELMDataset:
    """Dataset container passed to :meth:`~pypielm.core.base.BasePIELM.fit`.

    Fields:

    * ``X_colloc``  — Interior collocation points ``(N_pde, d)``.
    * ``X_bc``      — Boundary condition points ``(N_bc, d)``  (optional).
    * ``y_bc``      — BC target values ``(N_bc, m)``            (optional).
    * ``X_ic``      — Initial condition points ``(N_ic, d)``   (optional).
    * ``y_ic``      — IC target values ``(N_ic, m)``           (optional).
    * ``X_data``    — Observed data points ``(N_obs, d)``      (optional).
    * ``y_data``    — Observed target values ``(N_obs, m)``    (optional).
    * ``meta``      — Free-form metadata dict.
    """

    X_colloc: torch.Tensor

    X_bc: torch.Tensor | None = None
    y_bc: torch.Tensor | None = None

    X_ic: torch.Tensor | None = None
    y_ic: torch.Tensor | None = None

    X_data: torch.Tensor | None = None
    y_data: torch.Tensor | None = None

    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        X_colloc: Any,
        *,
        X_bc: Any | None = None,
        y_bc: Any | None = None,
        X_ic: Any | None = None,
        y_ic: Any | None = None,
        X_data: Any | None = None,
        y_data: Any | None = None,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
        meta: dict[str, Any] | None = None,
    ) -> PIELMDataset:
        """Construct a :class:`PIELMDataset` from numpy arrays or lists.

        Args:
            X_colloc: Interior collocation points, array-like ``(N, d)``.
            X_bc: Boundary points.
            y_bc: Boundary target values.
            X_ic: Initial condition points.
            y_ic: IC target values.
            X_data: Observed data points.
            y_data: Observed data targets.
            dtype: Target tensor dtype.
            device: Target device.
            meta: Optional metadata.

        Returns:
            A new :class:`PIELMDataset` instance.
        """
        return cls(
            X_colloc=_to_tensor(X_colloc, dtype, device),
            X_bc=_maybe_tensor(X_bc, dtype, device),
            y_bc=_maybe_tensor(y_bc, dtype, device),
            X_ic=_maybe_tensor(X_ic, dtype, device),
            y_ic=_maybe_tensor(y_ic, dtype, device),
            X_data=_maybe_tensor(X_data, dtype, device),
            y_data=_maybe_tensor(y_data, dtype, device),
            meta=meta if meta is not None else {},
        )

    # ------------------------------------------------------------------
    # Device migration
    # ------------------------------------------------------------------

    def to(
        self,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
    ) -> PIELMDataset:
        """Move all tensors to *device* (and optionally cast to *dtype*).

        Returns a new :class:`PIELMDataset`; the original is unchanged.
        """
        def _move(t: torch.Tensor | None) -> torch.Tensor | None:
            if t is None:
                return None
            if dtype is not None:
                return t.to(device=device, dtype=dtype)
            return t.to(device=device)

        return PIELMDataset(
            X_colloc=_move(self.X_colloc),  # type: ignore[arg-type]
            X_bc=_move(self.X_bc),
            y_bc=_move(self.y_bc),
            X_ic=_move(self.X_ic),
            y_ic=_move(self.y_ic),
            X_data=_move(self.X_data),
            y_data=_move(self.y_data),
            meta=dict(self.meta),
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_colloc = self.X_colloc.shape[0] if self.X_colloc is not None else 0
        n_bc     = self.X_bc.shape[0]     if self.X_bc     is not None else 0
        n_ic     = self.X_ic.shape[0]     if self.X_ic     is not None else 0
        n_obs    = self.X_data.shape[0]   if self.X_data   is not None else 0
        return (
            f"PIELMDataset("
            f"colloc={n_colloc}, bc={n_bc}, ic={n_ic}, obs={n_obs})"
        )
