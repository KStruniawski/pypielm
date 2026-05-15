"""Data preprocessing transforms.

Provides:
- :class:`Normalizer` — min-max or z-score normalisation (fit on train only).
- :class:`FeatureExpander` — polynomial / trigonometric feature augmentation.
- :class:`Pipeline` — sequential composition of transforms.
"""

from __future__ import annotations

from typing import Any

import torch


class Normalizer:
    """Min-max or z-score normalisation of tensors.

    Args:
        method: ``'minmax'`` (scale to [0, 1]) or ``'zscore'`` (zero mean,
            unit variance).
        eps: Small constant added to denominator to avoid division by zero.
    """

    def __init__(
        self,
        method: str = "minmax",
        eps: float = 1e-8,
    ) -> None:
        if method not in ("minmax", "zscore"):
            raise ValueError(f"method must be 'minmax' or 'zscore', got '{method}'.")
        self.method = method
        self.eps = eps
        self._fitted = False
        # Populated by fit()
        self._loc: torch.Tensor | None = None   # min  or mean
        self._scale: torch.Tensor | None = None  # range or std

    # ------------------------------------------------------------------
    def fit(self, X: torch.Tensor) -> "Normalizer":
        """Compute normalisation statistics from *X* (rows = samples)."""
        if X.ndim == 1:
            X = X.unsqueeze(1)
        if self.method == "minmax":
            self._loc = X.min(dim=0).values
            self._scale = (X.max(dim=0).values - self._loc).clamp(min=self.eps)
        else:  # zscore
            self._loc = X.mean(dim=0)
            self._scale = X.std(dim=0, unbiased=False).clamp(min=self.eps)
        self._fitted = True
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Apply normalisation (must call :meth:`fit` first)."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        if X.ndim == 1:
            X = X.unsqueeze(1)
        return (X - self._loc.to(X.device)) / self._scale.to(X.device)

    def inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Reverse the normalisation."""
        if not self._fitted:
            raise RuntimeError("Call fit() before inverse_transform().")
        if X.ndim == 1:
            X = X.unsqueeze(1)
        return X * self._scale.to(X.device) + self._loc.to(X.device)

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Fit and transform in one call."""
        return self.fit(X).transform(X)


class FeatureExpander:
    """Expand input features with polynomial or trigonometric terms.

    Useful for augmenting low-dimensional spatial coordinates before passing
    them to a model.

    Args:
        degree: Polynomial degree for monomial expansion (1 = keep original).
        trig: If ``True``, append ``sin`` and ``cos`` of each original feature.

    The expansion is stateless — ``fit_transform`` and ``transform`` behave
    identically.  No ``fit()`` call is needed.
    """

    def __init__(
        self,
        degree: int = 1,
        trig: bool = False,
    ) -> None:
        if degree < 1:
            raise ValueError(f"degree must be >= 1, got {degree}.")
        self.degree = degree
        self.trig = trig

    def _expand(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim == 1:
            X = X.unsqueeze(1)
        parts = [X]
        for p in range(2, self.degree + 1):
            parts.append(X ** p)
        if self.trig:
            parts.append(torch.sin(X))
            parts.append(torch.cos(X))
        return torch.cat(parts, dim=-1)

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Return augmented feature matrix."""
        return self._expand(X)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`fit_transform` (stateless)."""
        return self._expand(X)


class Pipeline:
    """Sequential composition of transform steps.

    Args:
        steps: List of transform objects.  Each must expose at least one of:
            - ``fit_transform(X)`` (called on the first invocation)
            - ``transform(X)``     (called on subsequent invocations)

    On the first call to :meth:`fit_transform` each step's ``fit_transform``
    is used.  Subsequent calls to :meth:`transform` use each step's
    ``transform`` method (falling back to ``fit_transform`` for stateless
    steps that lack it).
    """

    def __init__(self, steps: list[Any]) -> None:
        if not steps:
            raise ValueError("Pipeline requires at least one step.")
        self.steps = list(steps)

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Apply all steps sequentially using their ``fit_transform`` method."""
        for step in self.steps:
            X = step.fit_transform(X)
        return X

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Apply all fitted steps in sequence."""
        for step in self.steps:
            fn = getattr(step, "transform", None) or getattr(step, "fit_transform")
            X = fn(X)
        return X
