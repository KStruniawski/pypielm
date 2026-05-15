"""PyTorch Dataset / DataLoader adapter.

Wraps a standard ``torch.utils.data.Dataset`` to produce a
:class:`~pypielm.data.dataset.PIELMDataset`.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.utils.data

from pypielm.data.dataset import PIELMDataset

_VALID_ROLES = {"colloc", "bc", "ic", "data"}


class TorchDatasetAdapter:
    """Convert a :class:`torch.utils.data.Dataset` into a :class:`~pypielm.data.dataset.PIELMDataset`.

    The ``torch.utils.data.Dataset`` must return ``(x, y)`` tuples where *x*
    are spatial coordinates and *y* are target values.  All items are loaded
    eagerly into memory.

    Args:
        dataset: Source PyTorch dataset returning ``(x, y)`` pairs.
        role: Determines which field of :class:`~pypielm.data.dataset.PIELMDataset`
            the loaded points populate.  One of ``'colloc'``, ``'bc'``,
            ``'ic'``, ``'data'``.
        dtype: Tensor dtype.
        device: Target device.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        role: str = "data",
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> None:
        if role not in _VALID_ROLES:
            raise ValueError(
                f"role must be one of {sorted(_VALID_ROLES)}, got '{role}'."
            )
        self.dataset = dataset
        self.role = role
        self.dtype = dtype
        self.device = device

    def load(self) -> PIELMDataset:
        """Eagerly load all items and return a :class:`~pypielm.data.dataset.PIELMDataset`."""
        xs: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []

        for item in self.dataset:
            if not (isinstance(item, (tuple, list)) and len(item) == 2):
                raise ValueError(
                    "Each item from the dataset must be a (x, y) tuple."
                )
            x, y = item
            xs.append(torch.as_tensor(x, dtype=self.dtype))
            ys.append(torch.as_tensor(y, dtype=self.dtype))

        if not xs:
            raise ValueError("Dataset is empty.")

        X = torch.stack(xs).to(self.device)
        Y = torch.stack(ys).to(self.device)

        if X.ndim == 1:
            X = X.unsqueeze(1)
        if Y.ndim == 1:
            Y = Y.unsqueeze(1)

        kwargs: dict[str, Any] = {}
        if self.role == "colloc":
            kwargs = {"X_colloc": X}
        elif self.role == "bc":
            kwargs = {"X_colloc": X, "X_bc": X, "y_bc": Y}
        elif self.role == "ic":
            kwargs = {"X_colloc": X, "X_ic": X, "y_ic": Y}
        else:  # "data"
            kwargs = {"X_colloc": X, "y_data": Y}

        return PIELMDataset(**kwargs)
