"""NumPy .npz file adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pypielm.data.dataset import PIELMDataset

_FIELD_KEYS = ("X_colloc", "X_bc", "y_bc", "X_ic", "y_ic", "X_data", "y_data")


class NPZAdapter:
    """Load a :class:`~pypielm.data.dataset.PIELMDataset` from a NumPy ``.npz`` file.

    Expected keys in the archive: ``'X_colloc'``, and optionally
    ``'X_bc'``, ``'y_bc'``, ``'X_ic'``, ``'y_ic'``, ``'X_data'``, ``'y_data'``.

    When no ``'X_colloc'`` key is present but the archive has exactly two
    arrays, the first is treated as ``X_colloc`` and the second as ``y_data``.

    Args:
        path: Path to the ``.npz`` file.
        dtype: Tensor dtype.
        device: Target device.
    """

    def __init__(
        self,
        path: str | Path,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> None:
        self.path = Path(path)
        self.dtype = dtype
        self.device = device

    def load(self) -> PIELMDataset:
        """Load and return the dataset."""
        with np.load(self.path, allow_pickle=False) as archive:
            keys = list(archive.files)
            data = {k: np.asarray(archive[k]) for k in keys}

        def _t(arr: np.ndarray) -> torch.Tensor:
            t = torch.tensor(arr, dtype=self.dtype, device=self.device)
            if t.ndim == 1:
                t = t.unsqueeze(1)
            return t

        if "X_colloc" in data:
            kwargs = {}
            for field in _FIELD_KEYS:
                if field in data:
                    kwargs[field] = _t(data[field])
            return PIELMDataset(**kwargs)  # type: ignore[arg-type]

        # Fallback: two arrays → (X_colloc, y_data)
        if len(keys) == 2:
            return PIELMDataset(
                X_colloc=_t(data[keys[0]]),
                y_data=_t(data[keys[1]]),
            )

        raise ValueError(
            f"NPZ file '{self.path}' has no 'X_colloc' key and does not contain "
            "exactly two arrays.  Provide a file with standard field names."
        )
