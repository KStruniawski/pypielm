"""CSV file adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from pypielm.data.dataset import PIELMDataset


class CSVAdapter:
    """Load a :class:`~pypielm.data.dataset.PIELMDataset` from a CSV file.

    By default (when *column_map* is ``None``) the adapter treats all columns
    except the last as ``X_colloc`` and the last column as ``y_data``.

    Args:
        path: Path to the CSV file.
        column_map: Optional dict mapping field names (``'X_colloc'``,
            ``'X_bc'``, ``'y_bc'``, ``'X_ic'``, ``'y_ic'``,
            ``'X_data'``, ``'y_data'``) to lists of column names or
            column indices (0-based integers).  When ``None`` the default
            heuristic applies.
        delimiter: Field delimiter (default ``','``).
        dtype: Target tensor dtype.
        device: Target device.
    """

    def __init__(
        self,
        path: str | Path,
        column_map: dict[str, list[str | int]] | None = None,
        delimiter: str = ",",
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> None:
        self.path = Path(path)
        self.column_map = column_map
        self.delimiter = delimiter
        self.dtype = dtype
        self.device = device

    # ------------------------------------------------------------------
    def load(self) -> PIELMDataset:
        """Read the CSV and return a :class:`~pypielm.data.dataset.PIELMDataset`."""
        import numpy as np

        # Read header to support column-name addressing
        with open(self.path, encoding="utf-8") as fh:
            first_line = fh.readline().rstrip("\n")

        # Detect whether first row is a header (non-numeric)
        sample_values = first_line.split(self.delimiter)
        is_header = False
        try:
            float(sample_values[0].strip())
        except ValueError:
            is_header = True

        if is_header:
            col_names = [c.strip() for c in sample_values]
            arr = np.genfromtxt(
                self.path,
                delimiter=self.delimiter,
                skip_header=1,
                dtype=float,
            )
        else:
            col_names = [str(i) for i in range(len(sample_values))]
            arr = np.genfromtxt(
                self.path,
                delimiter=self.delimiter,
                dtype=float,
            )

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        name_to_idx: dict[str, int] = {name: i for i, name in enumerate(col_names)}

        def _cols(spec: list[str | int]) -> np.ndarray:
            indices = []
            for s in spec:
                if isinstance(s, int):
                    indices.append(s)
                else:
                    if s not in name_to_idx:
                        raise KeyError(f"Column '{s}' not found in CSV header.")
                    indices.append(name_to_idx[s])
            sub = arr[:, indices]
            return sub

        def _t(data: np.ndarray) -> torch.Tensor:
            t = torch.tensor(data, dtype=self.dtype, device=self.device)
            if t.ndim == 1:
                t = t.unsqueeze(1)
            return t

        if self.column_map is not None:
            kwargs: dict[str, Any] = {}
            for role, spec in self.column_map.items():
                kwargs[role] = _t(_cols(spec))
            if "X_colloc" not in kwargs:
                raise ValueError("column_map must include an 'X_colloc' entry.")
            return PIELMDataset(**kwargs)

        # Default: last column is y_data, remainder is X_colloc
        X_colloc = _t(arr[:, :-1])
        y_data = _t(arr[:, -1:])
        return PIELMDataset(X_colloc=X_colloc, y_data=y_data)
