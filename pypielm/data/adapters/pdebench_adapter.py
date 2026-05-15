"""PDEBench dataset adapter.

PDEBench (https://github.com/pdebench/PDEBench) distributes datasets as HDF5
files.  This adapter reads a single PDE task file and converts it to
:class:`~pypielm.data.dataset.PIELMDataset`.

h5py is an optional dependency — a clear ``ImportError`` is raised if missing.
"""

from __future__ import annotations

from pathlib import Path

import torch

from pypielm.data.dataset import PIELMDataset


class PDEBenchAdapter:
    """Load a :class:`~pypielm.data.dataset.PIELMDataset` from a PDEBench HDF5 file.

    The adapter reads the spatial grid coordinates (``x``) and the solution
    field (``u``) and flattens them into ``(N, d)`` and ``(N, m)`` arrays
    suitable for a PIELM collocation problem.

    Args:
        path: Path to the ``.h5`` / ``.hdf5`` file.
        equation: Equation name key inside the HDF5 file (top-level group),
            e.g. ``'1D_Advection'``.  When ``None`` the first top-level group
            is used.
        sample_idx: Index of the trajectory/sample to load (time snapshot 0
            is used as the target field).
        dtype: Tensor dtype.
        device: Target device.
    """

    def __init__(
        self,
        path: str | Path,
        equation: str | None = None,
        sample_idx: int = 0,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> None:
        self.path = Path(path)
        self.equation = equation
        self.sample_idx = sample_idx
        self.dtype = dtype
        self.device = device

    def load(self) -> PIELMDataset:  # pragma: no cover
        """Load and return the dataset.

        Requires the optional ``h5py`` dependency and an actual HDF5 file;
        excluded from automated coverage runs.
        """
        try:
            import h5py  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "h5py is required for PDEBenchAdapter.  Install it with:\n"
                "  pip install h5py"
            ) from exc

        import numpy as np

        with h5py.File(self.path, "r") as f:
            eq_key = self.equation
            if eq_key is None:
                eq_key = list(f.keys())[0]
            if eq_key not in f:
                raise KeyError(
                    f"Equation key '{eq_key}' not found in '{self.path}'. "
                    f"Available keys: {list(f.keys())}"
                )
            grp = f[eq_key]

            # Try common PDEBench layout: grp['x'] for coordinates, grp['u'] for solution
            # Fallback: treat all numeric datasets as candidate arrays
            x_arr: np.ndarray | None = None
            u_arr: np.ndarray | None = None

            for cand in ("x", "coords", "grid"):
                if cand in grp:
                    x_arr = np.asarray(grp[cand])
                    break
            for cand in ("u", "sol", "solution", "output"):
                if cand in grp:
                    u_arr = np.asarray(grp[cand])
                    break

            # If u is 3-D (sample, time, space) take the requested sample at t=0
            if u_arr is not None and u_arr.ndim == 3:
                u_arr = u_arr[self.sample_idx, 0, :]  # (space,)
            elif u_arr is not None and u_arr.ndim == 2:
                u_arr = u_arr[self.sample_idx, :]

            if x_arr is None or u_arr is None:
                raise ValueError(
                    f"Could not locate coordinate/solution arrays in group '{eq_key}'. "
                    "Provide explicit dataset keys via the 'equation' argument or "
                    "extend this adapter for your file layout."
                )

        # Flatten to 2-D
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        if u_arr.ndim == 1:
            u_arr = u_arr[:, None]

        def _t(a: np.ndarray) -> torch.Tensor:
            return torch.tensor(a, dtype=self.dtype, device=self.device)

        return PIELMDataset(
            X_colloc=_t(x_arr),
            y_data=_t(u_arr),
            meta={"source": "pdebench", "equation": eq_key, "file": str(self.path)},
        )
