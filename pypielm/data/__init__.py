"""Data loading and preprocessing.

Public surface::

    from pypielm.data import (
        PIELMDataset,
        auto_load,
        Normalizer,
        FeatureExpander,
        Pipeline,
        CSVAdapter,
        NPZAdapter,
        PINNacleAdapter,
        PDEBenchAdapter,
        TorchDatasetAdapter,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .adapters import (
    CSVAdapter,
    NPZAdapter,
    PDEBenchAdapter,
    PINNacleAdapter,
    TorchDatasetAdapter,
)
from .dataset import PIELMDataset
from .transforms import FeatureExpander, Normalizer, Pipeline


def auto_load(
    path: str | Path,
    *,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
    **kwargs: Any,
) -> PIELMDataset:
    """Infer file format from *path* and load a :class:`PIELMDataset`.

    Dispatch table (by file extension):

    * ``.csv``                   → :class:`CSVAdapter`
    * ``.npz``, ``.npy``         → :class:`NPZAdapter`
    * ``.dat``, ``.txt``         → :class:`PINNacleAdapter`
    * ``.h5``, ``.hdf5``         → :class:`PDEBenchAdapter`

    Extra *kwargs* are forwarded to the selected adapter constructor.

    Args:
        path: Path to the data file.
        dtype: Tensor dtype.
        device: Target device.
        **kwargs: Extra arguments for the chosen adapter.

    Returns:
        A :class:`PIELMDataset` instance.

    Raises:
        ValueError: If the file extension is not recognised.
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return CSVAdapter(path, dtype=dtype, device=device, **kwargs).load()

    if suffix in {".npz", ".npy"}:
        return NPZAdapter(path, dtype=dtype, device=device, **kwargs).load()

    if suffix in {".dat", ".txt"}:
        # PINNacleAdapter takes (root, task); for a direct path supply the
        # parent as root and the filename (without extension) as the task.
        root = kwargs.pop("root", path.parent)
        task = kwargs.pop("task", path.name)
        return PINNacleAdapter(root, task, dtype=dtype, device=device, **kwargs).load()

    if suffix in {".h5", ".hdf5"}:
        return PDEBenchAdapter(path, dtype=dtype, device=device, **kwargs).load()

    raise ValueError(
        f"Unrecognised file extension '{suffix}'. "
        "Supported: .csv, .npz, .npy, .dat, .txt, .h5, .hdf5"
    )


__all__ = [
    "PIELMDataset",
    "auto_load",
    "Normalizer",
    "FeatureExpander",
    "Pipeline",
    "CSVAdapter",
    "NPZAdapter",
    "PINNacleAdapter",
    "PDEBenchAdapter",
    "TorchDatasetAdapter",
]
