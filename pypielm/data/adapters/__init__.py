"""Data adapters for loading datasets from various external formats.

Public surface::

    from pypielm.data.adapters import (
        CSVAdapter,
        NPZAdapter,
        PINNacleAdapter,
        PDEBenchAdapter,
        TorchDatasetAdapter,
    )
"""

from __future__ import annotations

from .csv_adapter import CSVAdapter
from .npz_adapter import NPZAdapter
from .pdebench_adapter import PDEBenchAdapter
from .pinnacle_adapter import PINNacleAdapter
from .torch_adapter import TorchDatasetAdapter

__all__ = [
    "CSVAdapter",
    "NPZAdapter",
    "PINNacleAdapter",
    "PDEBenchAdapter",
    "TorchDatasetAdapter",
]
