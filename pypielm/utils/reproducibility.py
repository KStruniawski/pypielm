"""Utility helpers: reproducibility and device selection."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, *, deterministic: bool = True) -> None:
    """Set all relevant random seeds for full reproducibility.

    Seeds: Python ``random``, ``numpy.random``, PyTorch CPU, PyTorch CUDA, and
    Apple MPS (seeded implicitly via :func:`torch.manual_seed`).

    Args:
        seed: Integer seed value.
        deterministic: If ``True``, enables deterministic CUDA algorithms
            (may reduce performance but guarantees reproducibility).
            Has no effect on MPS (MPS operations are always deterministic).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also covers MPS
    if torch.cuda.is_available():  # pragma: no cover
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        # Required for deterministic CuBLAS ops on CUDA >= 10.2
        if torch.cuda.is_available():  # pragma: no cover
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import contextlib
        with contextlib.suppress(RuntimeError):  # pragma: no cover
            torch.use_deterministic_algorithms(True)


def get_device(prefer_cuda: bool = True, prefer_mps: bool = True) -> torch.device:
    """Return the best available :class:`torch.device`.

    Priority order: CUDA > MPS (Apple Silicon) > CPU.

    Args:
        prefer_cuda: If ``True`` and CUDA is available, returns a CUDA device.
        prefer_mps: If ``True`` and Apple MPS is available (and CUDA is not),
            returns an MPS device.  Ignored when ``prefer_cuda=True`` and CUDA
            is present.

    Returns:
        A :class:`torch.device`.
    """
    if prefer_cuda and torch.cuda.is_available():  # pragma: no cover
        return torch.device("cuda")
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

