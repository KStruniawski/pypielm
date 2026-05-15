"""Shared pytest fixtures for the PyPIELM test suite.

Device strategy
---------------
* ``--accelerator cpu``   — CPU only (default, always works, used in CI)
* ``--accelerator cuda``  — force CUDA; skip entire session if unavailable
* ``--accelerator mps``   — force Apple MPS; skip entire session if unavailable
* ``--accelerator auto``  — best available: CUDA > MPS > CPU

Individual device fixtures (``cuda_device``, ``mps_device``) skip the
specific test when that accelerator is not present, so the suite is always
safe to run on any machine.
"""

from __future__ import annotations

import os

# Must be set before torch (and thus libcublas) is imported so that
# deterministic CuBLAS algorithms work on CUDA >= 10.2.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest
import torch

from pypielm.data.dataset import PIELMDataset


# ---------------------------------------------------------------------------
# CLI option
# ---------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--accelerator",
        default="cpu",
        choices=["cpu", "cuda", "mps", "auto"],
        help="Compute device to use for accelerator-aware tests. "
             "Use 'auto' to select the best available device automatically.",
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _resolve_accelerator(name: str) -> torch.device:
    """Resolve an accelerator name to a ``torch.device``, skipping if absent."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA accelerator requested but not available.")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            pytest.skip("MPS accelerator requested but not available.")
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Session-scoped device fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def device() -> torch.device:
    """Default compute device — always CPU, safe for CI."""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def accelerator_device(request: pytest.FixtureRequest) -> torch.device:
    """Device selected by ``--accelerator`` CLI option (default: cpu)."""
    name: str = request.config.getoption("--accelerator")
    return _resolve_accelerator(name)


@pytest.fixture(scope="session")
def cuda_device() -> torch.device:
    """CUDA device; skips the test if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this machine.")
    return torch.device("cuda")


@pytest.fixture(scope="session")
def mps_device() -> torch.device:
    """Apple MPS device; skips the test if MPS is not available."""
    if not torch.backends.mps.is_available():
        pytest.skip("Apple MPS not available on this machine.")
    return torch.device("mps")


# ---------------------------------------------------------------------------
# Dtype
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def dtype() -> torch.dtype:
    return torch.float64


# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def poisson_1d_data() -> PIELMDataset:
    """Tiny synthetic 1D Poisson dataset: -u'' = π² sin(πx) on [0,1]."""
    x_colloc = torch.linspace(0.0, 1.0, 50, dtype=torch.float64).unsqueeze(1)
    x_bc = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    y_bc = torch.zeros(2, 1, dtype=torch.float64)
    return PIELMDataset(X_colloc=x_colloc, X_bc=x_bc, y_bc=y_bc)


@pytest.fixture
def small_dataset(poisson_1d_data: PIELMDataset) -> PIELMDataset:
    """Alias for poisson_1d_data (convenience)."""
    return poisson_1d_data

