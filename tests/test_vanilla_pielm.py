"""Tests for VanillaPIELM and CorePIELM.

Covers:
* Fluent fit() API (returns self)
* predict() after fit
* Reproducibility: same seed -> same beta
* Registry lookup via get_model()
* 1-D Poisson solve with relative L2 error < 1e-3
* MPS device (float32, skipped if unavailable)
* CUDA device (skipped if unavailable)
"""

from __future__ import annotations

import pytest
import torch

from pypielm.core.solver import WeightedLinearSystem
from pypielm.data.dataset import PIELMDataset
from pypielm.models.registry import get_model
from pypielm.models.vanilla import CorePIELM, VanillaPIELM
from pypielm.pde.operators import AnalyticLaplacian

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_poisson_1d(n_colloc: int = 100, dtype=torch.float64, device="cpu"):
    X_colloc = torch.linspace(0.0, 1.0, n_colloc, dtype=dtype).unsqueeze(1).to(device)
    X_bc = torch.tensor([[0.0], [1.0]], dtype=dtype, device=device)
    y_bc = torch.zeros(2, 1, dtype=dtype, device=device)
    dataset = PIELMDataset(X_colloc=X_colloc, X_bc=X_bc, y_bc=y_bc)

    def poisson_op(fm, X):
        H_lap = AnalyticLaplacian(fm)(X)
        f = -2.0 * torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)
        return WeightedLinearSystem(H_lap, f)

    X_test = torch.linspace(0.0, 1.0, 200, dtype=dtype, device=device).unsqueeze(1)
    y_exact = X_test * (1.0 - X_test)
    return dataset, poisson_op, X_test, y_exact


# ---------------------------------------------------------------------------
# VanillaPIELM
# ---------------------------------------------------------------------------

def test_vanilla_pielm_fit_returns_self():
    X = torch.rand(50, 1, dtype=torch.float64)
    y = torch.sin(X)
    ds = PIELMDataset(X_colloc=X, X_data=X, y_data=y)
    model = VanillaPIELM(hidden_dim=50, seed=0)
    result = model.fit(ds)
    assert result is model


def test_vanilla_pielm_predict_shape():
    X = torch.rand(50, 1, dtype=torch.float64)
    y = torch.sin(X)
    ds = PIELMDataset(X_colloc=X, X_data=X, y_data=y)
    model = VanillaPIELM(hidden_dim=50, seed=0).fit(ds)
    preds = model.predict(X)
    assert preds.shape == (50, 1)


def test_vanilla_pielm_reproducibility():
    X = torch.rand(60, 1, dtype=torch.float64)
    y = torch.sin(X)
    ds = PIELMDataset(X_colloc=X, X_data=X, y_data=y)
    m1 = VanillaPIELM(hidden_dim=50, seed=7).fit(ds)
    m2 = VanillaPIELM(hidden_dim=50, seed=7).fit(ds)
    assert torch.allclose(m1._beta, m2._beta, atol=0.0, rtol=0.0)


def test_vanilla_pielm_different_seeds_differ():
    X = torch.rand(60, 1, dtype=torch.float64)
    y = torch.sin(X)
    ds = PIELMDataset(X_colloc=X, X_data=X, y_data=y)
    m1 = VanillaPIELM(hidden_dim=50, seed=0).fit(ds)
    m2 = VanillaPIELM(hidden_dim=50, seed=1).fit(ds)
    assert not torch.allclose(m1._beta, m2._beta, atol=1e-6)


# ---------------------------------------------------------------------------
# CorePIELM
# ---------------------------------------------------------------------------

def test_core_pielm_fit_returns_self():
    ds, op, _, _ = _make_poisson_1d(50)
    model = CorePIELM(hidden_dim=50, seed=0)
    result = model.fit(ds, pde_operator=op)
    assert result is model


def test_core_pielm_predict_shape():
    ds, op, X_test, _ = _make_poisson_1d(80)
    model = CorePIELM(hidden_dim=100, seed=0).fit(ds, pde_operator=op)
    preds = model.predict(X_test)
    assert preds.shape == (200, 1)


def test_core_pielm_reproducibility():
    ds, op, _, _ = _make_poisson_1d(80)
    m1 = CorePIELM(hidden_dim=100, seed=3).fit(ds, pde_operator=op)
    m2 = CorePIELM(hidden_dim=100, seed=3).fit(ds, pde_operator=op)
    assert torch.allclose(m1._beta, m2._beta, atol=0.0, rtol=0.0)


def test_core_pielm_1d_poisson_accuracy():
    """Solve u''=-2 on [0,1] with zero BCs; exact solution u=x*(1-x)."""
    ds, op, X_test, y_exact = _make_poisson_1d(100)
    model = CorePIELM(hidden_dim=200, ridge_lambda=1e-8, seed=42).fit(ds, pde_operator=op)
    err = model.score(X_test, y_exact, metric="relative_l2")
    assert err < 1e-3, f"Poisson 1D error {err:.3e} exceeds threshold 1e-3"


def test_core_pielm_registry_lookup():
    model = get_model("core_pielm", hidden_dim=50, seed=0)
    assert isinstance(model, CorePIELM)


def test_core_pielm_numpy_input():
    """CorePIELM should accept numpy arrays as input to predict."""
    ds, op, X_test, y_exact = _make_poisson_1d(80)
    model = CorePIELM(hidden_dim=100, seed=0).fit(ds, pde_operator=op)
    X_np = X_test.numpy()
    preds = model.predict(X_np)
    assert isinstance(preds, torch.Tensor)
    assert preds.shape == (200, 1)


def test_core_pielm_get_feature_matrix():
    ds, op, X_test, _ = _make_poisson_1d(60)
    model = CorePIELM(hidden_dim=80, seed=0).fit(ds, pde_operator=op)
    H = model.get_feature_matrix(X_test)
    assert H.shape == (200, 80)


def test_core_pielm_score_returns_float():
    ds, op, X_test, y_exact = _make_poisson_1d(80)
    model = CorePIELM(hidden_dim=100, seed=0).fit(ds, pde_operator=op)
    s = model.score(X_test, y_exact)
    assert isinstance(s, float)
    assert s >= 0.0


# ---------------------------------------------------------------------------
# GPU: MPS
# ---------------------------------------------------------------------------

@pytest.mark.mps
def test_core_pielm_mps_fit_predict(mps_device):
    ds, op, X_test, y_exact = _make_poisson_1d(100, dtype=torch.float32, device=mps_device)
    model = CorePIELM(hidden_dim=200, ridge_lambda=1e-6, seed=42,
                      device=mps_device, dtype=torch.float32)
    model.fit(ds, pde_operator=op)
    preds = model.predict(X_test)
    assert preds.shape == (200, 1)
    assert preds.device.type == "mps"


# ---------------------------------------------------------------------------
# GPU: CUDA
# ---------------------------------------------------------------------------

@pytest.mark.cuda
def test_core_pielm_cuda_fit_predict(cuda_device):
    ds, op, X_test, y_exact = _make_poisson_1d(100, dtype=torch.float64, device=cuda_device)
    model = CorePIELM(hidden_dim=200, ridge_lambda=1e-8, seed=42,
                      device=cuda_device, dtype=torch.float64)
    model.fit(ds, pde_operator=op)
    preds = model.predict(X_test)
    assert preds.shape == (200, 1)
    assert preds.device.type == "cuda"
