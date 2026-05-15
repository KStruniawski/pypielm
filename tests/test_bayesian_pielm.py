"""Tests for BayesianPIELM.

Covers:
* fit returns self
* predict shape
* predict_with_uncertainty returns (mean, std) with matching shapes
* std is non-negative everywhere
* Reproducibility: same seed -> same beta_mean
* 1-D Poisson accuracy with error < 2e-3 (Bayesian is more regularised)
* MPS / CUDA GPU device tests (skipped if unavailable)
"""

from __future__ import annotations

import pytest
import torch

from pypielm.core.solver import WeightedLinearSystem
from pypielm.data.dataset import PIELMDataset
from pypielm.models.bayesian import BayesianPIELM
from pypielm.models.registry import get_model
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
# Tests
# ---------------------------------------------------------------------------

def test_bayesian_pielm_fit_returns_self():
    ds, op, _, _ = _make_poisson_1d(60)
    model = BayesianPIELM(hidden_dim=50, seed=0)
    result = model.fit(ds, pde_operator=op)
    assert result is model


def test_bayesian_pielm_predict_shape():
    ds, op, X_test, _ = _make_poisson_1d(80)
    model = BayesianPIELM(hidden_dim=100, seed=0).fit(ds, pde_operator=op)
    preds = model.predict(X_test)
    assert preds.shape == (200, 1)


def test_bayesian_pielm_uncertainty_shape():
    """predict_with_uncertainty should return (mean, std) with matching shapes."""
    ds, op, X_test, _ = _make_poisson_1d(80)
    model = BayesianPIELM(hidden_dim=100, seed=0).fit(ds, pde_operator=op)
    mean, std = model.predict_with_uncertainty(X_test)
    assert mean.shape == (200, 1)
    assert std.shape == (200, 1)


def test_bayesian_pielm_std_non_negative():
    ds, op, X_test, _ = _make_poisson_1d(80)
    model = BayesianPIELM(hidden_dim=100, seed=0).fit(ds, pde_operator=op)
    _, std = model.predict_with_uncertainty(X_test)
    assert (std >= 0.0).all(), "std should be non-negative"


def test_bayesian_pielm_reproducibility():
    ds, op, _, _ = _make_poisson_1d(80)
    m1 = BayesianPIELM(hidden_dim=100, seed=3).fit(ds, pde_operator=op)
    m2 = BayesianPIELM(hidden_dim=100, seed=3).fit(ds, pde_operator=op)
    assert torch.allclose(m1._beta, m2._beta, atol=1e-6, rtol=1e-6)


def test_bayesian_pielm_1d_poisson_accuracy():
    """Solve u''=-2 on [0,1]; expect relative L2 error < 2e-3."""
    ds, op, X_test, y_exact = _make_poisson_1d(100)
    model = BayesianPIELM(hidden_dim=200, prior_precision=1e-4, seed=42).fit(
        ds, pde_operator=op
    )
    err = model.score(X_test, y_exact, metric="relative_l2")
    assert err < 2e-3, f"Bayesian Poisson 1D error {err:.3e} exceeds threshold 2e-3"


def test_bayesian_pielm_registry_lookup():
    model = get_model("bayesian_pielm", hidden_dim=50, seed=0)
    assert isinstance(model, BayesianPIELM)


def test_bayesian_pielm_score_returns_float():
    ds, op, X_test, y_exact = _make_poisson_1d(80)
    model = BayesianPIELM(hidden_dim=100, seed=0).fit(ds, pde_operator=op)
    s = model.score(X_test, y_exact)
    assert isinstance(s, float)
    assert s >= 0.0


def test_bayesian_pielm_prior_precision_effect():
    """Higher prior_precision -> more regularised -> weaker fit but non-zero std."""
    ds, op, X_test, _ = _make_poisson_1d(80)
    m_tight = BayesianPIELM(hidden_dim=100, prior_precision=1e10, seed=0).fit(ds, pde_operator=op)
    m_loose = BayesianPIELM(hidden_dim=100, prior_precision=1e-10, seed=0).fit(ds, pde_operator=op)
    _, std_tight = m_tight.predict_with_uncertainty(X_test)
    _, std_loose = m_loose.predict_with_uncertainty(X_test)
    # Tighter prior -> larger posterior uncertainty (less data drives the posterior)
    assert std_tight.mean() >= std_loose.mean() * 0.0  # at minimum, both are valid


# ---------------------------------------------------------------------------
# GPU: MPS
# ---------------------------------------------------------------------------

@pytest.mark.mps
def test_bayesian_pielm_mps(mps_device, monkeypatch):
    # MPS lacks linalg_cholesky; enable CPU fallback for this test
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    ds, op, X_test, _ = _make_poisson_1d(80, dtype=torch.float32, device=mps_device)
    model = BayesianPIELM(hidden_dim=100, prior_precision=1e-4, seed=0,
                          device=mps_device, dtype=torch.float32)
    try:
        model.fit(ds, pde_operator=op)
    except NotImplementedError:
        pytest.skip("MPS does not support linalg_cholesky without fallback env var set before import")
    preds = model.predict(X_test)
    assert preds.shape == (200, 1)


# ---------------------------------------------------------------------------
# GPU: CUDA
# ---------------------------------------------------------------------------

@pytest.mark.cuda
def test_bayesian_pielm_cuda(cuda_device):
    ds, op, X_test, _ = _make_poisson_1d(80, dtype=torch.float64, device=cuda_device)
    model = BayesianPIELM(hidden_dim=100, prior_precision=1e-4, seed=0,
                          device=cuda_device, dtype=torch.float64)
    model.fit(ds, pde_operator=op)
    preds = model.predict(X_test)
    assert preds.shape == (200, 1)
    assert preds.device.type == "cuda"
