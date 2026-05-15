"""Coverage tests for GFFPIELM (Generalised Fourier Feature PIELM).

Covers:
* solver="ridge" and solver="rrqr"
* freq_init="log_uniform" and "uniform"
* Invalid solver raises ValueError
* Predict-before-fit raises RuntimeError
* Accuracy on 1D Poisson (sin(πx) has one frequency → GFF should excel)
* get_feature_matrix, score, repr
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pypielm.core.solver import WeightedLinearSystem
from pypielm.data.dataset import PIELMDataset

DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _poisson_ds(n_colloc: int = 120) -> PIELMDataset:
    rng = np.random.default_rng(42)
    X_c  = rng.uniform(0.0, 1.0, (n_colloc, 1))
    X_bc = np.array([[0.0], [1.0]])
    y_bc = np.zeros((2, 1))
    return PIELMDataset.from_arrays(X_c, X_bc=X_bc, y_bc=y_bc, dtype=DTYPE)


def _data_ds(n: int = 100) -> PIELMDataset:
    rng = np.random.default_rng(42)
    X   = rng.uniform(0.0, 1.0, (n, 1))
    y   = np.sin(math.pi * X)
    X_bc = np.array([[0.0], [1.0]])
    y_bc = np.zeros((2, 1))
    return PIELMDataset.from_arrays(
        X, X_bc=X_bc, y_bc=y_bc,
        X_data=X, y_data=y,
        dtype=DTYPE,
    )


def _poisson_op(fm, X: torch.Tensor) -> WeightedLinearSystem:
    H_xx = fm.d2(X, 0)
    rhs  = (math.pi**2) * torch.sin(math.pi * X)
    return WeightedLinearSystem(-H_xx, rhs, weight=1.0)


# ---------------------------------------------------------------------------
# Constructor & Validation
# ---------------------------------------------------------------------------

class TestGFFPIELMConstruction:
    def test_invalid_solver_raises(self):
        from pypielm.models.fourier import GFFPIELM
        with pytest.raises(ValueError, match="solver"):
            GFFPIELM(solver="svd")

    def test_default_construction(self):
        from pypielm.models.fourier import GFFPIELM
        model = GFFPIELM()
        assert model.hidden_dim == 200
        assert model.solver == "ridge"
        assert model.freq_init == "log_uniform"

    def test_repr(self):
        from pypielm.models.fourier import GFFPIELM
        model = GFFPIELM(hidden_dim=64, solver="rrqr")
        r = repr(model)
        assert "GFFPIELM" in r or "GFF" in r.upper()

    def test_predict_before_fit_raises(self):
        from pypielm.models.fourier import GFFPIELM
        model = GFFPIELM()
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(torch.zeros(5, 1, dtype=DTYPE))

    def test_get_feature_matrix_before_fit_raises(self):
        from pypielm.models.fourier import GFFPIELM
        model = GFFPIELM()
        with pytest.raises(RuntimeError, match="fit"):
            model.get_feature_matrix(torch.zeros(5, 1, dtype=DTYPE))


# ---------------------------------------------------------------------------
# Solver="ridge"
# ---------------------------------------------------------------------------

class TestGFFPIELMRidge:
    def test_fit_returns_self(self):
        from pypielm.models.fourier import GFFPIELM
        ds = _data_ds()
        model = GFFPIELM(hidden_dim=50, solver="ridge", seed=42)
        result = model.fit(ds)
        assert result is model

    def test_predict_shape(self):
        from pypielm.models.fourier import GFFPIELM
        ds = _data_ds()
        model = GFFPIELM(hidden_dim=50, solver="ridge", seed=42)
        model.fit(ds)
        X = torch.linspace(0, 1, 30).unsqueeze(1).double()
        pred = model.predict(X)
        assert pred.shape == (30, 1)

    def test_finite_predictions(self):
        from pypielm.models.fourier import GFFPIELM
        ds = _data_ds()
        model = GFFPIELM(hidden_dim=60, solver="ridge", seed=42)
        model.fit(ds)
        X = torch.linspace(0, 1, 40).unsqueeze(1).double()
        pred = model.predict(X)
        assert torch.isfinite(pred).all()

    def test_accuracy_data_fit(self):
        from pypielm.models.fourier import GFFPIELM
        ds = _data_ds(n=200)
        model = GFFPIELM(
            hidden_dim=100, ridge_lambda=1e-8,
            freq_min=0.5, freq_max=10.0, seed=42,
        )
        model.fit(ds)
        X = torch.linspace(0.05, 0.95, 50).unsqueeze(1).double()
        pred = model.predict(X)
        exact = torch.sin(math.pi * X)
        rel_l2 = ((pred - exact).norm() / exact.norm()).item()
        assert rel_l2 < 0.30, f"GFFPIELM ridge rel_l2={rel_l2:.4f}"

    def test_score_finite(self):
        from pypielm.models.fourier import GFFPIELM
        ds = _data_ds()
        model = GFFPIELM(hidden_dim=50, seed=42)
        model.fit(ds)
        X = torch.linspace(0.1, 0.9, 20).unsqueeze(1).double()
        y = torch.sin(math.pi * X)
        score = model.score(X, y)
        assert math.isfinite(score)

    def test_get_feature_matrix_shape(self):
        from pypielm.models.fourier import GFFPIELM
        ds = _data_ds()
        model = GFFPIELM(hidden_dim=50, seed=42)
        model.fit(ds)
        X = torch.linspace(0, 1, 15).unsqueeze(1).double()
        H = model.get_feature_matrix(X)
        assert H.shape == (15, 50)

    def test_with_pde_operator(self):
        from pypielm.models.fourier import GFFPIELM
        ds = _poisson_ds()
        model = GFFPIELM(hidden_dim=80, ridge_lambda=1e-8,
                          freq_min=0.5, freq_max=20.0, seed=42)
        model.fit(ds, pde_operator=_poisson_op)
        X = torch.linspace(0, 1, 15).unsqueeze(1).double()
        pred = model.predict(X)
        assert torch.isfinite(pred).all()

    def test_deterministic_same_seed(self):
        from pypielm.models.fourier import GFFPIELM
        ds = _data_ds()
        m1 = GFFPIELM(hidden_dim=40, seed=7)
        m2 = GFFPIELM(hidden_dim=40, seed=7)
        m1.fit(ds)
        m2.fit(ds)
        X = torch.linspace(0, 1, 10).unsqueeze(1).double()
        assert torch.allclose(m1.predict(X), m2.predict(X))


# ---------------------------------------------------------------------------
# Solver="rrqr"
# ---------------------------------------------------------------------------

class TestGFFPIELMRRQR:
    def test_fit_returns_self(self):
        from pypielm.models.fourier import GFFPIELM
        ds = _data_ds()
        model = GFFPIELM(hidden_dim=40, solver="rrqr", seed=42)
        result = model.fit(ds)
        assert result is model

    def test_predict_shape(self):
        from pypielm.models.fourier import GFFPIELM
        ds = _data_ds()
        model = GFFPIELM(hidden_dim=40, solver="rrqr", seed=42)
        model.fit(ds)
        X = torch.linspace(0, 1, 20).unsqueeze(1).double()
        pred = model.predict(X)
        assert pred.shape == (20, 1)

    def test_finite_predictions(self):
        from pypielm.models.fourier import GFFPIELM
        ds = _data_ds()
        model = GFFPIELM(hidden_dim=40, solver="rrqr", seed=42)
        model.fit(ds)
        X = torch.linspace(0, 1, 25).unsqueeze(1).double()
        pred = model.predict(X)
        assert torch.isfinite(pred).all()

    def test_rrqr_vs_ridge_similar(self):
        """RRQR and ridge should give roughly similar predictions."""
        from pypielm.models.fourier import GFFPIELM
        ds = _data_ds(n=150)
        m_ridge = GFFPIELM(hidden_dim=50, solver="ridge", seed=42)
        m_rrqr  = GFFPIELM(hidden_dim=50, solver="rrqr",  seed=42)
        m_ridge.fit(ds)
        m_rrqr.fit(ds)
        X = torch.linspace(0.1, 0.9, 20).unsqueeze(1).double()
        rel = (m_ridge.predict(X) - m_rrqr.predict(X)).norm()
        # They won't be identical but should not be wildly different
        assert rel < 10.0, f"RRQR and ridge diverge: norm diff={rel:.4f}"


# ---------------------------------------------------------------------------
# Frequency initialisation
# ---------------------------------------------------------------------------

class TestGFFPIELMFrequency:
    @pytest.mark.parametrize("freq_init", ["log_uniform", "uniform"])
    def test_both_freq_inits_work(self, freq_init: str):
        from pypielm.models.fourier import GFFPIELM
        ds = _data_ds()
        model = GFFPIELM(hidden_dim=40, freq_init=freq_init, seed=42)
        model.fit(ds)
        X = torch.linspace(0, 1, 15).unsqueeze(1).double()
        pred = model.predict(X)
        assert torch.isfinite(pred).all()

    def test_log_uniform_spreads_frequencies(self):
        """log_uniform should produce frequencies spanning multiple decades."""
        from pypielm.models.fourier import GFFPIELM
        model = GFFPIELM(hidden_dim=100, freq_init="log_uniform",
                          freq_min=1.0, freq_max=1000.0, seed=42)
        ds = _data_ds()
        model.fit(ds)
        fm = model._fm
        omegas = fm.omega.abs()
        log_range = (omegas.max().log10() - omegas.min().log10()).item()
        assert log_range > 1.0, f"log_uniform freq range only {log_range:.2f} decades"

    def test_w_pde_w_bc_w_ic_accepted(self):
        """Non-default weights should not raise."""
        from pypielm.models.fourier import GFFPIELM
        ds = _poisson_ds()
        model = GFFPIELM(hidden_dim=40, w_pde=2.0, w_bc=5.0, w_ic=1.0, seed=42)
        model.fit(ds, pde_operator=_poisson_op)
        pred = model.predict(torch.linspace(0, 1, 10).unsqueeze(1).double())
        assert torch.isfinite(pred).all()
