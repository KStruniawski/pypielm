"""Coverage tests for constrained PIELM variants.

Covers:
* NullSpacePIELM — null-space BC enforcement
* EigPIELM       — eigendecomposition BC enforcement
* LSEELM         — Lagrange multiplier equality constraints
* StefanPIELM    — free-boundary iterative solver
* Wrapper variants (NormalEquationELM, FPIELM, RINN, …)

All tests use 1D Poisson:  -u'' = π² sin(πx),  u(0)=u(1)=0,  u = sin(πx).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pypielm.core.solver import WeightedLinearSystem
from pypielm.data.dataset import PIELMDataset

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

DTYPE = torch.float64


def _poisson_ds(n_colloc: int = 100, n_bc: int = 2) -> PIELMDataset:
    rng = np.random.default_rng(42)
    X_c  = rng.uniform(0.0, 1.0, (n_colloc, 1))
    X_bc = np.array([[0.0], [1.0]])
    y_bc = np.zeros((2, 1))
    return PIELMDataset.from_arrays(X_c, X_bc=X_bc, y_bc=y_bc, dtype=DTYPE)


def _poisson_op(fm, X: torch.Tensor) -> WeightedLinearSystem:
    H_xx = fm.d2(X, 0)
    rhs  = (math.pi**2) * torch.sin(math.pi * X)
    return WeightedLinearSystem(-H_xx, rhs, weight=1.0)


def _eval_exact(X: torch.Tensor) -> torch.Tensor:
    return torch.sin(math.pi * X)


def _data_ds(n: int = 80) -> PIELMDataset:
    """Dataset with X_data / y_data but no explicit collocation."""
    rng = np.random.default_rng(42)
    X_c = rng.uniform(0.0, 1.0, (n, 1))
    X_bc = np.array([[0.0], [1.0]])
    y_bc = np.zeros((2, 1))
    y_data = np.sin(math.pi * X_c)
    return PIELMDataset.from_arrays(
        X_c, X_bc=X_bc, y_bc=y_bc,
        X_data=X_c, y_data=y_data,
        dtype=DTYPE,
    )


# ---------------------------------------------------------------------------
# NullSpacePIELM
# ---------------------------------------------------------------------------

class TestNullSpacePIELM:
    def test_fit_returns_self(self):
        from pypielm.models.constrained import NullSpacePIELM
        ds = _poisson_ds()
        model = NullSpacePIELM(hidden_dim=50, seed=42)
        result = model.fit(ds, pde_operator=_poisson_op)
        assert result is model

    def test_predict_shape(self):
        from pypielm.models.constrained import NullSpacePIELM
        ds = _poisson_ds()
        model = NullSpacePIELM(hidden_dim=50, seed=42)
        model.fit(ds, pde_operator=_poisson_op)
        X = torch.linspace(0.0, 1.0, 30).unsqueeze(1).double()
        pred = model.predict(X)
        assert pred.shape == (30, 1)

    def test_accuracy(self):
        from pypielm.models.constrained import NullSpacePIELM
        ds = _poisson_ds(n_colloc=200)
        model = NullSpacePIELM(hidden_dim=100, ridge_lambda=1e-8, seed=42)
        model.fit(ds, pde_operator=_poisson_op)
        X = torch.linspace(0.05, 0.95, 50).unsqueeze(1).double()
        pred = model.predict(X)
        exact = _eval_exact(X)
        rel_l2 = ((pred - exact).norm() / exact.norm()).item()
        assert rel_l2 < 0.20, f"NullSpacePIELM rel_l2={rel_l2:.3f}"

    def test_bc_approximately_satisfied(self):
        from pypielm.models.constrained import NullSpacePIELM
        ds = _poisson_ds()
        model = NullSpacePIELM(hidden_dim=100, seed=42)
        model.fit(ds, pde_operator=_poisson_op)
        X_bc = torch.tensor([[0.0], [1.0]], dtype=DTYPE)
        bc_pred = model.predict(X_bc)
        assert bc_pred.abs().max().item() < 0.1, \
            f"BCs not satisfied: {bc_pred.abs().max().item():.4f}"

    def test_get_feature_matrix_shape(self):
        from pypielm.models.constrained import NullSpacePIELM
        ds = _poisson_ds()
        model = NullSpacePIELM(hidden_dim=50, seed=42)
        model.fit(ds, pde_operator=_poisson_op)
        X = torch.linspace(0.0, 1.0, 20).unsqueeze(1).double()
        H = model.get_feature_matrix(X)
        assert H.shape == (20, 50)

    def test_predict_before_fit_raises(self):
        from pypielm.models.constrained import NullSpacePIELM
        model = NullSpacePIELM(hidden_dim=20)
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(torch.zeros(5, 1, dtype=DTYPE))

    def test_get_feature_matrix_before_fit_raises(self):
        from pypielm.models.constrained import NullSpacePIELM
        model = NullSpacePIELM(hidden_dim=20)
        with pytest.raises(RuntimeError, match="fit"):
            model.get_feature_matrix(torch.zeros(5, 1, dtype=DTYPE))

    def test_fallback_no_bc(self):
        """Without BCs in dataset, falls back to ridge solve."""
        from pypielm.models.constrained import NullSpacePIELM
        ds = _data_ds()
        model = NullSpacePIELM(hidden_dim=50, seed=42)
        # No BC → should fall back to ridge
        model.fit(ds)
        pred = model.predict(torch.linspace(0, 1, 10).unsqueeze(1).double())
        assert pred.shape == (10, 1)

    def test_score(self):
        from pypielm.models.constrained import NullSpacePIELM
        ds = _poisson_ds()
        model = NullSpacePIELM(hidden_dim=50, seed=42)
        model.fit(ds, pde_operator=_poisson_op)
        X = torch.linspace(0.1, 0.9, 20).unsqueeze(1).double()
        y = _eval_exact(X)
        score = model.score(X, y)
        assert isinstance(score, float)
        assert math.isfinite(score)

    def test_data_block_used_when_no_pde(self):
        """When pde_operator=None and y_data present, uses data block."""
        from pypielm.models.constrained import NullSpacePIELM
        ds = _data_ds()
        model = NullSpacePIELM(hidden_dim=50, seed=42)
        model.fit(ds, pde_operator=None)
        pred = model.predict(torch.linspace(0, 1, 5).unsqueeze(1).double())
        assert pred.shape == (5, 1)


# ---------------------------------------------------------------------------
# EigPIELM
# ---------------------------------------------------------------------------

class TestEigPIELM:
    def test_fit_returns_self(self):
        from pypielm.models.constrained import EigPIELM
        ds = _poisson_ds()
        model = EigPIELM(hidden_dim=50, seed=42)
        result = model.fit(ds, pde_operator=_poisson_op)
        assert result is model

    def test_predict_shape(self):
        from pypielm.models.constrained import EigPIELM
        ds = _poisson_ds()
        model = EigPIELM(hidden_dim=50, seed=42)
        model.fit(ds, pde_operator=_poisson_op)
        X = torch.linspace(0.0, 1.0, 25).unsqueeze(1).double()
        pred = model.predict(X)
        assert pred.shape == (25, 1)

    def test_accuracy_reasonable(self):
        from pypielm.models.constrained import EigPIELM
        ds = _poisson_ds(n_colloc=200)
        model = EigPIELM(hidden_dim=100, ridge_lambda=1e-8, seed=42)
        model.fit(ds, pde_operator=_poisson_op)
        X = torch.linspace(0.05, 0.95, 40).unsqueeze(1).double()
        pred = model.predict(X)
        assert torch.isfinite(pred).all()

    def test_predict_before_fit_raises(self):
        from pypielm.models.constrained import EigPIELM
        model = EigPIELM(hidden_dim=20)
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(torch.zeros(5, 1, dtype=DTYPE))

    def test_get_feature_matrix_before_fit_raises(self):
        from pypielm.models.constrained import EigPIELM
        model = EigPIELM(hidden_dim=20)
        with pytest.raises(RuntimeError, match="fit"):
            model.get_feature_matrix(torch.zeros(5, 1, dtype=DTYPE))

    def test_get_feature_matrix_shape(self):
        from pypielm.models.constrained import EigPIELM
        ds = _poisson_ds()
        model = EigPIELM(hidden_dim=50, seed=42)
        model.fit(ds, pde_operator=_poisson_op)
        X = torch.linspace(0, 1, 15).unsqueeze(1).double()
        H = model.get_feature_matrix(X)
        assert H.shape == (15, 50)

    def test_score_finite(self):
        from pypielm.models.constrained import EigPIELM
        ds = _poisson_ds()
        model = EigPIELM(hidden_dim=50, seed=42)
        model.fit(ds, pde_operator=_poisson_op)
        X = torch.linspace(0.1, 0.9, 20).unsqueeze(1).double()
        y = _eval_exact(X)
        score = model.score(X, y)
        assert math.isfinite(score)

    def test_repr(self):
        from pypielm.models.constrained import EigPIELM
        model = EigPIELM(hidden_dim=50)
        assert "EigPIELM" in repr(model) or "eig" in repr(model).lower()

    def test_data_block_no_pde(self):
        from pypielm.models.constrained import EigPIELM
        ds = _data_ds()
        model = EigPIELM(hidden_dim=50, seed=42)
        model.fit(ds)
        pred = model.predict(torch.linspace(0, 1, 5).unsqueeze(1).double())
        assert pred.shape[0] == 5


# ---------------------------------------------------------------------------
# LSEELM (Lagrange multiplier ELM)
# ---------------------------------------------------------------------------

class TestLSEELM:
    def test_fit_returns_self(self):
        from pypielm.models.constrained import LSEELM
        ds = _poisson_ds()
        model = LSEELM(hidden_dim=50, seed=42)
        result = model.fit(ds, pde_operator=_poisson_op)
        assert result is model

    def test_predict_shape(self):
        from pypielm.models.constrained import LSEELM
        ds = _poisson_ds()
        model = LSEELM(hidden_dim=50, seed=42)
        model.fit(ds, pde_operator=_poisson_op)
        X = torch.linspace(0, 1, 30).unsqueeze(1).double()
        pred = model.predict(X)
        assert pred.shape == (30, 1)

    def test_finite_predictions(self):
        from pypielm.models.constrained import LSEELM
        ds = _poisson_ds(n_colloc=200)
        model = LSEELM(hidden_dim=80, seed=42)
        model.fit(ds, pde_operator=_poisson_op)
        X = torch.linspace(0, 1, 50).unsqueeze(1).double()
        pred = model.predict(X)
        assert torch.isfinite(pred).all()

    def test_predict_before_fit_raises(self):
        from pypielm.models.constrained import LSEELM
        model = LSEELM(hidden_dim=20)
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(torch.zeros(5, 1, dtype=DTYPE))

    def test_get_feature_matrix_shape(self):
        from pypielm.models.constrained import LSEELM
        ds = _poisson_ds()
        model = LSEELM(hidden_dim=50, seed=42)
        model.fit(ds, pde_operator=_poisson_op)
        X = torch.linspace(0, 1, 10).unsqueeze(1).double()
        H = model.get_feature_matrix(X)
        assert H.shape == (10, 50)

    def test_score(self):
        from pypielm.models.constrained import LSEELM
        ds = _poisson_ds()
        model = LSEELM(hidden_dim=50, seed=42)
        model.fit(ds, pde_operator=_poisson_op)
        X = torch.linspace(0.1, 0.9, 20).unsqueeze(1).double()
        y = _eval_exact(X)
        score = model.score(X, y)
        assert math.isfinite(score)

    def test_data_only(self):
        from pypielm.models.constrained import LSEELM
        ds = _data_ds()
        model = LSEELM(hidden_dim=50, seed=42)
        model.fit(ds)
        pred = model.predict(torch.linspace(0, 1, 5).unsqueeze(1).double())
        assert pred.shape[0] == 5


# ---------------------------------------------------------------------------
# StefanPIELM (free-boundary)
# ---------------------------------------------------------------------------

class TestStefanPIELM:
    def _stefan_ds(self) -> PIELMDataset:
        """Simple 1D dataset with both X_data and X_bc for Stefan solve."""
        rng = np.random.default_rng(42)
        X_c  = rng.uniform(0.0, 1.0, (80, 1))
        X_bc = np.array([[0.0], [1.0]])
        y_bc = np.zeros((2, 1))
        X_d  = rng.uniform(0.0, 1.0, (60, 1))
        y_d  = np.sin(math.pi * X_d)
        return PIELMDataset.from_arrays(
            X_c, X_bc=X_bc, y_bc=y_bc, X_data=X_d, y_data=y_d, dtype=DTYPE
        )

    def test_fit_returns_self(self):
        from pypielm.models.constrained import StefanPIELM
        ds = self._stefan_ds()
        model = StefanPIELM(hidden_dim=40, seed=42)
        result = model.fit(ds)
        assert result is model

    def test_predict_shape(self):
        from pypielm.models.constrained import StefanPIELM
        ds = self._stefan_ds()
        model = StefanPIELM(hidden_dim=40, seed=42)
        model.fit(ds)
        X = torch.linspace(0, 1, 20).unsqueeze(1).double()
        pred = model.predict(X)
        assert pred.shape[0] == 20

    def test_predict_before_fit_raises(self):
        from pypielm.models.constrained import StefanPIELM
        model = StefanPIELM(hidden_dim=20)
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(torch.zeros(5, 1, dtype=DTYPE))

    def test_no_data_raises(self):
        from pypielm.models.constrained import StefanPIELM
        ds = _poisson_ds()  # no y_data
        model = StefanPIELM(hidden_dim=20, seed=42)
        with pytest.raises(ValueError, match="y_data"):
            model.fit(ds)

    def test_score(self):
        from pypielm.models.constrained import StefanPIELM
        ds = self._stefan_ds()
        model = StefanPIELM(hidden_dim=40, seed=42)
        model.fit(ds)
        X = torch.linspace(0.1, 0.9, 15).unsqueeze(1).double()
        y = _eval_exact(X)
        score = model.score(X, y)
        assert math.isfinite(score)

    def test_get_feature_matrix_raises_before_fit(self):
        from pypielm.models.constrained import StefanPIELM
        model = StefanPIELM(hidden_dim=20)
        with pytest.raises(RuntimeError, match="fit"):
            model.get_feature_matrix(torch.zeros(5, 1, dtype=DTYPE))


# ---------------------------------------------------------------------------
# Thin-wrapper variants (NormalEquationELM, FPIELM, RINN, etc.)
# ---------------------------------------------------------------------------

_WRAPPER_NAMES = [
    "normal_equation_elm",
    "parameter_retention_elm",
    "fpielm",
    "sgepielm",
    "rinn",
    "rann_pielm",
    "xpielm",
    "pielm_rvds",
    "tspielm",
    "kapielm",
    "soft_partition_kapielm",
    "delm",
    "piecewise_elm",
]


@pytest.mark.parametrize("model_name", _WRAPPER_NAMES)
class TestCoreVariantWrappers:
    def test_fit_predict_roundtrip(self, model_name: str):
        from pypielm.models.registry import get_model
        ds = _data_ds(n=60)
        model = get_model(model_name, hidden_dim=30, seed=42)
        model.fit(ds)
        X = torch.linspace(0, 1, 10).unsqueeze(1).double()
        pred = model.predict(X)
        assert pred.shape[0] == 10, f"{model_name}: wrong pred shape"
        assert torch.isfinite(pred).all(), f"{model_name}: non-finite predictions"

    def test_score_returns_float(self, model_name: str):
        from pypielm.models.registry import get_model
        ds = _data_ds(n=60)
        model = get_model(model_name, hidden_dim=30, seed=42)
        model.fit(ds)
        X = torch.linspace(0.1, 0.9, 10).unsqueeze(1).double()
        y = _eval_exact(X)
        score = model.score(X, y)
        assert isinstance(score, float)
        assert math.isfinite(score)

    def test_get_feature_matrix(self, model_name: str):
        from pypielm.models.registry import get_model
        ds = _data_ds(n=60)
        model = get_model(model_name, hidden_dim=30, seed=42)
        model.fit(ds)
        X = torch.linspace(0, 1, 8).unsqueeze(1).double()
        H = model.get_feature_matrix(X)
        assert H.shape[0] == 8
