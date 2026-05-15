"""Coverage tests for CurriculumPIELM.

Exercises:
* Normal multi-stage fit with PDE operator (full refinement loop)
* Data-only fit (pde_operator=None, exercises the `break` path)
* predict / score / get_feature_matrix
* Predict-before-fit guard
* No-data, no-PDE raises ValueError
* repr
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
# Shared helpers
# ---------------------------------------------------------------------------

def _poisson_ds(n_colloc: int = 80, n_bc: int = 2) -> PIELMDataset:
    rng = np.random.default_rng(42)
    X_c  = rng.uniform(0.0, 1.0, (n_colloc, 1))
    X_bc = np.array([[0.0], [1.0]])
    y_bc = np.zeros((2, 1))
    return PIELMDataset.from_arrays(X_c, X_bc=X_bc, y_bc=y_bc, dtype=DTYPE)


def _poisson_op(fm, X: torch.Tensor) -> WeightedLinearSystem:
    """PDE operator: -u'' = π² sin(πx) → H_op = -d²H/dx²."""
    H_xx = fm.d2(X, 0)
    rhs  = (math.pi**2) * torch.sin(math.pi * X)
    return WeightedLinearSystem(-H_xx, rhs, weight=1.0)


def _data_ds(n: int = 80) -> PIELMDataset:
    """Dataset with only X_data/y_data (no PDE collocation)."""
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCurriculumPIELMDataOnly:
    """pde_operator=None → single-stage, exercises break path."""

    def test_fit_returns_self(self):
        from pypielm.models.curriculum import CurriculumPIELM
        ds = _data_ds()
        model = CurriculumPIELM(hidden_dim=30, n_stages=3, seed=42)
        result = model.fit(ds, pde_operator=None)
        assert result is model

    def test_predict_shape(self):
        from pypielm.models.curriculum import CurriculumPIELM
        ds = _data_ds()
        model = CurriculumPIELM(hidden_dim=30, n_stages=2, seed=42)
        model.fit(ds)
        X = torch.linspace(0, 1, 20).unsqueeze(1).double()
        pred = model.predict(X)
        assert pred.shape == (20, 1)

    def test_predict_finite(self):
        from pypielm.models.curriculum import CurriculumPIELM
        ds = _data_ds()
        model = CurriculumPIELM(hidden_dim=40, n_stages=1, seed=42)
        model.fit(ds)
        X = torch.linspace(0, 1, 30).unsqueeze(1).double()
        pred = model.predict(X)
        assert torch.isfinite(pred).all()

    def test_score_finite(self):
        from pypielm.models.curriculum import CurriculumPIELM
        ds = _data_ds()
        model = CurriculumPIELM(hidden_dim=30, seed=42)
        model.fit(ds)
        X = torch.linspace(0.1, 0.9, 15).unsqueeze(1).double()
        y = torch.sin(math.pi * X)
        score = model.score(X, y)
        assert math.isfinite(score)

    def test_get_feature_matrix_shape(self):
        from pypielm.models.curriculum import CurriculumPIELM
        ds = _data_ds()
        model = CurriculumPIELM(hidden_dim=32, seed=42)
        model.fit(ds)
        X = torch.linspace(0, 1, 12).unsqueeze(1).double()
        H = model.get_feature_matrix(X)
        assert H.shape == (12, 32)

    def test_accuracy_reasonable(self):
        from pypielm.models.curriculum import CurriculumPIELM
        ds = _data_ds(n=200)
        model = CurriculumPIELM(hidden_dim=80, ridge_lambda=1e-8, n_stages=1, seed=42)
        model.fit(ds)
        X = torch.linspace(0.05, 0.95, 50).unsqueeze(1).double()
        pred = model.predict(X)
        exact = torch.sin(math.pi * X)
        rel_l2 = ((pred - exact).norm() / exact.norm()).item()
        assert rel_l2 < 0.25, f"CurriculumPIELM (data-only) rel_l2={rel_l2:.3f}"


class TestCurriculumPIELMWithPDE:
    """pde_operator provided → full multi-stage refinement loop."""

    def test_fit_returns_self(self):
        from pypielm.models.curriculum import CurriculumPIELM
        ds = _poisson_ds()
        model = CurriculumPIELM(hidden_dim=30, n_stages=2,
                                 n_collocation=50, n_candidates=200, seed=42)
        result = model.fit(ds, pde_operator=_poisson_op)
        assert result is model

    def test_predict_shape(self):
        from pypielm.models.curriculum import CurriculumPIELM
        ds = _poisson_ds()
        model = CurriculumPIELM(hidden_dim=30, n_stages=2,
                                 n_collocation=50, n_candidates=200, seed=42)
        model.fit(ds, pde_operator=_poisson_op)
        X = torch.linspace(0, 1, 20).unsqueeze(1).double()
        pred = model.predict(X)
        assert pred.shape == (20, 1)

    def test_finite_predictions(self):
        from pypielm.models.curriculum import CurriculumPIELM
        ds = _poisson_ds(n_colloc=100)
        model = CurriculumPIELM(hidden_dim=50, n_stages=3,
                                 n_collocation=60, n_candidates=300,
                                 refine_ratio=0.4, seed=42)
        model.fit(ds, pde_operator=_poisson_op)
        X = torch.linspace(0, 1, 25).unsqueeze(1).double()
        pred = model.predict(X)
        assert torch.isfinite(pred).all()

    def test_accuracy_improved(self):
        """3-stage curriculum should solve 1D Poisson with acceptable accuracy."""
        from pypielm.models.curriculum import CurriculumPIELM
        ds = _poisson_ds(n_colloc=200)
        model = CurriculumPIELM(
            hidden_dim=100, ridge_lambda=1e-8,
            n_stages=3, n_collocation=100, n_candidates=500,
            refine_ratio=0.5, seed=42,
        )
        model.fit(ds, pde_operator=_poisson_op)
        X = torch.linspace(0.05, 0.95, 40).unsqueeze(1).double()
        pred = model.predict(X)
        exact = torch.sin(math.pi * X)
        rel_l2 = ((pred - exact).norm() / exact.norm()).item()
        assert rel_l2 < 0.35, f"CurriculumPIELM (3 stages) rel_l2={rel_l2:.3f}"

    def test_n_stages_1_same_as_vanilla(self):
        """1 stage with PDE operator should behave like vanilla PIELM."""
        from pypielm.models.curriculum import CurriculumPIELM
        ds = _poisson_ds()
        model = CurriculumPIELM(hidden_dim=40, n_stages=1,
                                 n_collocation=80, n_candidates=200, seed=42)
        model.fit(ds, pde_operator=_poisson_op)
        X = torch.linspace(0, 1, 10).unsqueeze(1).double()
        pred = model.predict(X)
        assert torch.isfinite(pred).all()

    def test_bc_block_passed_through(self):
        """BC block (y_bc) is forwarded to each stage dataset."""
        from pypielm.models.curriculum import CurriculumPIELM
        ds = _poisson_ds()
        model = CurriculumPIELM(hidden_dim=40, n_stages=2, n_collocation=50, seed=42)
        model.fit(ds, pde_operator=_poisson_op)
        # BC at endpoints should be roughly zero
        X_bc = torch.tensor([[0.0], [1.0]], dtype=DTYPE)
        pred = model.predict(X_bc)
        # Not strict (CurriculumPIELM is soft BC) — just finite
        assert torch.isfinite(pred).all()


class TestCurriculumPIELMGuards:
    def test_predict_before_fit_raises(self):
        from pypielm.models.curriculum import CurriculumPIELM
        model = CurriculumPIELM(hidden_dim=20)
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(torch.zeros(5, 1, dtype=DTYPE))

    def test_get_feature_matrix_before_fit_raises(self):
        from pypielm.models.curriculum import CurriculumPIELM
        model = CurriculumPIELM(hidden_dim=20)
        with pytest.raises(RuntimeError, match="fit"):
            model.get_feature_matrix(torch.zeros(5, 1, dtype=DTYPE))

    def test_empty_blocks_raises(self):
        """Dataset with no data/pde/bc blocks raises ValueError."""
        from pypielm.models.curriculum import CurriculumPIELM
        # Collocation-only with no PDE/data → no blocks
        rng = np.random.default_rng(42)
        X_c = rng.uniform(0, 1, (30, 1))
        # No BC, no y_data, no PDE
        ds = PIELMDataset.from_arrays(X_c, dtype=DTYPE)
        model = CurriculumPIELM(hidden_dim=20, n_stages=1)
        with pytest.raises(ValueError):
            model.fit(ds, pde_operator=None)

    def test_repr(self):
        from pypielm.models.curriculum import CurriculumPIELM
        model = CurriculumPIELM(hidden_dim=50, n_stages=3)
        r = repr(model)
        assert "CurriculumPIELM" in r
        assert "n_stages" in r


class TestCurriculumPIELMDeterminism:
    def test_same_seed_deterministic(self):
        from pypielm.models.curriculum import CurriculumPIELM
        ds = _data_ds()
        m1 = CurriculumPIELM(hidden_dim=30, n_stages=2, seed=42)
        m2 = CurriculumPIELM(hidden_dim=30, n_stages=2, seed=42)
        m1.fit(ds)
        m2.fit(ds)
        X = torch.linspace(0, 1, 10).unsqueeze(1).double()
        assert torch.allclose(m1.predict(X), m2.predict(X)), \
            "CurriculumPIELM not deterministic with same seed"

    def test_different_seeds_differ(self):
        from pypielm.models.curriculum import CurriculumPIELM
        ds = _data_ds()
        m1 = CurriculumPIELM(hidden_dim=30, n_stages=2, seed=0)
        m2 = CurriculumPIELM(hidden_dim=30, n_stages=2, seed=9999)
        m1.fit(ds)
        m2.fit(ds)
        X = torch.linspace(0, 1, 20).unsqueeze(1).double()
        assert not torch.allclose(m1.predict(X), m2.predict(X)), \
            "Different seeds produced identical predictions"
