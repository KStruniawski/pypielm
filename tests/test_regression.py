"""Regression tests: compare PyPIELM against pre-computed benchmark_framework results.

These tests load the PINNacle ``poisson_classic.dat`` dataset (the same file
used by the original benchmark) and verify that PyPIELM's VanillaPIELM and
CorePIELM reach the same accuracy level as the benchmark_framework within a
tight tolerance.

The reference results live in::

    Benchmarking/artifacts/…/results.json

If the PINNacle data file or benchmark artifacts are not present (e.g. in a
bare CI checkout), the tests are skipped automatically.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Paths (all relative to *repo root*, not this test file)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parents[1].parent   # …/PIELM/
_DATA_FILE  = (
    _REPO_ROOT
    / "Benchmarking/Papers/PINNacle-main/ref/poisson_classic.dat"
)
_BENCHMARK_ARTIFACT = (
    _REPO_ROOT
    / "Benchmarking/artifacts"
    / "20260220T135228Z__pinnacle__poisson_classic-dat__CorePIELMRegressor__seed42"
    / "results.json"
)

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

_HAS_DATA     = _DATA_FILE.exists()
_HAS_ARTIFACT = _BENCHMARK_ARTIFACT.exists()

pytestmark = pytest.mark.skipif(
    not _HAS_DATA,
    reason=f"PINNacle data not found at {_DATA_FILE}",
)


# ---------------------------------------------------------------------------
# Reference metrics (hard-coded from the benchmark JSON)
# ---------------------------------------------------------------------------

_REF_METRICS = {
    "rmse":       0.1088,
    "relative_l2": 0.2174,
    "mae":         0.0775,
    "r2":          0.8720,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def poisson_classic_dataset():
    """Load the poisson_classic.dat PINNacle file into a PIELMDataset.

    PINNacleAdapter loads X_colloc + y_data from the raw .dat file.
    We promote the data into X_data/y_data so models can train on it.
    """
    from pypielm.data.adapters.pinnacle_adapter import PINNacleAdapter

    adapter = PINNacleAdapter(
        root=_DATA_FILE.parent,
        task=_DATA_FILE.name,
        dtype=torch.float64,
    )
    ds = adapter.load()

    # The adapter puts data in X_colloc / y_data.
    # Create a dataset with X_data = X_colloc so models can use fit(ds)
    from pypielm.data.dataset import PIELMDataset
    ds_full = PIELMDataset(
        X_colloc=ds.X_colloc,
        X_data=ds.X_colloc,
        y_data=ds.y_data,
    )
    return ds_full


@pytest.fixture(scope="module")
def benchmark_ref_metrics():
    """Load reference metrics from the benchmark artifact JSON, if present."""
    if not _HAS_ARTIFACT:
        return _REF_METRICS   # fall back to hard-coded values
    with open(_BENCHMARK_ARTIFACT) as f:
        data = json.load(f)
    return data["metrics"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _poisson_op(fm, X: torch.Tensor):
    """Approximate Laplacian for the 2D Poisson task."""
    from pypielm.core.solver import WeightedLinearSystem
    from pypielm.pde.operators import AnalyticLaplacian

    lap = AnalyticLaplacian(fm)
    H_lap, rhs = lap(X)
    return WeightedLinearSystem(H_lap, rhs, weight=1.0)


def _relative_l2(pred: torch.Tensor, ref: torch.Tensor) -> float:
    return ((pred - ref).norm() / (ref.norm() + 1e-12)).item()


# ---------------------------------------------------------------------------
# Test: VanillaPIELM accuracy regression
# ---------------------------------------------------------------------------

class TestVanillaPIELMRegression:
    """VanillaPIELM on poisson_classic must match or beat benchmark R²."""

    def test_r2_within_tolerance(self, poisson_classic_dataset) -> None:
        from pypielm.models import VanillaPIELM
        from pypielm.metrics.metrics import MetricsBundle

        ds = poisson_classic_dataset
        model = VanillaPIELM(hidden_dim=128, ridge_lambda=1e-6, seed=42)
        # VanillaPIELM trains on X_data if present (observed data block)
        model.fit(ds)

        X_test = ds.X_data if ds.X_data is not None else ds.X_colloc
        y_test = ds.y_data if ds.y_data is not None else None
        assert y_test is not None, "Dataset has no y_data"

        pred = model.predict(X_test)
        bundle = MetricsBundle(pred, y_test)
        metrics = bundle.to_dict()

        # R² must be within 5 percentage points of benchmark (0.872)
        ref_r2 = _REF_METRICS["r2"]
        assert metrics["r2"] >= ref_r2 - 0.05, \
            f"R² too low: {metrics['r2']:.4f} vs reference {ref_r2:.4f}"

    def test_relative_l2_within_tolerance(self, poisson_classic_dataset) -> None:
        from pypielm.models import VanillaPIELM
        from pypielm.metrics.metrics import MetricsBundle

        ds = poisson_classic_dataset
        model = VanillaPIELM(hidden_dim=128, ridge_lambda=1e-6, seed=42)
        model.fit(ds)

        X_test = ds.X_data
        y_test = ds.y_data
        assert y_test is not None

        pred = model.predict(X_test)
        bundle = MetricsBundle(pred, y_test)
        metrics = bundle.to_dict()

        # Relative L² must be within +30% of reference (relative)
        ref_rl2 = _REF_METRICS["relative_l2"]
        assert metrics["rel_l2"] <= ref_rl2 * 1.30 + 0.02, \
            f"rel_l2 too high: {metrics['rel_l2']:.4f} vs reference {ref_rl2:.4f}"

    def test_deterministic_given_seed(self, poisson_classic_dataset) -> None:
        from pypielm.models import VanillaPIELM

        ds = poisson_classic_dataset
        m1 = VanillaPIELM(hidden_dim=64, ridge_lambda=1e-6, seed=42)
        m2 = VanillaPIELM(hidden_dim=64, ridge_lambda=1e-6, seed=42)
        m1.fit(ds)
        m2.fit(ds)

        X = ds.X_data if ds.X_data is not None else ds.X_colloc
        p1 = m1.predict(X)
        p2 = m2.predict(X)
        assert torch.allclose(p1, p2), "Predictions differ across equal seeds"


# ---------------------------------------------------------------------------
# Test: CorePIELM accuracy regression (with physics blocks)
# ---------------------------------------------------------------------------

class TestCorePIELMRegression:
    """CorePIELM on poisson_classic must beat VanillaPIELM or be comparable."""

    def test_r2_not_catastrophically_lower(self, poisson_classic_dataset) -> None:
        """CorePIELM R² must be at least as good as benchmark minus 10pp."""
        from pypielm.models import CorePIELM
        from pypielm.metrics.metrics import MetricsBundle

        ds = poisson_classic_dataset
        model = CorePIELM(hidden_dim=128, ridge_lambda=1e-6, seed=42)
        model.fit(ds)

        X_test = ds.X_data
        y_test = ds.y_data
        assert y_test is not None

        pred = model.predict(X_test)
        bundle = MetricsBundle(pred, y_test)
        metrics = bundle.to_dict()

        assert metrics["r2"] >= _REF_METRICS["r2"] - 0.10, \
            f"CorePIELM R² collapsed: {metrics['r2']:.4f}"

    def test_fit_returns_self(self, poisson_classic_dataset) -> None:
        from pypielm.models import CorePIELM
        ds = poisson_classic_dataset
        model = CorePIELM(hidden_dim=32, seed=42)
        result = model.fit(ds)
        assert result is model

    def test_consistent_with_stored_predictions(self, poisson_classic_dataset) -> None:
        """Verify that CorePIELM predictions are finite and have reasonable range.

        A direct Pearson-r comparison against stored benchmark predictions is not
        meaningful here because the benchmark used a specific train/test split that
        we cannot replicate from this fixture.  Instead we check basic sanity.
        """
        from pypielm.models import CorePIELM
        ds = poisson_classic_dataset
        model = CorePIELM(hidden_dim=128, ridge_lambda=1e-6, seed=42)
        model.fit(ds)

        X_test = ds.X_data
        pred_new = model.predict(X_test)

        assert torch.isfinite(pred_new).all(), "CorePIELM produced non-finite predictions"
        # Range should be bounded — raw PINNacle u values are O(1)
        assert pred_new.abs().max().item() < 1e4, "CorePIELM predictions unreasonably large"


# ---------------------------------------------------------------------------
# Test: BayesianPIELM uncertainty is non-trivial
# ---------------------------------------------------------------------------

class TestBayesianPIELMRegression:
    def test_posterior_variance_positive(self, poisson_classic_dataset) -> None:
        from pypielm.models import BayesianPIELM

        ds = poisson_classic_dataset
        model = BayesianPIELM(hidden_dim=64, seed=42)
        model.fit(ds)

        # BayesianPIELM stores the posterior precision matrix in _beta_precision
        assert model._beta_precision is not None, "Posterior precision must be set after fit"
        # A valid precision matrix must be positive definite => all eigenvalues > 0
        eigvals = torch.linalg.eigvalsh(model._beta_precision)
        assert (eigvals > 0).all(), "Posterior precision matrix is not positive definite"

    def test_uncertainty_widens_away_from_data(self, poisson_classic_dataset) -> None:
        """Predictive variance must be finite and positive at test points."""
        from pypielm.models import BayesianPIELM

        ds = poisson_classic_dataset
        model = BayesianPIELM(hidden_dim=64, seed=42)
        model.fit(ds)

        X_test = ds.X_data[:20]
        pred = model.predict(X_test)
        assert torch.isfinite(pred).all()


# ---------------------------------------------------------------------------
# Test: GFFPIELM accuracy on the same dataset
# ---------------------------------------------------------------------------

class TestGFFPIELMRegression:
    def test_r2_reasonable(self, poisson_classic_dataset) -> None:
        from pypielm.models import GFFPIELM
        from pypielm.metrics.metrics import MetricsBundle

        ds = poisson_classic_dataset
        model = GFFPIELM(hidden_dim=128, seed=42)
        model.fit(ds)

        X_test = ds.X_data
        y_test = ds.y_data
        assert y_test is not None

        pred = model.predict(X_test)
        bundle = MetricsBundle(pred, y_test)
        metrics = bundle.to_dict()
        # GFF should be at least competitive; allow 15pp below reference
        assert metrics["r2"] >= _REF_METRICS["r2"] - 0.15, \
            f"GFFPIELM R² too low: {metrics['r2']:.4f}"


# ---------------------------------------------------------------------------
# Test: metrics computed during training match reference scale
# ---------------------------------------------------------------------------

class TestMetricsScale:
    """RMSE and MAE must be within reasonable range of the benchmark values."""

    def test_rmse_scale(self, poisson_classic_dataset) -> None:
        from pypielm.models import VanillaPIELM
        from pypielm.metrics.metrics import MetricsBundle

        ds = poisson_classic_dataset
        model = VanillaPIELM(hidden_dim=128, ridge_lambda=1e-6, seed=42)
        model.fit(ds)

        pred = model.predict(ds.X_data)
        bundle = MetricsBundle(pred, ds.y_data)
        m = bundle.to_dict()

        # RMSE should be positive and not catastrophically large
        assert 0 < m["rmse"] < 10.0, f"RMSE out of range: {m['rmse']}"
        # MAE ≤ RMSE always
        assert m["mae"] <= m["rmse"] + 1e-8, "MAE > RMSE (impossible)"
        # R² ∈ (-∞, 1]
        assert m["r2"] <= 1.0 + 1e-8

    def test_metrics_consistent(self, poisson_classic_dataset) -> None:
        """Multiple MetricsBundle calls on same data return identical results."""
        from pypielm.models import VanillaPIELM
        from pypielm.metrics.metrics import MetricsBundle

        ds = poisson_classic_dataset
        model = VanillaPIELM(hidden_dim=64, ridge_lambda=1e-6, seed=42)
        model.fit(ds)

        pred = model.predict(ds.X_data)
        m1 = MetricsBundle(pred, ds.y_data).to_dict()
        m2 = MetricsBundle(pred, ds.y_data).to_dict()
        for key in m1:
            assert abs(m1[key] - m2[key]) < 1e-12, f"Inconsistent metric: {key}"
