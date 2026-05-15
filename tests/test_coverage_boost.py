"""Targeted tests to push coverage above 90 %.

Covers all easily reachable branches that are not exercised by the main test
suite:

* ``core/base.py``  — every ``_compute_metric`` branch + ``_stack_blocks``
* ``models/vanilla.py``  — error paths, ``score()``, ``get_feature_matrix``,
  ``forward()``, ``__repr__``, X_ic/y_ic dataset path
* ``models/bayesian.py``  — ``predict_with_uncertainty``, ``score``,
  ``get_feature_matrix``, ``__repr__``, w_data override, no-blocks error
* ``io/export.py``  — ``to_onnx`` ImportError, ``to_torchscript`` trace /
  invalid-method
* ``utils/reproducibility.py``  — ``get_device`` branches,
  ``seed_everything`` deterministic flag
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_dataset(n: int = 20, d: int = 2, *, x_ic: bool = False):
    """Return a PIELMDataset with X_data / y_data (1-D → 2-D target)."""
    from pypielm.data.dataset import PIELMDataset

    X = torch.randn(n, d, dtype=torch.float64)
    y = torch.randn(n, 1, dtype=torch.float64)
    kw: dict = {"X_colloc": X, "X_data": X, "y_data": y}
    if x_ic:
        X_ic = torch.randn(5, d, dtype=torch.float64)
        y_ic = torch.randn(5, 1, dtype=torch.float64)
        kw["X_ic"] = X_ic
        kw["y_ic"] = y_ic
    return PIELMDataset(**kw)


# ---------------------------------------------------------------------------
# _compute_metric — all metric branches (core/base.py lines 200-218)
# ---------------------------------------------------------------------------

class TestComputeMetric:
    """Call score() with every metric to cover _compute_metric branches."""

    @pytest.fixture(autouse=True)
    def _fit(self):
        from pypielm.models import VanillaPIELM

        ds = _simple_dataset()
        self.model = VanillaPIELM(hidden_dim=12, seed=0, dtype=torch.float64)
        self.model.fit(ds)
        self.X = ds.X_data
        self.y = ds.y_data

    def test_score_default_relative_l2(self):
        s = self.model.score(self.X, self.y)
        assert isinstance(s, float)

    def test_score_rmse(self):
        s = self.model.score(self.X, self.y, metric="rmse")
        assert isinstance(s, float) and s >= 0

    def test_score_mae(self):
        s = self.model.score(self.X, self.y, metric="mae")
        assert isinstance(s, float) and s >= 0

    def test_score_r2(self):
        s = self.model.score(self.X, self.y, metric="r2")
        assert isinstance(s, float) and s <= 1.0

    def test_score_max_error(self):
        s = self.model.score(self.X, self.y, metric="max_error")
        assert isinstance(s, float) and s >= 0

    def test_score_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            self.model.score(self.X, self.y, metric="banana")

    def test_score_flat_target_triggers_reshape(self):
        """1-D y reshapes to match pred (line 200 in _compute_metric)."""
        y_flat = self.y.squeeze()
        s = self.model.score(self.X, y_flat, metric="relative_l2")
        assert isinstance(s, float)

    def test_score_zero_reference_triggers_safe_denom(self):
        """Zero-norm reference → safe denominator of 1 (line 204)."""
        y_zero = torch.zeros_like(self.y)
        s = self.model.score(self.X, y_zero, metric="relative_l2")
        assert isinstance(s, float)

    def test_score_r2_constant_target(self):
        """Constant target → ss_tot ~ 0, triggers ss_tot < 1e-30 branch."""
        y_const = torch.ones_like(self.y)
        s = self.model.score(self.X, y_const, metric="r2")
        # Should return 1.0 or 0.0, not raise
        assert isinstance(s, float)


# ---------------------------------------------------------------------------
# _stack_blocks — empty list (core/base.py line 234)
# ---------------------------------------------------------------------------

def test_stack_blocks_empty_raises():
    from pypielm.core.base import _stack_blocks

    with pytest.raises(ValueError, match="non-empty"):
        _stack_blocks([])


# ---------------------------------------------------------------------------
# VanillaPIELM — error / utility paths
# ---------------------------------------------------------------------------

class TestVanillaPIELMCoverage:

    def test_fit_no_y_data_raises(self):
        from pypielm.data.dataset import PIELMDataset
        from pypielm.models import VanillaPIELM

        ds = PIELMDataset(X_colloc=torch.randn(10, 2))
        model = VanillaPIELM(hidden_dim=8, seed=0)
        with pytest.raises(ValueError, match="y_data"):
            model.fit(ds)

    def test_predict_before_fit_raises(self):
        from pypielm.models import VanillaPIELM

        model = VanillaPIELM(hidden_dim=8, seed=0)
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(torch.zeros(2, 2))

    def test_get_feature_matrix_before_fit_raises(self):
        from pypielm.models import VanillaPIELM

        model = VanillaPIELM(hidden_dim=8, seed=0)
        with pytest.raises(RuntimeError, match="fit"):
            model.get_feature_matrix(torch.zeros(2, 2))

    def test_get_feature_matrix_after_fit(self):
        from pypielm.models import VanillaPIELM

        ds = _simple_dataset()
        model = VanillaPIELM(hidden_dim=8, seed=0, dtype=torch.float64)
        model.fit(ds)
        H = model.get_feature_matrix(ds.X_data)
        assert H.shape == (20, 8)

    def test_forward_interface(self):
        """nn.Module forward() → predict()."""
        from pypielm.models import VanillaPIELM

        ds = _simple_dataset()
        model = VanillaPIELM(hidden_dim=8, seed=0, dtype=torch.float64)
        model.fit(ds)
        out = model(ds.X_data)
        assert out.shape == (20, 1)

    def test_repr(self):
        from pypielm.models import VanillaPIELM

        r = repr(VanillaPIELM(hidden_dim=16, ridge_lambda=1e-5, seed=0))
        assert "VanillaPIELM" in r and "16" in r

    def test_collect_blocks_x_ic_y_ic_from_dataset(self):
        """X_ic / y_ic in dataset (no explicit ics arg) cover elif branch."""
        from pypielm.models import CorePIELM

        ds = _simple_dataset(x_ic=True)
        # Remove X_data so the IC block is the only block
        from pypielm.data.dataset import PIELMDataset
        ds_no_data = PIELMDataset(
            X_colloc=ds.X_colloc,
            X_ic=ds.X_ic,
            y_ic=ds.y_ic,
        )
        model = CorePIELM(hidden_dim=12, seed=0, dtype=torch.float64)
        model.fit(ds_no_data)
        pred = model.predict(ds.X_colloc)
        assert pred.shape[0] == 20


# ---------------------------------------------------------------------------
# CorePIELM — error / utility paths + rrqr solver
# ---------------------------------------------------------------------------

class TestCorePIELMCoverage:

    def test_predict_before_fit_raises(self):
        from pypielm.models import CorePIELM

        model = CorePIELM(hidden_dim=8, seed=0)
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(torch.zeros(2, 2))

    def test_get_feature_matrix_before_fit_raises(self):
        from pypielm.models import CorePIELM

        model = CorePIELM(hidden_dim=8, seed=0)
        with pytest.raises(RuntimeError, match="fit"):
            model.get_feature_matrix(torch.zeros(2, 2))

    def test_get_feature_matrix_after_fit(self):
        from pypielm.models import CorePIELM

        ds = _simple_dataset()
        model = CorePIELM(hidden_dim=8, seed=0, dtype=torch.float64)
        model.fit(ds)
        H = model.get_feature_matrix(ds.X_data)
        assert H.shape == (20, 8)

    def test_score_rmse(self):
        from pypielm.models import CorePIELM

        ds = _simple_dataset()
        model = CorePIELM(hidden_dim=8, seed=0, dtype=torch.float64)
        model.fit(ds)
        s = model.score(ds.X_data, ds.y_data, metric="rmse")
        assert isinstance(s, float) and s >= 0

    def test_no_blocks_raises(self):
        """fit() with no data / physics raises ValueError."""
        from pypielm.data.dataset import PIELMDataset
        from pypielm.models import CorePIELM

        ds = PIELMDataset(X_colloc=torch.randn(10, 2))
        model = CorePIELM(hidden_dim=8, seed=0)
        with pytest.raises(ValueError, match="No observation blocks"):
            model.fit(ds)

    def test_repr(self):
        from pypielm.models import CorePIELM

        r = repr(CorePIELM(hidden_dim=16, solver="rrqr", seed=0))
        assert "CorePIELM" in r and "rrqr" in r

    def test_forward_interface(self):
        from pypielm.models import CorePIELM

        ds = _simple_dataset()
        model = CorePIELM(hidden_dim=8, seed=0, dtype=torch.float64)
        model.fit(ds)
        out = model(ds.X_data)
        assert out.shape == (20, 1)

    def test_rrqr_solver(self):
        from pypielm.models import CorePIELM

        ds = _simple_dataset(n=30)
        model = CorePIELM(hidden_dim=8, solver="rrqr", seed=0, dtype=torch.float64)
        model.fit(ds)
        pred = model.predict(ds.X_data)
        assert pred.shape == (30, 1)


# ---------------------------------------------------------------------------
# BayesianPIELM — error / utility paths
# ---------------------------------------------------------------------------

class TestBayesianPIELMCoverage:

    @pytest.fixture
    def fitted(self):
        from pypielm.models import BayesianPIELM

        ds = _simple_dataset()
        model = BayesianPIELM(hidden_dim=16, seed=0, dtype=torch.float64)
        model.fit(ds)
        return model, ds

    def test_predict_with_uncertainty_shape(self, fitted):
        model, ds = fitted
        mean, std = model.predict_with_uncertainty(ds.X_data)
        assert mean.shape == (20, 1)
        assert std.shape == (20, 1)
        assert (std >= 0).all()

    def test_predict_with_uncertainty_multioutput(self):
        """Multi-output (out_dim > 1) triggers std.expand_as(mean)."""
        from pypielm.data.dataset import PIELMDataset
        from pypielm.models import BayesianPIELM

        X = torch.randn(20, 2, dtype=torch.float64)
        y = torch.randn(20, 2, dtype=torch.float64)  # 2 outputs
        ds = PIELMDataset(X_colloc=X, X_data=X, y_data=y)
        model = BayesianPIELM(hidden_dim=16, seed=0, dtype=torch.float64)
        model.fit(ds)
        mean, std = model.predict_with_uncertainty(X)
        assert mean.shape == (20, 2)
        assert std.shape == (20, 2)

    def test_predict_before_fit_raises(self):
        from pypielm.models import BayesianPIELM

        with pytest.raises(RuntimeError, match="fit"):
            BayesianPIELM(hidden_dim=8).predict(torch.zeros(2, 2))

    def test_predict_with_uncertainty_before_fit_raises(self):
        from pypielm.models import BayesianPIELM

        with pytest.raises(RuntimeError, match="fit"):
            BayesianPIELM(hidden_dim=8).predict_with_uncertainty(torch.zeros(2, 2))

    def test_score_r2(self, fitted):
        model, ds = fitted
        s = model.score(ds.X_data, ds.y_data, metric="r2")
        assert isinstance(s, float) and s <= 1.0

    def test_get_feature_matrix_before_fit_raises(self):
        from pypielm.models import BayesianPIELM

        with pytest.raises(RuntimeError, match="fit"):
            BayesianPIELM(hidden_dim=8).get_feature_matrix(torch.zeros(2, 2))

    def test_get_feature_matrix_after_fit(self, fitted):
        model, ds = fitted
        H = model.get_feature_matrix(ds.X_data)
        assert H.shape == (20, 16)

    def test_repr(self):
        from pypielm.models import BayesianPIELM

        r = repr(BayesianPIELM(hidden_dim=32, prior_precision=1e-3))
        assert "BayesianPIELM" in r and "32" in r

    def test_fit_w_data_override(self):
        """w_data != 1.0 triggers the weight-override branch in fit()."""
        from pypielm.models import BayesianPIELM

        ds = _simple_dataset()
        model = BayesianPIELM(hidden_dim=8, w_data=0.5, seed=0, dtype=torch.float64)
        model.fit(ds)
        pred = model.predict(ds.X_data)
        assert pred.shape == (20, 1)

    def test_fit_no_blocks_raises(self):
        """No data, no physics → ValueError."""
        from pypielm.data.dataset import PIELMDataset
        from pypielm.models import BayesianPIELM

        ds = PIELMDataset(X_colloc=torch.randn(10, 2))
        model = BayesianPIELM(hidden_dim=8)
        with pytest.raises(ValueError, match="No observation blocks"):
            model.fit(ds)


# ---------------------------------------------------------------------------
# io/export.py — to_onnx ImportError + to_torchscript branches
# ---------------------------------------------------------------------------

class TestExportCoverage:

    @pytest.fixture
    def simple_model(self):
        from pypielm.models import VanillaPIELM

        ds = _simple_dataset()
        m = VanillaPIELM(hidden_dim=8, seed=0, dtype=torch.float64)
        m.fit(ds)
        return m

    @pytest.mark.skipif(
        importlib.util.find_spec("onnx") is not None,
        reason="onnx is installed; ImportError path cannot be triggered",
    )
    def test_to_onnx_raises_import_error(self, simple_model, tmp_path):
        from pypielm.io.export import to_onnx

        with pytest.raises(ImportError, match="onnx"):
            to_onnx(simple_model, tmp_path / "model.onnx")

    def test_to_torchscript_invalid_method_raises(self, simple_model, tmp_path):
        from pypielm.io.export import to_torchscript

        with pytest.raises(ValueError, match="method must be"):
            to_torchscript(simple_model, tmp_path / "model.pt", method="invalid")

    def test_to_torchscript_trace(self, simple_model, tmp_path):
        from pypielm.io.export import to_torchscript

        scripted = to_torchscript(
            simple_model,
            tmp_path / "model.pt",
            example_input=torch.zeros(1, 2, dtype=torch.float64),
            method="trace",
        )
        assert scripted is not None
        out = scripted(torch.zeros(3, 2, dtype=torch.float64))
        assert out.shape[0] == 3


# ---------------------------------------------------------------------------
# utils/reproducibility.py — device selection and seed_everything
# ---------------------------------------------------------------------------

class TestReproducibilityUtils:

    def test_get_device_mps_or_cpu(self):
        """prefer_cuda=False should return MPS (Apple Silicon) or CPU."""
        from pypielm.utils.reproducibility import get_device

        dev = get_device(prefer_cuda=False, prefer_mps=True)
        assert dev.type in ("mps", "cpu")

    def test_get_device_cpu_only(self):
        from pypielm.utils.reproducibility import get_device

        dev = get_device(prefer_cuda=False, prefer_mps=False)
        assert dev.type == "cpu"

    def test_seed_everything_deterministic_true(self):
        from pypielm.utils.reproducibility import seed_everything

        seed_everything(42, deterministic=True)   # must not raise

    def test_seed_everything_deterministic_false(self):
        from pypielm.utils.reproducibility import seed_everything

        seed_everything(0, deterministic=False)   # must not raise


# ---------------------------------------------------------------------------
# Final 90 % push: cover specific uncovered branches
# ---------------------------------------------------------------------------

class TestFinalCoverageBoost:
    """Target the specific lines still uncovered after the prior test run."""

    # vanilla.py line 230 — CorePIELM invalid solver guard
    def test_core_pielm_invalid_solver_raises(self):
        from pypielm.models import CorePIELM

        with pytest.raises(ValueError, match="solver must be"):
            CorePIELM(solver="invalid")

    # vanilla.py lines 69-71 — explicit bcs list passed to CorePIELM.fit
    def test_core_pielm_fit_with_explicit_bcs(self):
        from pypielm.data.dataset import PIELMDataset
        from pypielm.models import CorePIELM
        from pypielm.pde.constraints import DirichletBC

        X_bc = torch.zeros(4, 1, dtype=torch.float64)
        bc = DirichletBC(
            boundary_fn=lambda x: torch.zeros(x.shape[0], dtype=x.dtype),
            points=X_bc,
        )
        X = torch.linspace(0.0, 1.0, 20, dtype=torch.float64).unsqueeze(1)
        ds = PIELMDataset(X_colloc=X)
        model = CorePIELM(hidden_dim=10, seed=0, dtype=torch.float64)
        model.fit(ds, bcs=[bc])
        assert model.predict(X).shape == (20, 1)

    # vanilla.py lines 79-81 — explicit ics list passed to CorePIELM.fit
    def test_core_pielm_fit_with_explicit_ics(self):
        from pypielm.data.dataset import PIELMDataset
        from pypielm.models import CorePIELM
        from pypielm.pde.constraints import InitialCondition

        X_ic = torch.zeros(4, 1, dtype=torch.float64)
        ic = InitialCondition(
            ic_fn=lambda x: torch.zeros(x.shape[0], dtype=x.dtype),
            points=X_ic,
        )
        X = torch.linspace(0.0, 1.0, 20, dtype=torch.float64).unsqueeze(1)
        ds = PIELMDataset(X_colloc=X)
        model = CorePIELM(hidden_dim=10, seed=0, dtype=torch.float64)
        model.fit(ds, ics=[ic])
        assert model.predict(X).shape == (20, 1)

    # pde/constraints.py line 120 — NeumannBC with 1D flux_fn (triggers unsqueeze)
    def test_neumann_bc_1d_flux_fn(self):
        from pypielm.core.feature_maps import RandomFeatureMap
        from pypielm.pde.constraints import NeumannBC

        pts = torch.linspace(0.0, 1.0, 4, dtype=torch.float64).unsqueeze(1)
        normal = torch.ones(4, 1, dtype=torch.float64)
        bc = NeumannBC(
            flux_fn=lambda x: torch.zeros(x.shape[0], dtype=x.dtype),  # 1-D output → triggers unsqueeze
            normal=normal,
            points=pts,
        )
        fm = RandomFeatureMap(input_dim=1, hidden_dim=8, seed=0, dtype=torch.float64)
        sys = bc.assemble(fm)
        assert sys.y.ndim == 2

    # pde/constraints.py line 154 — InitialCondition with 1D ic_fn (triggers unsqueeze)
    def test_initial_condition_1d_ic_fn(self):
        from pypielm.core.feature_maps import RandomFeatureMap
        from pypielm.pde.constraints import InitialCondition

        pts = torch.linspace(0.0, 1.0, 4, dtype=torch.float64).unsqueeze(1)
        ic = InitialCondition(
            ic_fn=lambda x: torch.zeros(x.shape[0], dtype=x.dtype),  # 1-D output → triggers unsqueeze
            points=pts,
        )
        fm = RandomFeatureMap(input_dim=1, hidden_dim=8, seed=0, dtype=torch.float64)
        sys = ic.assemble(fm)
        assert sys.y.ndim == 2

    # pde/collocation.py lines 56-58 — UnionDomain empty list raises ValueError
    def test_union_domain_empty_raises(self):
        from pypielm.pde.collocation import UnionDomain

        with pytest.raises(ValueError, match="at least one domain"):
            UnionDomain([])

    # pde/collocation.py line 62 — UnionDomain.dim property
    def test_union_domain_dim(self):
        from pypielm.pde.collocation import BoxDomain, UnionDomain

        d = BoxDomain(lb=torch.zeros(2), ub=torch.ones(2))
        ud = UnionDomain([d])
        assert ud.dim == 2

    # pde/operators.py line 27 — _ensure_2d: gradient called with 1-D u
    def test_gradient_1d_u_input(self):
        from pypielm.pde.operators import gradient

        x = torch.linspace(0.1, 0.9, 15, dtype=torch.float64).unsqueeze(1)
        x.requires_grad_(True)
        u = torch.sin(x[:, 0])  # shape (15,) — 1-D, triggers _ensure_2d
        g = gradient(u, x)
        assert g.shape == (15, 1)

    # pde/operators.py lines 212-216 — AnalyticLaplacian d2 fallback (no laplacian attr)
    def test_analytic_laplacian_d2_fallback(self):
        from pypielm.pde.operators import AnalyticLaplacian

        class _D2OnlyFM:
            """Minimal feature map: has d2 but no laplacian method."""
            hidden_dim = 6

            def __call__(self, X):
                return torch.ones(X.shape[0], self.hidden_dim, dtype=X.dtype)

            def d2(self, X, k):
                return torch.zeros(X.shape[0], self.hidden_dim, dtype=X.dtype)

        al = AnalyticLaplacian(feature_map=_D2OnlyFM(), input_dim=2)
        X = torch.ones(5, 2, dtype=torch.float64)
        result = al(X)
        assert result.shape == (5, 6)

    # io/checkpoint.py line 23 — _get_model_registry_name fallback for unregistered class
    def test_checkpoint_unregistered_model_name(self):
        from pypielm.io.checkpoint import _get_model_registry_name
        from pypielm.models import VanillaPIELM

        # Subclass without @register → not in MODEL_REGISTRY
        class _MyCustomModel(VanillaPIELM):
            pass

        name = _get_model_registry_name(_MyCustomModel(hidden_dim=4))
        # fallback returns "module.ClassName"
        assert "." in name and "_MyCustomModel" in name

    # pde/collocation.py lines 142-149 — LatinHypercubeSampler (requires scipy)
    def test_lhs_sampler_produces_points(self):
        from pypielm.pde.collocation import BoxDomain, LHSSampler

        domain = BoxDomain(lb=torch.zeros(2), ub=torch.ones(2))
        sampler = LHSSampler(domain=domain, n_points=30, seed=7)
        pts = sampler.sample()
        assert pts.shape == (30, 2)
        assert (pts >= 0).all() and (pts <= 1).all()
