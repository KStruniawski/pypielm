"""Tests for pypielm.pde operators, samplers, and BC/IC constraints."""

from __future__ import annotations

import math

import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_2d_grid(n: int = 30, dtype=torch.float64, requires_grad: bool = True):
    """Return a regular (n*n, 2) grid over [0,1]^2 with grad tracking."""
    xs = torch.linspace(0.0, 1.0, n, dtype=dtype)
    ys = torch.linspace(0.0, 1.0, n, dtype=dtype)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    X = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    X.requires_grad_(requires_grad)
    return X


def _make_1d_pts(n: int = 50, dtype=torch.float64, requires_grad: bool = True):
    pts = torch.linspace(0.1, 0.9, n, dtype=dtype).unsqueeze(1)
    pts.requires_grad_(requires_grad)
    return pts


# ===========================================================================
# Differential operators
# ===========================================================================

class TestGradient:
    """Tests for pypielm.pde.operators.gradient."""

    def test_shape_1d(self):
        from pypielm.pde.operators import gradient
        x = _make_1d_pts(20)
        u = torch.sin(x[:, 0:1])
        g = gradient(u, x)
        assert g.shape == (20, 1)

    def test_shape_2d(self):
        from pypielm.pde.operators import gradient
        X = _make_2d_grid(10)
        u = (X[:, 0:1] * X[:, 1:2])
        g = gradient(u, X)
        assert g.shape == (100, 2)

    def test_values_linear_1d(self):
        """grad(a*x) should be constant a."""
        from pypielm.pde.operators import gradient
        a = 3.7
        x = _make_1d_pts(15)
        u = a * x
        g = gradient(u, x)
        assert g.shape == (15, 1)
        assert torch.allclose(g, torch.full_like(g, a), atol=1e-10)

    def test_values_product_2d(self):
        """grad(x*y) = [y, x]."""
        from pypielm.pde.operators import gradient
        X = _make_2d_grid(8)
        u = X[:, 0:1] * X[:, 1:2]
        g = gradient(u, X)
        assert torch.allclose(g[:, 0:1], X[:, 1:2].detach(), atol=1e-10)
        assert torch.allclose(g[:, 1:2], X[:, 0:1].detach(), atol=1e-10)

    def test_values_sin_cos_1d(self):
        """grad(sin(π x)) ≈ π cos(π x)."""
        from pypielm.pde.operators import gradient
        x = _make_1d_pts(40)
        u = torch.sin(math.pi * x)
        g = gradient(u, x)
        expected = math.pi * torch.cos(math.pi * x.detach())
        assert torch.allclose(g, expected, atol=1e-6)


class TestLaplacian:
    """Tests for pypielm.pde.operators.laplacian."""

    def test_shape_1d(self):
        from pypielm.pde.operators import laplacian
        x = _make_1d_pts(20)
        u = x ** 2
        lap = laplacian(u, x)
        assert lap.shape == (20, 1)

    def test_shape_2d(self):
        from pypielm.pde.operators import laplacian
        X = _make_2d_grid(8)
        u = (X[:, 0:1] ** 2 + X[:, 1:2] ** 2)
        lap = laplacian(u, X)
        assert lap.shape == (64, 1)

    def test_constant_field_is_zero(self):
        """Laplacian of constant = 0."""
        from pypielm.pde.operators import laplacian
        X = _make_2d_grid(6)
        u = torch.ones(36, 1, dtype=torch.float64)
        u = u + 0.0 * X.sum(dim=1, keepdim=True)  # tie to X for grad
        lap = laplacian(u, X)
        assert torch.allclose(lap, torch.zeros_like(lap), atol=1e-12)

    def test_laplacian_linear_is_zero(self):
        """Laplacian of a*x + b*y + c = 0."""
        from pypielm.pde.operators import laplacian
        X = _make_2d_grid(6)
        u = 2.0 * X[:, 0:1] - 3.0 * X[:, 1:2] + 7.0
        lap = laplacian(u, X)
        assert torch.allclose(lap, torch.zeros_like(lap), atol=1e-10)

    def test_laplacian_quadratic_1d(self):
        """Laplacian of x² = 2 in 1D."""
        from pypielm.pde.operators import laplacian
        x = _make_1d_pts(20)
        u = x ** 2
        lap = laplacian(u, x)
        assert torch.allclose(lap, torch.full_like(lap, 2.0), atol=1e-9)

    def test_laplacian_quadratic_2d(self):
        """Laplacian of (x² + y²) = 4 in 2D (∂²/∂x² = 2, ∂²/∂y² = 2)."""
        from pypielm.pde.operators import laplacian
        X = _make_2d_grid(8)
        u = X[:, 0:1] ** 2 + X[:, 1:2] ** 2
        lap = laplacian(u, X)
        assert torch.allclose(lap, torch.full_like(lap, 4.0), atol=1e-9)

    def test_laplacian_sin_product_2d(self):
        """Laplacian(sin(π x)sin(π y)) = -2π² sin(πx)sin(πy), atol=1e-4."""
        from pypielm.pde.operators import laplacian
        # Avoid boundary nodes to skip sin=0 trivial rows; use interior grid
        xs = torch.linspace(0.05, 0.95, 20, dtype=torch.float64)
        ys = torch.linspace(0.05, 0.95, 20, dtype=torch.float64)
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        X = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
        X.requires_grad_(True)
        u = torch.sin(math.pi * X[:, 0:1]) * torch.sin(math.pi * X[:, 1:2])
        lap = laplacian(u, X)
        expected = -2.0 * math.pi ** 2 * torch.sin(math.pi * X[:, 0:1].detach()) * torch.sin(math.pi * X[:, 1:2].detach())
        assert torch.allclose(lap, expected, atol=1e-4), (
            f"Max error: {(lap - expected).abs().max().item():.6e}"
        )


class TestDivergence:
    """Tests for pypielm.pde.operators.divergence."""

    def test_shape_2d(self):
        from pypielm.pde.operators import divergence
        X = _make_2d_grid(6)
        # flux = [x, y] → divergence = 2
        flux = X.clone()
        div = divergence(flux, X)
        assert div.shape == (36, 1)

    def test_div_constant_flux_is_zero(self):
        """Divergence of constant vector field = 0."""
        from pypielm.pde.operators import divergence
        X = _make_2d_grid(6)
        flux = torch.ones_like(X)
        # Need to attach flux to X for autograd
        flux = flux + 0.0 * X
        div = divergence(flux, X)
        assert torch.allclose(div, torch.zeros_like(div), atol=1e-10)

    def test_div_linear_field_2d(self):
        """Divergence of [x, y] = 2 in 2D."""
        from pypielm.pde.operators import divergence
        X = _make_2d_grid(5)
        flux = X.clone()  # [x, y]
        div = divergence(flux, X)
        assert torch.allclose(div, torch.full_like(div, 2.0), atol=1e-9)

    def test_div_1d(self):
        """Divergence of [x] = 1 in 1D."""
        from pypielm.pde.operators import divergence
        x = _make_1d_pts(15)
        flux = x.clone()
        div = divergence(flux, x)
        assert torch.allclose(div, torch.ones_like(div), atol=1e-9)


class TestAdvectionTerm:
    """Tests for pypielm.pde.operators.advection_term."""

    def test_shape_2d(self):
        from pypielm.pde.operators import advection_term
        X = _make_2d_grid(6)
        u = X[:, 0:1] * X[:, 1:2]
        v = torch.ones_like(X)
        v = v + 0.0 * X
        adv = advection_term(u, v, X)
        assert adv.shape == (36, 1)

    def test_zero_velocity(self):
        """Advection term with v=0 should be 0."""
        from pypielm.pde.operators import advection_term
        X = _make_2d_grid(5)
        u = X[:, 0:1] ** 2
        v = torch.zeros_like(X)
        v = v + 0.0 * X  # keep in graph
        adv = advection_term(u, v, X)
        assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-12)

    def test_uniform_flow_linear_u(self):
        """v=(1,0), u=x  → (v·∇)u = 1."""
        from pypielm.pde.operators import advection_term
        X = _make_2d_grid(6)
        u = X[:, 0:1]
        v = torch.zeros_like(X)
        v[:, 0] = 1.0
        v = v + 0.0 * X
        adv = advection_term(u, v, X)
        assert torch.allclose(adv, torch.ones_like(adv), atol=1e-9)


class TestAnalyticLaplacian:
    """Tests for pypielm.pde.operators.AnalyticLaplacian."""

    def test_analytic_matches_autograd_2d(self):
        """AnalyticLaplacian must agree with autograd laplacian within 1e-5.

        We compare column-by-column using the feature-map's analytic Laplacian
        against applying the scalar autograd laplacian to each output neuron.
        """
        from pypielm.pde.operators import AnalyticLaplacian, laplacian as autograd_lap
        from pypielm.core.feature_maps import FourierFeatureMap

        fm = FourierFeatureMap(input_dim=2, hidden_dim=32, freq_max=10.0, seed=7)
        X_raw = _make_2d_grid(6, requires_grad=False)

        analytic = AnalyticLaplacian(feature_map=fm)
        H_lap_analytic = analytic(X_raw)  # (N, hidden_dim) — purely analytic

        # Build autograd Laplacian column-by-column using scalar laplacian
        n = X_raw.shape[0]
        hidden = fm.hidden_dim
        H_lap_ag = torch.zeros(n, hidden, dtype=torch.float64)
        for j in range(hidden):
            X_j = X_raw.clone().requires_grad_(True)
            H_col = fm(X_j)[:, j : j + 1]  # (N, 1)
            lap_j = autograd_lap(H_col, X_j)  # (N, 1)
            H_lap_ag[:, j : j + 1] = lap_j.detach()

        assert torch.allclose(H_lap_analytic, H_lap_ag, atol=1e-5), (
            f"Max diff: {(H_lap_analytic - H_lap_ag).abs().max().item():.6e}"
        )

    def test_analytic_laplacian_no_fm_raises(self):
        """AnalyticLaplacian without feature_map should raise on call."""
        from pypielm.pde.operators import AnalyticLaplacian
        al = AnalyticLaplacian()
        X = _make_2d_grid(4, requires_grad=False)
        with pytest.raises((NotImplementedError, AttributeError, ValueError)):
            al(X)

    def test_analytic_laplacian_shape(self):
        from pypielm.pde.operators import AnalyticLaplacian
        from pypielm.core.feature_maps import FourierFeatureMap
        fm = FourierFeatureMap(input_dim=2, hidden_dim=32, freq_max=10.0, seed=0)
        al = AnalyticLaplacian(feature_map=fm)
        X = _make_2d_grid(5, requires_grad=False)
        out = al(X)
        assert out.shape == (25, 32)


# ===========================================================================
# Collocation samplers
# ===========================================================================

class TestBoxDomain:
    def test_dim(self):
        from pypielm.pde.collocation import BoxDomain
        d = BoxDomain([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        assert d.dim == 3

    def test_dtype_converted_to_float64(self):
        from pypielm.pde.collocation import BoxDomain
        d = BoxDomain([0, 1], [1, 2])
        assert d.lb.dtype == torch.float64
        assert d.ub.dtype == torch.float64


class TestUniformSampler:
    def test_shape(self):
        from pypielm.pde.collocation import BoxDomain, UniformSampler
        dom = BoxDomain([0.0, 0.0], [1.0, 1.0])
        pts = UniformSampler(dom, n_points=200).sample()
        assert pts.shape == (200, 2)

    def test_within_bounds_2d(self):
        from pypielm.pde.collocation import BoxDomain, UniformSampler
        lb, ub = [0.5, -1.0], [1.5, 2.0]
        dom = BoxDomain(lb, ub)
        pts = UniformSampler(dom, n_points=500).sample()
        assert (pts[:, 0] >= 0.5).all()
        assert (pts[:, 0] <= 1.5).all()
        assert (pts[:, 1] >= -1.0).all()
        assert (pts[:, 1] <= 2.0).all()

    def test_reproducible(self):
        from pypielm.pde.collocation import BoxDomain, UniformSampler
        dom = BoxDomain([0.0], [1.0])
        p1 = UniformSampler(dom, 100, seed=99).sample()
        p2 = UniformSampler(dom, 100, seed=99).sample()
        assert torch.equal(p1, p2)

    def test_different_seeds_differ(self):
        from pypielm.pde.collocation import BoxDomain, UniformSampler
        dom = BoxDomain([0.0], [1.0])
        p1 = UniformSampler(dom, 100, seed=1).sample()
        p2 = UniformSampler(dom, 100, seed=2).sample()
        assert not torch.equal(p1, p2)

    def test_1d_domain(self):
        from pypielm.pde.collocation import BoxDomain, UniformSampler
        dom = BoxDomain([0.0], [1.0])
        pts = UniformSampler(dom, 50).sample()
        assert pts.shape == (50, 1)
        assert (pts >= 0.0).all() and (pts <= 1.0).all()

    def test_dtype_float64(self):
        from pypielm.pde.collocation import BoxDomain, UniformSampler
        dom = BoxDomain([0.0, 0.0], [1.0, 1.0])
        pts = UniformSampler(dom, 10).sample()
        assert pts.dtype == torch.float64


class TestLHSSampler:
    def test_shape(self):
        from pypielm.pde.collocation import BoxDomain, LHSSampler
        dom = BoxDomain([0.0, 0.0], [1.0, 1.0])
        pts = LHSSampler(dom, n_points=100).sample()
        assert pts.shape == (100, 2)

    def test_within_bounds(self):
        from pypielm.pde.collocation import BoxDomain, LHSSampler
        lb, ub = [0.3, -2.0], [1.7, 3.0]
        dom = BoxDomain(lb, ub)
        pts = LHSSampler(dom, n_points=200).sample()
        assert (pts[:, 0] >= 0.3).all()
        assert (pts[:, 0] <= 1.7).all()
        assert (pts[:, 1] >= -2.0).all()
        assert (pts[:, 1] <= 3.0).all()

    def test_reproducible(self):
        from pypielm.pde.collocation import BoxDomain, LHSSampler
        dom = BoxDomain([0.0, 0.0], [1.0, 1.0])
        p1 = LHSSampler(dom, 80, seed=5).sample()
        p2 = LHSSampler(dom, 80, seed=5).sample()
        assert torch.allclose(p1, p2)

    def test_marginal_uniformity_ks(self):
        """Each marginal of LHS should look uniform: KS stat < 0.15."""
        pytest.importorskip("scipy")
        from scipy.stats import kstest
        from pypielm.pde.collocation import BoxDomain, LHSSampler
        dom = BoxDomain([0.0, 0.0], [1.0, 1.0])
        pts = LHSSampler(dom, n_points=500, seed=42).sample().numpy()
        for d in range(2):
            stat, pval = kstest(pts[:, d], "uniform")
            assert stat < 0.15, f"axis {d}: KS stat = {stat:.4f}"

    def test_dtype_float64(self):
        from pypielm.pde.collocation import BoxDomain, LHSSampler
        dom = BoxDomain([0.0], [1.0])
        pts = LHSSampler(dom, 10).sample()
        assert pts.dtype == torch.float64


class TestAdaptiveSampler:
    def test_shape(self):
        from pypielm.pde.collocation import BoxDomain, AdaptiveSampler
        dom = BoxDomain([0.0, 0.0], [1.0, 1.0])
        residual_fn = lambda X: (X[:, 0] - 0.5) ** 2 + (X[:, 1] - 0.5) ** 2
        pts = AdaptiveSampler(dom, residual_fn, n_points=100).sample()
        assert pts.shape == (100, 2)

    def test_respects_n_points(self):
        from pypielm.pde.collocation import BoxDomain, AdaptiveSampler
        dom = BoxDomain([0.0], [1.0])
        for n in [50, 200]:
            pts = AdaptiveSampler(
                dom, lambda X: X[:, 0], n_points=n, n_candidates=500
            ).sample()
            assert pts.shape[0] == n

    def test_within_domain(self):
        from pypielm.pde.collocation import BoxDomain, AdaptiveSampler
        dom = BoxDomain([-1.0, -1.0], [1.0, 1.0])
        pts = AdaptiveSampler(dom, lambda X: X.abs().sum(1), n_points=80).sample()
        assert (pts[:, 0] >= -1.0).all()
        assert (pts[:, 0] <= 1.0).all()

    def test_invalid_refine_ratio_raises(self):
        from pypielm.pde.collocation import BoxDomain, AdaptiveSampler
        dom = BoxDomain([0.0], [1.0])
        with pytest.raises(ValueError):
            AdaptiveSampler(dom, lambda X: X[:, 0], refine_ratio=0.0)
        with pytest.raises(ValueError):
            AdaptiveSampler(dom, lambda X: X[:, 0], refine_ratio=1.5)

    def test_concentrates_near_high_residual(self):
        """Adaptive sampler should concentrate points near x=1 for residual=x."""
        from pypielm.pde.collocation import BoxDomain, AdaptiveSampler
        dom = BoxDomain([0.0], [1.0])
        # residual peaks near x=1
        pts = AdaptiveSampler(
            dom,
            lambda X: X[:, 0],
            n_points=200,
            refine_ratio=0.9,
            n_candidates=5000,
            seed=0,
        ).sample()
        mean_x = pts[:, 0].mean().item()
        assert mean_x > 0.55, f"Expected mean > 0.55, got {mean_x:.3f}"


class TestGridSampler:
    def test_shape_1d(self):
        from pypielm.pde.collocation import BoxDomain, GridSampler
        dom = BoxDomain([0.0], [1.0])
        pts = GridSampler(dom, nx=10).sample()
        assert pts.shape == (10, 1)

    def test_shape_2d(self):
        from pypielm.pde.collocation import BoxDomain, GridSampler
        dom = BoxDomain([0.0, 0.0], [1.0, 1.0])
        pts = GridSampler(dom, nx=8, ny=12).sample()
        assert pts.shape == (8 * 12, 2)

    def test_1d_endpoints(self):
        from pypielm.pde.collocation import BoxDomain, GridSampler
        dom = BoxDomain([0.0], [1.0])
        pts = GridSampler(dom, nx=5).sample()
        assert abs(pts[0, 0].item()) < 1e-12
        assert abs(pts[-1, 0].item() - 1.0) < 1e-12

    def test_2d_within_bounds(self):
        from pypielm.pde.collocation import BoxDomain, GridSampler
        dom = BoxDomain([0.5, 1.0], [1.5, 3.0])
        pts = GridSampler(dom, nx=5, ny=5).sample()
        assert (pts[:, 0] >= 0.5).all() and (pts[:, 0] <= 1.5).all()
        assert (pts[:, 1] >= 1.0).all() and (pts[:, 1] <= 3.0).all()

    def test_dtype_float64(self):
        from pypielm.pde.collocation import BoxDomain, GridSampler
        dom = BoxDomain([0.0, 0.0], [1.0, 1.0])
        pts = GridSampler(dom, nx=4, ny=4).sample()
        assert pts.dtype == torch.float64


# ===========================================================================
# BC / IC constraints
# ===========================================================================

@pytest.fixture()
def fm2d():
    """2D RandomFeatureMap with hidden_dim=32."""
    from pypielm.core.feature_maps import RandomFeatureMap
    return RandomFeatureMap(input_dim=2, hidden_dim=32, seed=0)


@pytest.fixture()
def bc_pts_2d():
    """10 boundary points in 2D."""
    return torch.rand(10, 2, dtype=torch.float64)


class TestDirichletBC:
    def test_shape(self, fm2d, bc_pts_2d):
        from pypielm.pde.constraints import DirichletBC
        bc = DirichletBC(lambda x: torch.zeros(x.shape[0], 1, dtype=x.dtype), bc_pts_2d)
        wls = bc.assemble(fm2d)
        assert wls.H.shape == (10, 32)
        assert wls.y.shape == (10, 1)

    def test_values(self, fm2d, bc_pts_2d):
        """H should equal fm(points)."""
        from pypielm.pde.constraints import DirichletBC
        bc = DirichletBC(lambda x: x[:, 0:1], bc_pts_2d)
        wls = bc.assemble(fm2d)
        expected_H = fm2d(bc_pts_2d)
        assert torch.allclose(wls.H, expected_H)
        assert torch.allclose(wls.y, bc_pts_2d[:, 0:1])

    def test_weight(self, fm2d, bc_pts_2d):
        from pypielm.pde.constraints import DirichletBC
        bc = DirichletBC(lambda x: x[:, 0:1], bc_pts_2d, weight=5.0)
        wls = bc.assemble(fm2d)
        assert wls.weight == pytest.approx(5.0)

    def test_default_weight_1(self, fm2d, bc_pts_2d):
        from pypielm.pde.constraints import DirichletBC
        bc = DirichletBC(lambda x: x[:, 0:1], bc_pts_2d)
        wls = bc.assemble(fm2d)
        assert wls.weight == pytest.approx(1.0)

    def test_scalar_boundary_fn_output(self, fm2d, bc_pts_2d):
        """boundary_fn returning (N,) shape should be squeezed to (N,1)."""
        from pypielm.pde.constraints import DirichletBC
        bc = DirichletBC(lambda x: x[:, 0], bc_pts_2d)  # returns (N,) not (N,1)
        wls = bc.assemble(fm2d)
        assert wls.y.shape == (10, 1)


class TestNeumannBC:
    def test_shape(self, fm2d, bc_pts_2d):
        from pypielm.pde.constraints import NeumannBC
        normal = torch.zeros_like(bc_pts_2d)
        normal[:, 0] = 1.0  # outward normal along x
        bc = NeumannBC(lambda x: torch.zeros(x.shape[0], 1, dtype=x.dtype), normal, bc_pts_2d)
        wls = bc.assemble(fm2d)
        assert wls.H.shape == (10, 32)
        assert wls.y.shape == (10, 1)

    def test_x_normal_equals_d1_x(self, fm2d, bc_pts_2d):
        """For normal = e_0 (x-direction), H_n = fm.d1(pts, 0)."""
        from pypielm.pde.constraints import NeumannBC
        normal = torch.zeros_like(bc_pts_2d)
        normal[:, 0] = 1.0
        bc = NeumannBC(lambda x: torch.zeros(x.shape[0], 1, dtype=x.dtype), normal, bc_pts_2d)
        wls = bc.assemble(fm2d)
        expected = fm2d.d1(bc_pts_2d, 0)
        assert torch.allclose(wls.H, expected, atol=1e-12)

    def test_weight(self, fm2d, bc_pts_2d):
        from pypielm.pde.constraints import NeumannBC
        normal = torch.zeros_like(bc_pts_2d); normal[:, 1] = 1.0
        bc = NeumannBC(lambda x: x[:, 0:1], normal, bc_pts_2d, weight=2.5)
        wls = bc.assemble(fm2d)
        assert wls.weight == pytest.approx(2.5)


class TestInitialCondition:
    def test_shape(self, fm2d):
        from pypielm.pde.constraints import InitialCondition
        pts = torch.rand(15, 2, dtype=torch.float64)
        ic = InitialCondition(lambda x: torch.zeros(x.shape[0], 1, dtype=x.dtype), pts)
        wls = ic.assemble(fm2d)
        assert wls.H.shape == (15, 32)
        assert wls.y.shape == (15, 1)

    def test_H_equals_fm_points(self, fm2d):
        from pypielm.pde.constraints import InitialCondition
        pts = torch.rand(8, 2, dtype=torch.float64)
        ic = InitialCondition(lambda x: x[:, 0:1], pts)
        wls = ic.assemble(fm2d)
        assert torch.allclose(wls.H, fm2d(pts))

    def test_y_from_ic_fn(self, fm2d):
        from pypielm.pde.constraints import InitialCondition
        pts = torch.rand(8, 2, dtype=torch.float64)
        ic_fn = lambda x: torch.sin(x[:, 0:1])
        ic = InitialCondition(ic_fn, pts)
        wls = ic.assemble(fm2d)
        assert torch.allclose(wls.y, torch.sin(pts[:, 0:1]))

    def test_weight(self, fm2d):
        from pypielm.pde.constraints import InitialCondition
        pts = torch.rand(5, 2, dtype=torch.float64)
        ic = InitialCondition(lambda x: x[:, 0:1], pts, weight=3.0)
        wls = ic.assemble(fm2d)
        assert wls.weight == pytest.approx(3.0)


class TestPeriodicBC:
    def test_shape(self, fm2d):
        from pypielm.pde.constraints import PeriodicBC
        n = 12
        pts_l = torch.rand(n, 2, dtype=torch.float64); pts_l[:, 0] = 0.0
        pts_r = pts_l.clone(); pts_r[:, 0] = 1.0
        bc = PeriodicBC(axis=0, points_left=pts_l, points_right=pts_r)
        wls = bc.assemble(fm2d)
        assert wls.H.shape == (n, 32)
        assert wls.y.shape == (n, 1)

    def test_y_is_zeros(self, fm2d):
        from pypielm.pde.constraints import PeriodicBC
        n = 8
        pts_l = torch.rand(n, 2, dtype=torch.float64); pts_l[:, 0] = 0.0
        pts_r = pts_l.clone(); pts_r[:, 0] = 1.0
        bc = PeriodicBC(axis=0, points_left=pts_l, points_right=pts_r)
        wls = bc.assemble(fm2d)
        assert torch.allclose(wls.y, torch.zeros_like(wls.y))

    def test_H_is_difference(self, fm2d):
        from pypielm.pde.constraints import PeriodicBC
        n = 8
        pts_l = torch.rand(n, 2, dtype=torch.float64); pts_l[:, 0] = 0.0
        pts_r = pts_l.clone(); pts_r[:, 0] = 1.0
        bc = PeriodicBC(axis=0, points_left=pts_l, points_right=pts_r)
        wls = bc.assemble(fm2d)
        expected = fm2d(pts_l) - fm2d(pts_r)
        assert torch.allclose(wls.H, expected)

    def test_same_points_gives_zero_H(self, fm2d):
        """If left == right (trivially periodic), H should be zero."""
        from pypielm.pde.constraints import PeriodicBC
        pts = torch.rand(6, 2, dtype=torch.float64)
        bc = PeriodicBC(axis=0, points_left=pts, points_right=pts)
        wls = bc.assemble(fm2d)
        assert torch.allclose(wls.H, torch.zeros_like(wls.H), atol=1e-14)

    def test_weight(self, fm2d):
        from pypielm.pde.constraints import PeriodicBC
        n = 5
        pts_l = torch.rand(n, 2, dtype=torch.float64)
        pts_r = torch.rand(n, 2, dtype=torch.float64)
        bc = PeriodicBC(axis=1, points_left=pts_l, points_right=pts_r, weight=10.0)
        wls = bc.assemble(fm2d)
        assert wls.weight == pytest.approx(10.0)
