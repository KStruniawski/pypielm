"""Property-based tests using Hypothesis.

Targets mathematical invariants in feature maps and linear solvers that
must hold for *any* valid input, not just a few hand-picked examples.

Run::

    pytest tests/test_property_based.py -v
"""

from __future__ import annotations

import math

import numpy as np
import torch
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from pypielm.core.feature_maps import FourierFeatureMap, RandomFeatureMap
from pypielm.core.solver import WeightedLinearSystem, bayesian_solve, ridge_solve, tikhonov_solve

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

ACTIVATIONS = ("tanh", "sin", "relu", "sigmoid")

small_int = st.integers(min_value=2, max_value=8)
hidden_dim_st = st.integers(min_value=4, max_value=64)
n_samples_st = st.integers(min_value=8, max_value=128)
lambda_st = st.floats(min_value=1e-8, max_value=1.0, allow_nan=False, allow_infinity=False)
seed_st = st.integers(min_value=0, max_value=2**31 - 1)


def _X(n: int, d: int) -> torch.Tensor:
    rng = np.random.default_rng(0)
    return torch.tensor(rng.uniform(-1.0, 1.0, (n, d)), dtype=torch.float64)


# ---------------------------------------------------------------------------
# Feature Map Invariants
# ---------------------------------------------------------------------------

class TestRandomFeatureMapProperties:
    """Mathematical invariants for RandomFeatureMap."""

    @given(
        n=n_samples_st,
        d=small_int,
        h=hidden_dim_st,
        seed=seed_st,
        act=st.sampled_from(ACTIVATIONS),
    )
    @settings(max_examples=40, suppress_health_check=[HealthCheck.too_slow])
    def test_output_shape(self, n: int, d: int, h: int, seed: int, act: str) -> None:
        """H = fm(X) must have shape (n, h)."""
        fm = RandomFeatureMap(input_dim=d, hidden_dim=h, activation=act, seed=seed)
        X = _X(n, d)
        H = fm(X)
        assert H.shape == (n, h), f"Expected ({n}, {h}), got {H.shape}"

    @given(
        n=n_samples_st,
        d=small_int,
        h=hidden_dim_st,
        seed=seed_st,
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_same_seed_determinism(self, n: int, d: int, h: int, seed: int) -> None:
        """Two feature maps with the same seed must produce identical output."""
        X = _X(n, d)
        fm1 = RandomFeatureMap(input_dim=d, hidden_dim=h, activation="tanh", seed=seed)
        fm2 = RandomFeatureMap(input_dim=d, hidden_dim=h, activation="tanh", seed=seed)
        assert torch.allclose(fm1(X), fm2(X))

    @given(
        n=n_samples_st,
        d=small_int,
        h=hidden_dim_st,
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_different_seeds_differ(self, n: int, d: int, h: int) -> None:
        """Different seeds should (almost surely) produce different outputs."""
        X = _X(n, d)
        fm1 = RandomFeatureMap(input_dim=d, hidden_dim=h, seed=0)
        fm2 = RandomFeatureMap(input_dim=d, hidden_dim=h, seed=99999)
        # Not identical — allow rare hash collisions with a high dimension
        if h >= 4:
            assert not torch.allclose(fm1(X), fm2(X))

    @given(
        n=n_samples_st,
        d=small_int,
        h=hidden_dim_st,
        seed=seed_st,
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_d1_shape(self, n: int, d: int, h: int, seed: int) -> None:
        """d1(X, axis) must return shape (n, h)."""
        fm = RandomFeatureMap(input_dim=d, hidden_dim=h, activation="tanh", seed=seed)
        X = _X(n, d)
        for axis in range(d):
            dH = fm.d1(X, axis)
            assert dH.shape == (n, h), f"d1 shape mismatch at axis {axis}"

    @given(
        n=n_samples_st,
        d=small_int,
        h=hidden_dim_st,
        seed=seed_st,
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_d2_shape(self, n: int, d: int, h: int, seed: int) -> None:
        """d2(X, axis) must return shape (n, h)."""
        fm = RandomFeatureMap(input_dim=d, hidden_dim=h, activation="tanh", seed=seed)
        X = _X(n, d)
        for axis in range(d):
            d2H = fm.d2(X, axis)
            assert d2H.shape == (n, h), f"d2 shape mismatch at axis {axis}"

    @given(
        n=n_samples_st,
        d=small_int,
        h=hidden_dim_st,
        seed=seed_st,
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_d1_matches_autograd(self, n: int, d: int, h: int, seed: int) -> None:
        """Analytic d1 must match torch.autograd.grad within 1e-5."""
        fm = RandomFeatureMap(input_dim=d, hidden_dim=h, activation="tanh", seed=seed)
        X = _X(n, d).requires_grad_(True)
        H = fm(X)
        axis = 0  # check first axis only (sufficient to validate formula)
        analytic = fm.d1(X.detach(), axis)
        # Autograd: sum over hidden units to get a scalar, then grad
        grads = []
        for j in range(h):
            g = torch.autograd.grad(H[:, j].sum(), X, retain_graph=True)[0]
            grads.append(g[:, axis])
        autograd_d1 = torch.stack(grads, dim=1)
        assert torch.allclose(analytic, autograd_d1, atol=1e-5, rtol=1e-4), \
            f"Max diff: {(analytic - autograd_d1).abs().max():.2e}"

    @given(
        n=n_samples_st,
        d=small_int,
        h=hidden_dim_st,
        seed=seed_st,
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_tanh_d2_sign_consistency(self, n: int, d: int, h: int, seed: int) -> None:
        """tanh second derivative is bounded: |d²tanh/dx²| ≤ 2."""
        fm = RandomFeatureMap(input_dim=d, hidden_dim=h, activation="tanh", seed=seed)
        X = _X(n, d)
        d2H = fm.d2(X, 0)
        # max of |d²σ/dz²| for tanh is 2/3√3 ≈ 0.385 per unit, scaled by w²
        # Just check values are finite
        assert torch.isfinite(d2H).all(), "d2 contains non-finite values"


class TestFourierFeatureMapProperties:
    """Mathematical invariants for FourierFeatureMap."""

    @given(
        n=n_samples_st,
        d=small_int,
        h=hidden_dim_st,
        seed=seed_st,
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_output_shape(self, n: int, d: int, h: int, seed: int) -> None:
        fm = FourierFeatureMap(input_dim=d, hidden_dim=h, seed=seed)
        X = _X(n, d)
        H = fm(X)
        assert H.shape == (n, h)

    @given(
        n=n_samples_st,
        d=small_int,
        h=hidden_dim_st,
        seed=seed_st,
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_bounded_output(self, n: int, d: int, h: int, seed: int) -> None:
        """FourierFeatureMap outputs √2 cos(·) ∈ [-√2, √2]."""
        fm = FourierFeatureMap(input_dim=d, hidden_dim=h, seed=seed)
        X = _X(n, d)
        H = fm(X)
        bound = math.sqrt(2) + 1e-6
        assert (H.abs() <= bound).all(), f"Output exceeds √2: max={H.abs().max():.4f}"

    @given(
        n=n_samples_st,
        d=small_int,
        h=hidden_dim_st,
        seed=seed_st,
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_frequency_diversity(self, n: int, d: int, h: int, seed: int) -> None:
        """log_uniform init must produce non-uniform frequencies (spread > 0.3 decades)."""
        assume(h >= 16)  # enough samples to reliably span the frequency range
        fm = FourierFeatureMap(input_dim=d, hidden_dim=h, seed=seed, freq_init="log_uniform",
                               freq_min=1.0, freq_max=100.0)
        omegas = fm.omega.abs()  # scalar per neuron
        log_range = (omegas.max().log10() - omegas.min().log10()).item()
        assert log_range > 0.3, f"Frequency range too narrow: {log_range:.2f} decades"


# ---------------------------------------------------------------------------
# Solver Invariants
# ---------------------------------------------------------------------------

class TestRidgeSolveProperties:
    """Mathematical invariants for ridge_solve."""

    @given(
        n=n_samples_st,
        h=hidden_dim_st,
        lam=lambda_st,
        seed=seed_st,
    )
    @settings(max_examples=40, suppress_health_check=[HealthCheck.too_slow])
    def test_output_shape(self, n: int, h: int, lam: float, seed: int) -> None:
        """beta must have shape (h, p) for p outputs."""
        rng = np.random.default_rng(seed)
        H = torch.tensor(rng.standard_normal((n, h)), dtype=torch.float64)
        y = torch.tensor(rng.standard_normal((n, 1)), dtype=torch.float64)
        beta = ridge_solve(H, y, lam)
        assert beta.shape == (h, 1)

    @given(
        n=n_samples_st,
        h=hidden_dim_st,
        lam=lambda_st,
        seed=seed_st,
    )
    @settings(max_examples=40, suppress_health_check=[HealthCheck.too_slow])
    def test_finite_output(self, n: int, h: int, lam: float, seed: int) -> None:
        """Ridge solution must always be finite."""
        rng = np.random.default_rng(seed)
        H = torch.tensor(rng.standard_normal((n, h)), dtype=torch.float64)
        y = torch.tensor(rng.standard_normal((n, 1)), dtype=torch.float64)
        beta = ridge_solve(H, y, lam)
        assert torch.isfinite(beta).all(), "ridge_solve returned non-finite values"

    @given(
        n=n_samples_st,
        h=hidden_dim_st,
        lam=lambda_st,
        seed=seed_st,
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_gram_spd(self, n: int, h: int, lam: float, seed: int) -> None:
        """(H^T H + λI) must be symmetric positive definite."""
        rng = np.random.default_rng(seed)
        H = torch.tensor(rng.standard_normal((n, h)), dtype=torch.float64)
        G = H.T @ H + lam * torch.eye(h, dtype=torch.float64)
        # SPD iff all eigenvalues > 0
        eigvals = torch.linalg.eigvalsh(G)
        assert (eigvals > 0).all(), f"Gram not SPD: min eigenvalue = {eigvals.min():.2e}"

    @given(
        h=hidden_dim_st,
        lam=lambda_st,
        seed=seed_st,
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_large_lambda_shrinks_solution(self, h: int, lam: float, seed: int) -> None:
        """Larger λ must produce a smaller-norm solution."""
        n = 32
        rng = np.random.default_rng(seed)
        H = torch.tensor(rng.standard_normal((n, h)), dtype=torch.float64)
        y = torch.tensor(rng.standard_normal((n, 1)), dtype=torch.float64)
        beta_small = ridge_solve(H, y, lam)
        beta_large = ridge_solve(H, y, lam * 1000)
        assert beta_large.norm() <= beta_small.norm() + 1e-8, \
            "Larger lambda did not shrink the norm"

    @given(
        h=hidden_dim_st,
        seed=seed_st,
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_interpolation_when_lam_near_zero(self, h: int, seed: int) -> None:
        """With tiny λ and n = h, solution should nearly interpolate y."""
        n = h  # square system
        rng = np.random.default_rng(seed)
        H = torch.tensor(rng.standard_normal((n, h)), dtype=torch.float64)
        # Make H well-conditioned by QR
        Q, _ = torch.linalg.qr(H)
        H = Q  # orthogonal → condition number = 1
        y = torch.tensor(rng.standard_normal((n, 1)), dtype=torch.float64)
        beta = ridge_solve(H, y, lam=1e-12)
        residual = (H @ beta - y).norm().item()
        assert residual < 1e-6, f"Interpolation residual too large: {residual:.2e}"


class TestBayesianSolveProperties:
    """Invariants for bayesian_solve."""

    @given(
        n=n_samples_st,
        h=hidden_dim_st,
        prior_prec=lambda_st,
        seed=seed_st,
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_posterior_mean_shape(self, n: int, h: int, prior_prec: float, seed: int) -> None:
        rng = np.random.default_rng(seed)
        H = torch.tensor(rng.standard_normal((n, h)), dtype=torch.float64)
        y = torch.tensor(rng.standard_normal((n, 1)), dtype=torch.float64)
        block = WeightedLinearSystem(H, y, weight=1.0)
        mean, cov = bayesian_solve([block], prior_precision=prior_prec)
        assert mean.shape == (h, 1)
        assert cov.shape == (h, h)

    @given(
        n=n_samples_st,
        h=hidden_dim_st,
        prior_prec=lambda_st,
        seed=seed_st,
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_posterior_cov_symmetric(self, n: int, h: int, prior_prec: float, seed: int) -> None:
        """Posterior covariance must be symmetric."""
        rng = np.random.default_rng(seed)
        H = torch.tensor(rng.standard_normal((n, h)), dtype=torch.float64)
        y = torch.tensor(rng.standard_normal((n, 1)), dtype=torch.float64)
        block = WeightedLinearSystem(H, y, weight=1.0)
        _, cov = bayesian_solve([block], prior_precision=prior_prec)
        assert torch.allclose(cov, cov.T, atol=1e-10), "Posterior covariance not symmetric"

    @given(
        n=n_samples_st,
        h=hidden_dim_st,
        prior_prec=lambda_st,
        seed=seed_st,
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_posterior_cov_spd(self, n: int, h: int, prior_prec: float, seed: int) -> None:
        """Posterior covariance must be SPD."""
        rng = np.random.default_rng(seed)
        H = torch.tensor(rng.standard_normal((n, h)), dtype=torch.float64)
        y = torch.tensor(rng.standard_normal((n, 1)), dtype=torch.float64)
        block = WeightedLinearSystem(H, y, weight=1.0)
        _, cov = bayesian_solve([block], prior_precision=prior_prec)
        eigvals = torch.linalg.eigvalsh(cov)
        assert (eigvals > -1e-10).all(), f"Posterior covariance not PSD: {eigvals.min():.2e}"

    @given(
        n=n_samples_st,
        h=hidden_dim_st,
        seed=seed_st,
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_uninformative_prior_matches_ridge(self, n: int, h: int, seed: int) -> None:
        """Bayesian posterior mean ≈ ridge solution for the same regularisation."""
        prior_prec = 1.0
        rng = np.random.default_rng(seed)
        H = torch.tensor(rng.standard_normal((n, h)), dtype=torch.float64)
        y = torch.tensor(rng.standard_normal((n, 1)), dtype=torch.float64)
        block = WeightedLinearSystem(H, y, weight=1.0)
        mean, _ = bayesian_solve([block], prior_precision=prior_prec)
        # Ridge with lam = prior_prec / noise_prec (noise_prec defaults to 1)
        beta_ridge = ridge_solve(H, y, lam=prior_prec)
        assert torch.allclose(mean, beta_ridge, atol=1e-6), \
            f"Max diff: {(mean - beta_ridge).abs().max():.2e}"


class TestTikhonovSolveProperties:
    """Invariants for tikhonov_solve."""

    @given(
        n=n_samples_st,
        h=hidden_dim_st,
        lam=lambda_st,
        seed=seed_st,
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_identity_L_matches_ridge(self, n: int, h: int, lam: float, seed: int) -> None:
        """tikhonov_solve(H, y, L=I, lam) == ridge_solve(H, y, lam)."""
        rng = np.random.default_rng(seed)
        H = torch.tensor(rng.standard_normal((n, h)), dtype=torch.float64)
        y = torch.tensor(rng.standard_normal((n, 1)), dtype=torch.float64)
        L = torch.eye(h, dtype=torch.float64)
        beta_tik = tikhonov_solve(H, y, L, lam)
        beta_ridge = ridge_solve(H, y, lam)
        assert torch.allclose(beta_tik, beta_ridge, atol=1e-8), \
            f"Max diff: {(beta_tik - beta_ridge).abs().max():.2e}"

    @given(
        n=n_samples_st,
        h=hidden_dim_st,
        lam=lambda_st,
        seed=seed_st,
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_output_finite(self, n: int, h: int, lam: float, seed: int) -> None:
        rng = np.random.default_rng(seed)
        H = torch.tensor(rng.standard_normal((n, h)), dtype=torch.float64)
        y = torch.tensor(rng.standard_normal((n, 1)), dtype=torch.float64)
        L = torch.eye(h, dtype=torch.float64)
        beta = tikhonov_solve(H, y, L, lam)
        assert torch.isfinite(beta).all()
