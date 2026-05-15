"""Comprehensive tests for pypielm.core.solver."""

from __future__ import annotations

import pytest
import torch

from pypielm.core.solver import (
    BayesianSolveResult,
    WeightedLinearSystem,
    bayesian_solve,
    ridge_solve,
    rrqr_solve,
    tikhonov_solve,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def synthetic_system(
    N: int = 100,
    H: int = 30,
    out_dim: int = 1,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (H_mat, y, beta_true) for a well-conditioned synthetic system."""
    gen = torch.Generator().manual_seed(seed)
    beta_true = torch.randn(H, out_dim, dtype=dtype, device=device, generator=gen)
    X_feat = torch.randn(N, H, dtype=dtype, device=device, generator=gen)
    # Ensure reasonable conditioning
    y = X_feat @ beta_true + 1e-6 * torch.randn(N, out_dim, dtype=dtype, device=device, generator=gen)
    return X_feat, y, beta_true


# ===========================================================================
# ridge_solve
# ===========================================================================

class TestRidgeSolve:
    def test_output_shape_1d_y(self):
        H_mat, y, _ = synthetic_system(50, 20, 1)
        beta = ridge_solve(H_mat, y.squeeze(), lam=1e-6)
        assert beta.shape == (20, 1)

    def test_output_shape_2d_y(self):
        H_mat, y, _ = synthetic_system(50, 20, 3)
        beta = ridge_solve(H_mat, y, lam=1e-6)
        assert beta.shape == (20, 3)

    def test_small_lambda_recovers_solution(self):
        """With tiny λ and well-conditioned H, ridge should nearly minimise residual."""
        H_mat, y, beta_true = synthetic_system(200, 30, 1, seed=42)
        beta = ridge_solve(H_mat, y, lam=1e-12)
        y_pred = H_mat @ beta
        residual = (y - y_pred).norm() / y.norm()
        assert residual.item() < 1e-3, f"Residual too large: {residual.item()}"

    def test_larger_lambda_increases_residual(self):
        H_mat, y, _ = synthetic_system(100, 20, 1, seed=7)
        beta_small = ridge_solve(H_mat, y, lam=1e-10)
        beta_large = ridge_solve(H_mat, y, lam=1.0)
        res_small = ((y - H_mat @ beta_small) ** 2).mean()
        res_large = ((y - H_mat @ beta_large) ** 2).mean()
        assert res_small <= res_large + 1e-10

    def test_dtype_preserved_float64(self):
        H_mat, y, _ = synthetic_system(50, 20, 1, dtype=torch.float64)
        beta = ridge_solve(H_mat, y, lam=1e-8)
        assert beta.dtype == torch.float64

    def test_symmetry_of_normal_equations(self):
        """H^T H + λI is symmetric positive definite."""
        H_mat, y, _ = synthetic_system(80, 25, 1)
        lam = 1e-6
        A = H_mat.T @ H_mat + lam * torch.eye(25, dtype=torch.float64)
        # All eigenvalues should be positive
        eigvals = torch.linalg.eigvalsh(A)
        assert (eigvals > 0).all()

    def test_zero_lambda_same_as_lstsq(self):
        """ridge with λ=0 should give the same solution as least-squares."""
        H_mat, y, _ = synthetic_system(100, 20, 1, seed=11)
        beta_ridge = ridge_solve(H_mat, y, lam=0.0)
        beta_lstsq = torch.linalg.lstsq(H_mat, y, driver="gelsd").solution
        assert torch.allclose(beta_ridge, beta_lstsq, atol=1e-8)

    def test_multi_output(self):
        H_mat, y, _ = synthetic_system(80, 25, 4, seed=3)
        beta = ridge_solve(H_mat, y, lam=1e-8)
        assert beta.shape == (25, 4)
        residual = (y - H_mat @ beta).norm() / y.norm()
        assert residual.item() < 0.1


# ===========================================================================
# rrqr_solve
# ===========================================================================

class TestRrqrSolve:
    def test_output_shape_1d_y(self):
        H_mat, y, _ = synthetic_system(50, 20, 1)
        beta = rrqr_solve(H_mat, y.squeeze())
        assert beta.shape == (20, 1)

    def test_output_shape_2d_y(self):
        H_mat, y, _ = synthetic_system(50, 20, 3)
        beta = rrqr_solve(H_mat, y)
        assert beta.shape == (20, 3)

    def test_low_residual(self):
        H_mat, y, _ = synthetic_system(200, 30, 1, seed=55)
        beta = rrqr_solve(H_mat, y)
        residual = (y - H_mat @ beta).norm() / y.norm()
        assert residual.item() < 1e-3

    def test_dtype_float64(self):
        H_mat, y, _ = synthetic_system(50, 20, 1, dtype=torch.float64)
        beta = rrqr_solve(H_mat, y)
        assert beta.dtype == torch.float64

    def test_rank_deficient_system(self):
        """rrqr_solve should handle rank-deficient feature matrices."""
        # Create rank-deficient H (last column is duplicate of first)
        H_mat, y, _ = synthetic_system(50, 20, 1, seed=9)
        H_mat[:, -1] = H_mat[:, 0]  # make rank-deficient
        beta = rrqr_solve(H_mat, y)
        assert beta.shape == (20, 1)

    def test_consistent_with_ridge_well_conditioned(self):
        """For well-conditioned system, rrqr and ridge should give similar predictions."""
        H_mat, y, _ = synthetic_system(200, 30, 1, seed=13)
        beta_ridge = ridge_solve(H_mat, y, lam=1e-12)
        beta_rrqr = rrqr_solve(H_mat, y)
        y_pred_ridge = H_mat @ beta_ridge
        y_pred_rrqr = H_mat @ beta_rrqr
        assert torch.allclose(y_pred_ridge, y_pred_rrqr, atol=1e-4)


# ===========================================================================
# bayesian_solve
# ===========================================================================

class TestBayesianSolve:
    def test_returns_namedtuple(self):
        H_mat, y, _ = synthetic_system(50, 20, 1)
        block = WeightedLinearSystem(H=H_mat, y=y.squeeze(), weight=1.0)
        result = bayesian_solve([block], prior_precision=1e-4)
        assert isinstance(result, BayesianSolveResult)

    def test_result_fields_exist(self):
        H_mat, y, _ = synthetic_system(50, 20, 1)
        block = WeightedLinearSystem(H=H_mat, y=y.squeeze(), weight=1.0)
        result = bayesian_solve([block], prior_precision=1e-4)
        assert hasattr(result, "beta_mean")
        assert hasattr(result, "beta_cov")

    def test_beta_mean_shape(self):
        H_mat, y, _ = synthetic_system(50, 20, 1)
        block = WeightedLinearSystem(H=H_mat, y=y.squeeze(), weight=1.0)
        result = bayesian_solve([block], prior_precision=1e-4)
        assert result.beta_mean.shape == (20, 1)

    def test_precision_matrix_shape(self):
        H_mat, y, _ = synthetic_system(50, 20, 1)
        block = WeightedLinearSystem(H=H_mat, y=y.squeeze(), weight=1.0)
        result = bayesian_solve([block], prior_precision=1e-4)
        assert result.beta_cov.shape == (20, 20)

    def test_precision_matrix_spd(self):
        """The posterior precision matrix must be symmetric positive definite."""
        H_mat, y, _ = synthetic_system(80, 25, 1, seed=7)
        block = WeightedLinearSystem(H=H_mat, y=y.squeeze(), weight=1.0)
        result = bayesian_solve([block], prior_precision=1e-4)
        prec = result.beta_cov
        # Symmetry
        assert torch.allclose(prec, prec.T, atol=1e-10)
        # Positive definiteness
        eigvals = torch.linalg.eigvalsh(prec)
        assert (eigvals > 0).all()

    def test_posterior_mean_matches_ridge_single_block(self):
        """bayesian_solve with weight=1 and prior_precision=lam recovers ridge."""
        H_mat, y, _ = synthetic_system(100, 25, 1, seed=21)
        lam = 1e-4
        block = WeightedLinearSystem(H=H_mat, y=y.squeeze(), weight=1.0)
        bayes_result = bayesian_solve([block], prior_precision=lam)
        ridge_result = ridge_solve(H_mat, y, lam=lam)
        assert torch.allclose(bayes_result.beta_mean, ridge_result, atol=1e-8), (
            f"Bayesian posterior mean does not match ridge; "
            f"max err={(bayes_result.beta_mean - ridge_result).abs().max().item()}"
        )

    def test_multi_block(self):
        """Multiple blocks should accumulate correctly."""
        H1, y1, _ = synthetic_system(40, 20, 1, seed=0)
        H2, y2, _ = synthetic_system(30, 20, 1, seed=1)
        blocks = [
            WeightedLinearSystem(H=H1, y=y1.squeeze(), weight=1.0),
            WeightedLinearSystem(H=H2, y=y2.squeeze(), weight=2.0),
        ]
        result = bayesian_solve(blocks, prior_precision=1e-4)
        assert result.beta_mean.shape == (20, 1)
        assert result.beta_cov.shape == (20, 20)

    def test_empty_blocks_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            bayesian_solve([], prior_precision=1e-4)

    def test_dtype_float64(self):
        H_mat, y, _ = synthetic_system(50, 20, 1, dtype=torch.float64)
        block = WeightedLinearSystem(H=H_mat, y=y.squeeze(), weight=1.0)
        result = bayesian_solve([block], prior_precision=1e-4)
        assert result.beta_mean.dtype == torch.float64
        assert result.beta_cov.dtype == torch.float64

    def test_low_residual(self):
        H_mat, y, _ = synthetic_system(200, 30, 1, seed=99)
        block = WeightedLinearSystem(H=H_mat, y=y.squeeze(), weight=1.0)
        result = bayesian_solve([block], prior_precision=1e-8)
        y_pred = H_mat @ result.beta_mean
        residual = (y - y_pred).norm() / y.norm()
        assert residual.item() < 1e-2


# ===========================================================================
# tikhonov_solve
# ===========================================================================

class TestTikhonovSolve:
    def test_output_shape(self):
        H_mat, y, _ = synthetic_system(80, 25, 1)
        L = torch.eye(25, dtype=torch.float64)
        beta = tikhonov_solve(H_mat, y, L, lam=1e-6)
        assert beta.shape == (25, 1)

    def test_identity_L_equals_ridge(self):
        """tikhonov_solve with L=I should be identical to ridge_solve."""
        H_mat, y, _ = synthetic_system(100, 25, 1, seed=5)
        lam = 1e-6
        L = torch.eye(25, dtype=torch.float64)
        beta_tik = tikhonov_solve(H_mat, y, L, lam=lam)
        beta_ridge = ridge_solve(H_mat, y, lam=lam)
        assert torch.allclose(beta_tik, beta_ridge, atol=1e-10)

    def test_non_identity_L(self):
        """Non-identity L should produce a different solution than plain ridge."""
        H_mat, y, _ = synthetic_system(100, 25, 1, seed=6)
        lam = 1e-2
        L_eye = torch.eye(25, dtype=torch.float64)
        # Scale only first 5 components strongly
        L_custom = torch.eye(25, dtype=torch.float64)
        L_custom[:5, :5] *= 10.0
        beta_ridge = tikhonov_solve(H_mat, y, L_eye, lam=lam)
        beta_tik = tikhonov_solve(H_mat, y, L_custom, lam=lam)
        assert not torch.allclose(beta_ridge, beta_tik, atol=1e-6)

    def test_low_residual(self):
        H_mat, y, _ = synthetic_system(200, 30, 1, seed=15)
        L = torch.eye(30, dtype=torch.float64)
        beta = tikhonov_solve(H_mat, y, L, lam=1e-12)
        residual = (y - H_mat @ beta).norm() / y.norm()
        assert residual.item() < 1e-3

    def test_dtype_float64(self):
        H_mat, y, _ = synthetic_system(50, 20, 1, dtype=torch.float64)
        L = torch.eye(20, dtype=torch.float64)
        beta = tikhonov_solve(H_mat, y, L, lam=1e-8)
        assert beta.dtype == torch.float64

    def test_multi_output(self):
        H_mat, y, _ = synthetic_system(80, 25, 3, seed=8)
        L = torch.eye(25, dtype=torch.float64)
        beta = tikhonov_solve(H_mat, y, L, lam=1e-8)
        assert beta.shape == (25, 3)


# ===========================================================================
# WeightedLinearSystem
# ===========================================================================

class TestWeightedLinearSystem:
    def test_namedtuple_access(self):
        H = torch.eye(5, dtype=torch.float64)
        y = torch.zeros(5, dtype=torch.float64)
        blk = WeightedLinearSystem(H=H, y=y, weight=2.5)
        assert blk.H is H
        assert blk.y is y
        assert blk.weight == 2.5

    def test_default_weight(self):
        H = torch.eye(5, dtype=torch.float64)
        y = torch.zeros(5, dtype=torch.float64)
        blk = WeightedLinearSystem(H=H, y=y)
        assert blk.weight == 1.0

