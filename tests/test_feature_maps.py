"""Comprehensive tests for pypielm.core.feature_maps."""

from __future__ import annotations

import math

import pytest
import torch

from pypielm.core.feature_maps import AutogradFeatureMap, FourierFeatureMap, RandomFeatureMap

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_X(N: int = 20, d: int = 2, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(N, d, dtype=dtype)


# ===========================================================================
# RandomFeatureMap – construction
# ===========================================================================

class TestRandomFeatureMapConstruction:
    def test_buffers_exist(self):
        fm = RandomFeatureMap(input_dim=3, hidden_dim=50, seed=7)
        assert hasattr(fm, "W")
        assert hasattr(fm, "b")

    def test_weights_frozen(self):
        """W and b must be registered as buffers, not learnable parameters."""
        fm = RandomFeatureMap(input_dim=2, hidden_dim=30, seed=0)
        param_names = {n for n, _ in fm.named_parameters()}
        assert "W" not in param_names
        assert "b" not in param_names

    def test_W_shape(self):
        fm = RandomFeatureMap(input_dim=3, hidden_dim=50)
        assert fm.W.shape == (3, 50)

    def test_b_shape(self):
        fm = RandomFeatureMap(input_dim=3, hidden_dim=50)
        assert fm.b.shape == (50,)

    def test_dtype_float32(self):
        fm = RandomFeatureMap(input_dim=2, hidden_dim=10, dtype=torch.float32)
        assert fm.W.dtype == torch.float32

    def test_dtype_float64(self):
        fm = RandomFeatureMap(input_dim=2, hidden_dim=10, dtype=torch.float64)
        assert fm.W.dtype == torch.float64

    def test_seed_reproducibility(self):
        fm1 = RandomFeatureMap(input_dim=2, hidden_dim=40, seed=99)
        fm2 = RandomFeatureMap(input_dim=2, hidden_dim=40, seed=99)
        assert torch.allclose(fm1.W, fm2.W)

    def test_different_seeds_differ(self):
        fm1 = RandomFeatureMap(input_dim=2, hidden_dim=40, seed=1)
        fm2 = RandomFeatureMap(input_dim=2, hidden_dim=40, seed=2)
        assert not torch.allclose(fm1.W, fm2.W)

    def test_unknown_activation_raises(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            RandomFeatureMap(input_dim=2, hidden_dim=10, activation="swish")


# ===========================================================================
# RandomFeatureMap – forward
# ===========================================================================

class TestRandomFeatureMapForward:
    @pytest.mark.parametrize("activation", ["tanh", "sin", "relu", "sigmoid", "softplus"])
    def test_output_shape(self, activation):
        fm = RandomFeatureMap(input_dim=2, hidden_dim=50, activation=activation)
        X = make_X(20, 2)
        H = fm(X)
        assert H.shape == (20, 50)

    def test_output_dtype_preserved(self):
        fm = RandomFeatureMap(input_dim=2, hidden_dim=30, dtype=torch.float64)
        X = make_X(10, 2, dtype=torch.float64)
        assert fm(X).dtype == torch.float64

    def test_output_dtype_float32(self):
        fm = RandomFeatureMap(input_dim=2, hidden_dim=30, dtype=torch.float32)
        X = make_X(10, 2, dtype=torch.float32)
        assert fm(X).dtype == torch.float32

    def test_tanh_output_range(self):
        fm = RandomFeatureMap(input_dim=2, hidden_dim=200, activation="tanh")
        X = make_X(50, 2)
        H = fm(X)
        assert H.abs().max().item() <= 1.0 + 1e-9

    def test_relu_nonneg(self):
        fm = RandomFeatureMap(input_dim=2, hidden_dim=200, activation="relu")
        X = make_X(50, 2)
        H = fm(X)
        assert (H >= -1e-12).all()

    def test_sigmoid_range(self):
        fm = RandomFeatureMap(input_dim=2, hidden_dim=200, activation="sigmoid")
        X = make_X(50, 2)
        H = fm(X)
        assert H.min().item() >= -1e-9
        assert H.max().item() <= 1.0 + 1e-9

    def test_numpy_input(self):
        import numpy as np
        fm = RandomFeatureMap(input_dim=2, hidden_dim=20)
        X_np = np.random.rand(10, 2)
        H = fm(X_np)
        assert H.shape == (10, 20)

    def test_wrong_input_dim_raises(self):
        fm = RandomFeatureMap(input_dim=3, hidden_dim=20)
        X = make_X(10, 2)
        with pytest.raises(ValueError):
            fm(X)


# ===========================================================================
# RandomFeatureMap – d1 analytic vs autograd
# ===========================================================================

class TestRandomFeatureMapD1:
    @pytest.mark.parametrize("activation", ["tanh", "sin", "relu", "sigmoid", "softplus"])
    def test_d1_shape(self, activation):
        fm = RandomFeatureMap(input_dim=3, hidden_dim=40, activation=activation)
        X = make_X(15, 3)
        for axis in range(3):
            dH = fm.d1(X, axis)
            assert dH.shape == (15, 40)

    @pytest.mark.parametrize("activation", ["tanh", "sin", "sigmoid", "softplus"])
    @pytest.mark.parametrize("axis", [0, 1])
    def test_d1_matches_autograd(self, activation, axis):
        """Analytic d1 must match autograd within atol=1e-5."""
        fm = RandomFeatureMap(input_dim=2, hidden_dim=30, activation=activation, seed=0)
        X = make_X(10, 2)

        analytic = fm.d1(X, axis)

        # Reference via autograd
        X_req = X.clone().requires_grad_(True)
        Z = X_req @ fm.W + fm.b
        import torch.nn.functional as F
        acts = {"tanh": torch.tanh, "sin": torch.sin, "sigmoid": torch.sigmoid,
                "softplus": F.softplus}
        H = acts[activation](Z)
        ref = torch.zeros_like(H)
        for j in range(30):
            (grad,) = torch.autograd.grad(H[:, j].sum(), X_req, retain_graph=True)
            ref[:, j] = grad[:, axis]

        assert torch.allclose(analytic, ref, atol=1e-5), (
            f"d1 mismatch for activation={activation}, axis={axis}\n"
            f"max err={( analytic - ref).abs().max().item()}"
        )


# ===========================================================================
# RandomFeatureMap – d2 analytic vs autograd
# ===========================================================================

class TestRandomFeatureMapD2:
    @pytest.mark.parametrize("activation", ["tanh", "sin", "relu", "sigmoid", "softplus"])
    def test_d2_shape(self, activation):
        fm = RandomFeatureMap(input_dim=2, hidden_dim=20, activation=activation)
        X = make_X(8, 2)
        for axis in range(2):
            d2H = fm.d2(X, axis)
            assert d2H.shape == (8, 20)

    @pytest.mark.parametrize("activation", ["tanh", "sin", "sigmoid", "softplus"])
    @pytest.mark.parametrize("axis", [0, 1])
    def test_d2_matches_autograd(self, activation, axis):
        """Analytic d2 must match autograd within atol=1e-5."""
        fm = RandomFeatureMap(input_dim=2, hidden_dim=20, activation=activation, seed=1)
        X = make_X(8, 2)

        analytic = fm.d2(X, axis)

        import torch.nn.functional as F
        acts = {"tanh": torch.tanh, "sin": torch.sin, "sigmoid": torch.sigmoid,
                "softplus": F.softplus}
        ref = torch.zeros(8, 20, dtype=X.dtype)
        for j in range(20):
            X_req = X.clone().requires_grad_(True)
            Z = X_req @ fm.W + fm.b
            h_j = acts[activation](Z[:, j]).sum()
            (g1,) = torch.autograd.grad(h_j, X_req, create_graph=True)
            g1_sum = g1[:, axis].sum()
            (g2,) = torch.autograd.grad(g1_sum, X_req)
            ref[:, j] = g2[:, axis]

        assert torch.allclose(analytic, ref, atol=1e-5), (
            f"d2 mismatch for activation={activation}, axis={axis}\n"
            f"max err={(analytic - ref).abs().max().item()}"
        )

    def test_relu_d2_zero(self):
        """ReLU second derivative is identically zero (in the analytic sense)."""
        fm = RandomFeatureMap(input_dim=2, hidden_dim=50, activation="relu")
        X = make_X(20, 2)
        d2H = fm.d2(X, 0)
        assert torch.all(d2H == 0.0)


# ===========================================================================
# RandomFeatureMap – laplacian
# ===========================================================================

class TestRandomFeatureMapLaplacian:
    def test_laplacian_equals_sum_of_d2(self):
        fm = RandomFeatureMap(input_dim=3, hidden_dim=30, activation="tanh")
        X = make_X(12, 3)
        lap = fm.laplacian(X)
        manual = sum(fm.d2(X, ax) for ax in range(3))
        assert torch.allclose(lap, manual, atol=1e-12)


# ===========================================================================
# FourierFeatureMap – construction
# ===========================================================================

class TestFourierFeatureMapConstruction:
    def test_buffers_shape(self):
        fm = FourierFeatureMap(input_dim=2, hidden_dim=50)
        assert fm.W.shape == (2, 50)
        assert fm.b.shape == (50,)
        assert fm.omega.shape == (50,)

    def test_weights_frozen(self):
        fm = FourierFeatureMap(input_dim=2, hidden_dim=30)
        param_names = {n for n, _ in fm.named_parameters()}
        assert "W" not in param_names
        assert "omega" not in param_names
        assert "b" not in param_names

    def test_omega_positive(self):
        fm = FourierFeatureMap(input_dim=2, hidden_dim=100, freq_min=1.0, freq_max=50.0)
        assert (fm.omega > 0).all()

    def test_omega_range_uniform(self):
        fm = FourierFeatureMap(
            input_dim=1, hidden_dim=500, freq_init="uniform",
            freq_min=5.0, freq_max=20.0, seed=0
        )
        assert fm.omega.min().item() >= 5.0 - 1e-9
        assert fm.omega.max().item() <= 20.0 + 1e-9

    def test_invalid_freq_raises(self):
        with pytest.raises(ValueError):
            FourierFeatureMap(input_dim=2, hidden_dim=10, freq_min=100.0, freq_max=1.0)

    def test_seed_reproducibility(self):
        fm1 = FourierFeatureMap(input_dim=2, hidden_dim=30, seed=5)
        fm2 = FourierFeatureMap(input_dim=2, hidden_dim=30, seed=5)
        assert torch.allclose(fm1.W, fm2.W)
        assert torch.allclose(fm1.omega, fm2.omega)


# ===========================================================================
# FourierFeatureMap – forward
# ===========================================================================

class TestFourierFeatureMapForward:
    def test_output_shape(self):
        fm = FourierFeatureMap(input_dim=2, hidden_dim=60)
        X = make_X(25, 2)
        assert fm(X).shape == (25, 60)

    def test_output_dtype(self):
        fm = FourierFeatureMap(input_dim=2, hidden_dim=20, dtype=torch.float64)
        X = make_X(10, 2, dtype=torch.float64)
        assert fm(X).dtype == torch.float64

    def test_output_scale(self):
        """φ(x) = √2 cos(·), so |φ| ≤ √2."""
        fm = FourierFeatureMap(input_dim=2, hidden_dim=200)
        X = make_X(100, 2)
        assert fm(X).abs().max().item() <= math.sqrt(2.0) + 1e-9

    def test_numpy_input(self):
        import numpy as np
        fm = FourierFeatureMap(input_dim=2, hidden_dim=20)
        X_np = np.random.rand(10, 2)
        H = fm(X_np)
        assert H.shape == (10, 20)


# ===========================================================================
# FourierFeatureMap – d1 and d2 vs autograd
# ===========================================================================

class TestFourierFeatureMapDerivatives:
    @pytest.mark.parametrize("axis", [0, 1])
    def test_d1_matches_autograd(self, axis):
        fm = FourierFeatureMap(input_dim=2, hidden_dim=20, seed=3)
        X = make_X(8, 2)

        analytic = fm.d1(X, axis)

        ref = torch.zeros(8, 20, dtype=X.dtype)
        for j in range(20):
            X_req = X.clone().requires_grad_(True)
            A = (X_req @ fm.W) * fm.omega + fm.b
            h_j = (math.sqrt(2.0) * torch.cos(A[:, j])).sum()
            (g,) = torch.autograd.grad(h_j, X_req)
            ref[:, j] = g[:, axis]

        assert torch.allclose(analytic, ref, atol=1e-6), (
            f"FourierFeatureMap d1 mismatch axis={axis}; "
            f"max err={(analytic - ref).abs().max().item()}"
        )

    @pytest.mark.parametrize("axis", [0, 1])
    def test_d2_matches_autograd(self, axis):
        fm = FourierFeatureMap(input_dim=2, hidden_dim=20, seed=4)
        X = make_X(8, 2)

        analytic = fm.d2(X, axis)

        ref = torch.zeros(8, 20, dtype=X.dtype)
        for j in range(20):
            X_req = X.clone().requires_grad_(True)
            A = (X_req @ fm.W) * fm.omega + fm.b
            h_j = (math.sqrt(2.0) * torch.cos(A[:, j])).sum()
            (g1,) = torch.autograd.grad(h_j, X_req, create_graph=True)
            g1_sum = g1[:, axis].sum()
            (g2,) = torch.autograd.grad(g1_sum, X_req)
            ref[:, j] = g2[:, axis]

        assert torch.allclose(analytic, ref, atol=1e-6), (
            f"FourierFeatureMap d2 mismatch axis={axis}; "
            f"max err={(analytic - ref).abs().max().item()}"
        )

    def test_d1_shape(self):
        fm = FourierFeatureMap(input_dim=3, hidden_dim=30)
        X = make_X(10, 3)
        for ax in range(3):
            assert fm.d1(X, ax).shape == (10, 30)

    def test_d2_shape(self):
        fm = FourierFeatureMap(input_dim=3, hidden_dim=30)
        X = make_X(10, 3)
        for ax in range(3):
            assert fm.d2(X, ax).shape == (10, 30)

    def test_laplacian_equals_sum_of_d2(self):
        fm = FourierFeatureMap(input_dim=3, hidden_dim=25)
        X = make_X(12, 3)
        lap = fm.laplacian(X)
        manual = sum(fm.d2(X, ax) for ax in range(3))
        assert torch.allclose(lap, manual, atol=1e-12)


# ===========================================================================
# AutogradFeatureMap
# ===========================================================================

class TestAutogradFeatureMap:
    def test_forward_shape(self):
        fm = AutogradFeatureMap(
            input_dim=2, hidden_dim=10, activation_fn=torch.tanh, seed=0
        )
        X = make_X(8, 2)
        assert fm(X).shape == (8, 10)

    def test_forward_matches_manual(self):
        fm = AutogradFeatureMap(
            input_dim=2, hidden_dim=10, activation_fn=torch.tanh, seed=0
        )
        X = make_X(8, 2)
        H_auto = fm(X)
        H_manual = torch.tanh(X @ fm.W + fm.b)
        assert torch.allclose(H_auto, H_manual, atol=1e-12)

    def test_d1_matches_analytic_tanh(self):
        """AutogradFeatureMap d1 should match RandomFeatureMap d1 for tanh."""
        seed = 42
        fm_auto = AutogradFeatureMap(
            input_dim=2, hidden_dim=10, activation_fn=torch.tanh, seed=seed
        )
        fm_ref = RandomFeatureMap(input_dim=2, hidden_dim=10, activation="tanh", seed=seed)
        X = make_X(5, 2)
        d1_auto = fm_auto.d1(X, 0)
        d1_ref = fm_ref.d1(X, 0)
        assert torch.allclose(d1_auto, d1_ref, atol=1e-6)

    def test_d2_matches_analytic_tanh(self):
        seed = 42
        fm_auto = AutogradFeatureMap(
            input_dim=2, hidden_dim=10, activation_fn=torch.tanh, seed=seed
        )
        fm_ref = RandomFeatureMap(input_dim=2, hidden_dim=10, activation="tanh", seed=seed)
        X = make_X(5, 2)
        d2_auto = fm_auto.d2(X, 0)
        d2_ref = fm_ref.d2(X, 0)
        assert torch.allclose(d2_auto, d2_ref, atol=1e-5)

    def test_custom_activation(self):
        """AutogradFeatureMap works with an arbitrary smooth activation."""
        def act(z):
            return torch.sin(z) + 0.1 * torch.cos(2 * z)
        fm = AutogradFeatureMap(input_dim=2, hidden_dim=8, activation_fn=act, seed=0)
        X = make_X(5, 2)
        H = fm(X)
        assert H.shape == (5, 8)
        assert H.dtype == torch.float64

    def test_weights_frozen(self):
        fm = AutogradFeatureMap(
            input_dim=2, hidden_dim=10, activation_fn=torch.tanh
        )
        param_names = {n for n, _ in fm.named_parameters()}
        assert "W" not in param_names


# ===========================================================================
# Cross-check: all feature maps same weights via same seed
# ===========================================================================

class TestFeatureMapReproducibility:
    def test_random_fm_same_X_twice(self):
        fm = RandomFeatureMap(input_dim=2, hidden_dim=50, seed=7)
        X = make_X(20, 2)
        H1 = fm(X)
        H2 = fm(X)
        assert torch.allclose(H1, H2)

    def test_fourier_fm_same_X_twice(self):
        fm = FourierFeatureMap(input_dim=2, hidden_dim=50, seed=7)
        X = make_X(20, 2)
        H1 = fm(X)
        H2 = fm(X)
        assert torch.allclose(H1, H2)

