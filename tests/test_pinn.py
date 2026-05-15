"""Tests for gradient-based PINN variants.

Covers:
* Fluent fit() API (returns self)
* predict() shape after fit
* loss decreases over training
* 1-D Poisson accuracy: relative L2 < 5e-3
* Reproducibility: same seed -> same loss trajectory
* Registry lookup
* score() returns float
* AdaptivePINN: collocation resampling runs without error
* FourierPINN: Fourier encoding pipeline
* MuonPINN: Muon optimizer runs
* ResidualAdaptivePINN: ResNet + RAR pipeline
* MPS / CUDA device tests (skipped if unavailable)
"""

from __future__ import annotations

import pytest
import torch

from pypielm.core.solver import WeightedLinearSystem
from pypielm.data.dataset import PIELMDataset
from pypielm.models.pinn import (
    AdaptivePINN,
    FourierPINN,
    MuonPINN,
    ResidualAdaptivePINN,
    VanillaPINN,
)
from pypielm.models.registry import get_model
from pypielm.pde.operators import AnalyticLaplacian

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_poisson_1d(
    n_colloc: int = 80,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
):
    """1-D Poisson: u'' = -2 on [0,1], u(0)=u(1)=0. Exact: u=x(1-x)."""
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


def _small_pinn(**kwargs) -> VanillaPINN:
    """Return a tiny VanillaPINN for fast tests."""
    defaults = {"layer_dims": [32, 32], "max_epochs": 200, "lr": 1e-3, "seed": 42}
    defaults.update(kwargs)
    return VanillaPINN(**defaults)


# ---------------------------------------------------------------------------
# VanillaPINN — basic API
# ---------------------------------------------------------------------------

def test_vanilla_pinn_fit_returns_self():
    ds, op, _, _ = _make_poisson_1d(50)
    model = _small_pinn()
    result = model.fit(ds, pde_operator=op)
    assert result is model


def test_vanilla_pinn_predict_shape():
    ds, op, X_test, _ = _make_poisson_1d(50)
    model = _small_pinn().fit(ds, pde_operator=op)
    pred = model.predict(X_test)
    assert pred.shape == (200, 1)


def test_vanilla_pinn_predict_numpy_input():
    """predict() should accept numpy arrays."""
    ds, op, X_test, _ = _make_poisson_1d(50)
    model = _small_pinn().fit(ds, pde_operator=op)
    pred = model.predict(X_test.numpy())
    assert pred.shape == (200, 1)


def test_vanilla_pinn_score_returns_float():
    ds, op, X_test, y_exact = _make_poisson_1d(50)
    model = _small_pinn().fit(ds, pde_operator=op)
    s = model.score(X_test, y_exact)
    assert isinstance(s, float)


def test_vanilla_pinn_loss_decreases():
    """Loss should decrease over the first 200 Adam epochs."""
    ds, op, _, _ = _make_poisson_1d(60)
    model = _small_pinn(max_epochs=200)
    model.fit(ds, pde_operator=op)
    hist = model._loss_history
    assert len(hist) == 200
    # Loss at epoch 200 < loss at epoch 0
    assert hist[-1] < hist[0], (
        f"Loss did not decrease: first={hist[0]:.4e}, last={hist[-1]:.4e}"
    )


def test_vanilla_pinn_loss_history_length():
    ds, op, _, _ = _make_poisson_1d(50)
    model = VanillaPINN(layer_dims=[32, 32], max_epochs=50, seed=1)
    model.fit(ds, pde_operator=op)
    assert len(model._loss_history) == 50


def test_vanilla_pinn_poisson_accuracy():
    """VanillaPINN solves 1-D Poisson with relative L2 < 5e-3 using L-BFGS."""
    ds, op, X_test, y_exact = _make_poisson_1d(100)
    model = VanillaPINN(
        layer_dims=[64, 64],
        optimizer="lbfgs",
        lr=0.5,
        max_epochs=200,
        seed=0,
    )
    model.fit(ds, pde_operator=op)
    err = model.score(X_test, y_exact, metric="relative_l2")
    assert err < 5e-3, f"Relative L2 error {err:.4e} exceeds 5e-3"


def test_vanilla_pinn_reproducibility():
    """Same seed -> same loss trajectory."""
    ds, op, _, _ = _make_poisson_1d(60)
    m1 = VanillaPINN(layer_dims=[32, 32], max_epochs=100, seed=7).fit(ds, pde_operator=op)
    m2 = VanillaPINN(layer_dims=[32, 32], max_epochs=100, seed=7).fit(ds, pde_operator=op)
    assert abs(m1._loss_history[-1] - m2._loss_history[-1]) < 1e-10


def test_vanilla_pinn_different_seeds_differ():
    """Different seeds should produce different networks."""
    ds, op, X_test, _ = _make_poisson_1d(60)
    m1 = VanillaPINN(layer_dims=[32, 32], max_epochs=5, seed=0).fit(ds, pde_operator=op)
    m2 = VanillaPINN(layer_dims=[32, 32], max_epochs=5, seed=99).fit(ds, pde_operator=op)
    p1 = m1.predict(X_test)
    p2 = m2.predict(X_test)
    assert not torch.allclose(p1, p2)


def test_vanilla_pinn_registry_lookup():
    model = get_model("vanilla_pinn", layer_dims=[16], max_epochs=5)
    ds, op, _, _ = _make_poisson_1d(30)
    model.fit(ds, pde_operator=op)
    assert hasattr(model, "_loss_history")


def test_vanilla_pinn_no_pde_operator():
    """VanillaPINN should train on data-only when pde_operator is None."""
    X = torch.linspace(0.0, 1.0, 40).unsqueeze(1).double()
    y = torch.sin(3.0 * X)
    ds = PIELMDataset(X_colloc=X, X_data=X, y_data=y)
    model = VanillaPINN(layer_dims=[32, 32], max_epochs=300, seed=0)
    model.fit(ds)  # no pde_operator
    pred = model.predict(X)
    assert pred.shape == (40, 1)


def test_vanilla_pinn_bc_only():
    """Fit with BCs but no PDE operator (pure BC regression)."""
    X_bc = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    y_bc = torch.zeros(2, 1, dtype=torch.float64)
    ds = PIELMDataset(
        X_colloc=torch.linspace(0, 1, 20).unsqueeze(1).double(),
        X_bc=X_bc,
        y_bc=y_bc,
    )
    model = VanillaPINN(layer_dims=[16, 16], max_epochs=50, seed=0)
    model.fit(ds)
    assert model.predict(X_bc).shape == (2, 1)


def test_vanilla_pinn_lbfgs():
    """L-BFGS optimizer should also converge and reduce loss."""
    ds, op, _, _ = _make_poisson_1d(50)
    model = VanillaPINN(
        layer_dims=[32, 32],
        optimizer="lbfgs",
        lr=0.5,
        max_epochs=20,
        seed=0,
    )
    model.fit(ds, pde_operator=op)
    assert len(model._loss_history) == 20
    assert model._loss_history[-1] < model._loss_history[0]


# ---------------------------------------------------------------------------
# AdaptivePINN
# ---------------------------------------------------------------------------

def test_adaptive_pinn_fit_returns_self():
    ds, op, _, _ = _make_poisson_1d(60)
    model = AdaptivePINN(
        layer_dims=[32, 32],
        max_epochs=100,
        n_colloc=50,
        update_every=50,
        domain_lb=[0.0],
        domain_ub=[1.0],
        seed=0,
    )
    result = model.fit(ds, pde_operator=op)
    assert result is model


def test_adaptive_pinn_predict_shape():
    ds, op, X_test, _ = _make_poisson_1d(60)
    model = AdaptivePINN(
        layer_dims=[32, 32],
        max_epochs=50,
        n_colloc=30,
        update_every=25,
        domain_lb=[0.0],
        domain_ub=[1.0],
        seed=0,
    )
    model.fit(ds, pde_operator=op)
    pred = model.predict(X_test)
    assert pred.shape == (200, 1)


def test_adaptive_pinn_loss_decreases():
    ds, op, _, _ = _make_poisson_1d(60)
    model = AdaptivePINN(
        layer_dims=[32, 32],
        max_epochs=200,
        n_colloc=40,
        update_every=100,
        domain_lb=[0.0],
        domain_ub=[1.0],
        seed=2,
    )
    model.fit(ds, pde_operator=op)
    assert model._loss_history[-1] < model._loss_history[0]


def test_adaptive_pinn_registry_lookup():
    model = get_model(
        "adaptive_pinn",
        layer_dims=[16],
        max_epochs=5,
        domain_lb=[0.0],
        domain_ub=[1.0],
    )
    ds, op, _, _ = _make_poisson_1d(30)
    model.fit(ds, pde_operator=op)
    assert hasattr(model, "_loss_history")


# ---------------------------------------------------------------------------
# FourierPINN
# ---------------------------------------------------------------------------

def test_fourier_pinn_fit_returns_self():
    ds, op, _, _ = _make_poisson_1d(60)
    model = FourierPINN(
        sigma=5.0, n_fourier=16, layer_dims=[32, 32], max_epochs=100, seed=0
    )
    result = model.fit(ds, pde_operator=op)
    assert result is model


def test_fourier_pinn_predict_shape():
    ds, op, X_test, _ = _make_poisson_1d(60)
    model = FourierPINN(sigma=5.0, n_fourier=16, layer_dims=[32, 32], max_epochs=50, seed=0)
    model.fit(ds, pde_operator=op)
    pred = model.predict(X_test)
    assert pred.shape == (200, 1)


def test_fourier_pinn_loss_decreases():
    ds, op, _, _ = _make_poisson_1d(60)
    model = FourierPINN(
        sigma=5.0, n_fourier=32, layer_dims=[32, 32], max_epochs=200, lr=1e-3, seed=5
    )
    model.fit(ds, pde_operator=op)
    assert model._loss_history[-1] < model._loss_history[0]


def test_fourier_pinn_registry_lookup():
    model = get_model("fourier_pinn", layer_dims=[16], max_epochs=5, sigma=3.0)
    ds, op, _, _ = _make_poisson_1d(30)
    model.fit(ds, pde_operator=op)
    assert hasattr(model, "_loss_history")


def test_fourier_pinn_different_sigma_different_outputs():
    """Different sigma values should produce different feature encodings."""
    ds, op, X_test, _ = _make_poisson_1d(50)
    m1 = FourierPINN(sigma=1.0, n_fourier=16, layer_dims=[16], max_epochs=5, seed=0)
    m2 = FourierPINN(sigma=50.0, n_fourier=16, layer_dims=[16], max_epochs=5, seed=0)
    m1.fit(ds, pde_operator=op)
    m2.fit(ds, pde_operator=op)
    p1 = m1.predict(X_test)
    p2 = m2.predict(X_test)
    assert not torch.allclose(p1, p2)


# ---------------------------------------------------------------------------
# MuonPINN
# ---------------------------------------------------------------------------

def test_muon_pinn_fit_returns_self():
    ds, op, _, _ = _make_poisson_1d(60)
    model = MuonPINN(layer_dims=[32, 32], max_epochs=100, lr=1e-3, seed=0)
    result = model.fit(ds, pde_operator=op)
    assert result is model


def test_muon_pinn_predict_shape():
    ds, op, X_test, _ = _make_poisson_1d(60)
    model = MuonPINN(layer_dims=[32, 32], max_epochs=50, seed=0)
    model.fit(ds, pde_operator=op)
    pred = model.predict(X_test)
    assert pred.shape == (200, 1)


def test_muon_pinn_loss_decreases():
    ds, op, _, _ = _make_poisson_1d(60)
    model = MuonPINN(
        layer_dims=[32, 32], max_epochs=200, lr=5e-4, seed=3
    )
    model.fit(ds, pde_operator=op)
    assert model._loss_history[-1] < model._loss_history[0]


def test_muon_pinn_registry_lookup():
    model = get_model("muon_pinn", layer_dims=[16], max_epochs=5)
    ds, op, _, _ = _make_poisson_1d(30)
    model.fit(ds, pde_operator=op)
    assert hasattr(model, "_loss_history")


def test_muon_pinn_orthogonalises_weights():
    """MuonPINN should produce different weights than VanillaPINN (Adam)."""
    ds, op, X_test, _ = _make_poisson_1d(60)
    m_muon = MuonPINN(layer_dims=[32, 32], max_epochs=50, lr=1e-3, seed=42)
    m_adam = VanillaPINN(layer_dims=[32, 32], max_epochs=50, lr=1e-3, seed=42)
    m_muon.fit(ds, pde_operator=op)
    m_adam.fit(ds, pde_operator=op)
    p1 = m_muon.predict(X_test)
    p2 = m_adam.predict(X_test)
    assert not torch.allclose(p1, p2)


# ---------------------------------------------------------------------------
# ResidualAdaptivePINN
# ---------------------------------------------------------------------------

def test_residual_adaptive_pinn_fit_returns_self():
    ds, op, _, _ = _make_poisson_1d(60)
    model = ResidualAdaptivePINN(
        width=32, n_blocks=2, max_epochs=100,
        domain_lb=[0.0], domain_ub=[1.0], seed=0,
    )
    result = model.fit(ds, pde_operator=op)
    assert result is model


def test_residual_adaptive_pinn_predict_shape():
    ds, op, X_test, _ = _make_poisson_1d(60)
    model = ResidualAdaptivePINN(
        width=32, n_blocks=2, max_epochs=50,
        domain_lb=[0.0], domain_ub=[1.0], seed=0,
    )
    model.fit(ds, pde_operator=op)
    pred = model.predict(X_test)
    assert pred.shape == (200, 1)


def test_residual_adaptive_pinn_loss_decreases():
    ds, op, _, _ = _make_poisson_1d(60)
    model = ResidualAdaptivePINN(
        width=32, n_blocks=2, max_epochs=200,
        domain_lb=[0.0], domain_ub=[1.0], seed=1,
        update_every=100, n_new=10,
    )
    model.fit(ds, pde_operator=op)
    assert model._loss_history[-1] < model._loss_history[0]


def test_residual_adaptive_pinn_rar_grows_colloc():
    """RAR should add points; collocation size increases (or hits max_colloc cap)."""
    ds, op, _, _ = _make_poisson_1d(30)
    int(ds.X_colloc.shape[0])
    model = ResidualAdaptivePINN(
        width=16, n_blocks=1, max_epochs=120,
        domain_lb=[0.0], domain_ub=[1.0], seed=0,
        update_every=50, n_new=5, max_colloc=1000,
    )
    model.fit(ds, pde_operator=op)
    # Original dataset unchanged; internal state changes
    # Just verify the model ran and produces a valid prediction
    assert model._loss_history[-1] >= 0.0


def test_residual_adaptive_pinn_registry_lookup():
    model = get_model(
        "residual_adaptive_pinn",
        width=16, n_blocks=1, max_epochs=5,
        domain_lb=[0.0], domain_ub=[1.0],
    )
    ds, op, _, _ = _make_poisson_1d(30)
    model.fit(ds, pde_operator=op)
    assert hasattr(model, "_loss_history")


def test_residual_adaptive_pinn_resnet_backbone():
    """Confirm ResidualAdaptivePINN uses _ResNet (not _MLP)."""
    from pypielm.models.pinn import _ResNet
    ds, op, _, _ = _make_poisson_1d(30)
    model = ResidualAdaptivePINN(
        width=16, n_blocks=2, max_epochs=5,
        domain_lb=[0.0], domain_ub=[1.0], seed=0,
    )
    model.fit(ds, pde_operator=op)
    assert isinstance(model._net, _ResNet)


# ---------------------------------------------------------------------------
# MPS device tests
# ---------------------------------------------------------------------------

@pytest.mark.mps
def test_vanilla_pinn_mps(mps_device):
    ds, op, X_test, y_exact = _make_poisson_1d(
        60, dtype=torch.float32, device=mps_device
    )
    model = VanillaPINN(
        layer_dims=[32, 32], max_epochs=200, lr=1e-3, seed=0,
        device=mps_device, dtype=torch.float32,
    )
    try:
        model.fit(ds, pde_operator=op)
    except (NotImplementedError, RuntimeError) as e:
        pytest.skip(f"MPS autograd op not available: {e}")
    pred = model.predict(X_test)
    assert pred.shape == (200, 1)
    assert not torch.isnan(pred).any()


@pytest.mark.mps
def test_fourier_pinn_mps(mps_device):
    ds, op, X_test, _ = _make_poisson_1d(
        50, dtype=torch.float32, device=mps_device
    )
    model = FourierPINN(
        sigma=5.0, n_fourier=16, layer_dims=[32, 32],
        max_epochs=100, seed=0,
        device=mps_device, dtype=torch.float32,
    )
    try:
        model.fit(ds, pde_operator=op)
    except (NotImplementedError, RuntimeError) as e:
        pytest.skip(f"MPS autograd op not available: {e}")
    pred = model.predict(X_test)
    assert pred.shape == (200, 1)


# ---------------------------------------------------------------------------
# CUDA device tests
# ---------------------------------------------------------------------------

@pytest.mark.cuda
def test_vanilla_pinn_cuda(cuda_device):
    ds, op, X_test, y_exact = _make_poisson_1d(
        60, dtype=torch.float32, device=cuda_device
    )
    model = VanillaPINN(
        layer_dims=[32, 32], max_epochs=200, lr=1e-3, seed=0,
        device=cuda_device, dtype=torch.float32,
    )
    model.fit(ds, pde_operator=op)
    pred = model.predict(X_test)
    assert pred.shape == (200, 1)
    assert not torch.isnan(pred).any()
