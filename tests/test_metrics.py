"""Tests for pypielm.metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pypielm.metrics.metrics import (
    MetricsBundle,
    mae,
    max_error,
    r2_score,
    relative_l2,
    rmse,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ones(n: int = 50) -> tuple[torch.Tensor, torch.Tensor]:
    y_true = torch.linspace(0.0, 1.0, n).unsqueeze(1).float()
    return y_true.clone(), y_true  # perfect prediction


def _noisy(n: int = 50, noise: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    y_true = torch.linspace(0.0, 1.0, n).unsqueeze(1).float()
    y_pred = y_true + noise
    return y_pred, y_true


# ---------------------------------------------------------------------------
# rmse
# ---------------------------------------------------------------------------

def test_rmse_perfect_prediction():
    y_pred, y_true = _ones()
    assert rmse(y_pred, y_true) == pytest.approx(0.0, abs=1e-8)


def test_rmse_known_value():
    y_pred = torch.tensor([[1.0], [1.0]])
    y_true = torch.tensor([[0.0], [0.0]])
    assert rmse(y_pred, y_true) == pytest.approx(1.0, rel=1e-6)


def test_rmse_numpy_input():
    y_pred = np.ones((10, 1), dtype=np.float32)
    y_true = np.zeros((10, 1), dtype=np.float32)
    assert rmse(y_pred, y_true) == pytest.approx(1.0, rel=1e-6)


def test_rmse_returns_float():
    y_pred, y_true = _noisy()
    assert isinstance(rmse(y_pred, y_true), float)


# ---------------------------------------------------------------------------
# mae
# ---------------------------------------------------------------------------

def test_mae_perfect_prediction():
    y_pred, y_true = _ones()
    assert mae(y_pred, y_true) == pytest.approx(0.0, abs=1e-8)


def test_mae_known_value():
    y_pred = torch.tensor([[2.0], [2.0]])
    y_true = torch.tensor([[0.0], [0.0]])
    assert mae(y_pred, y_true) == pytest.approx(2.0, rel=1e-6)


def test_mae_returns_float():
    y_pred, y_true = _noisy()
    assert isinstance(mae(y_pred, y_true), float)


# ---------------------------------------------------------------------------
# relative_l2
# ---------------------------------------------------------------------------

def test_relative_l2_zero_for_perfect():
    y_pred, y_true = _ones()
    assert relative_l2(y_pred, y_true) == pytest.approx(0.0, abs=1e-8)


def test_relative_l2_positive_for_error():
    y_pred, y_true = _noisy(noise=0.1)
    assert relative_l2(y_pred, y_true) > 0.0


def test_relative_l2_returns_float():
    y_pred, y_true = _noisy()
    assert isinstance(relative_l2(y_pred, y_true), float)


def test_relative_l2_numpy_input():
    n = 20
    y_true = np.linspace(0.0, 1.0, n).reshape(-1, 1).astype(np.float32)
    y_pred = y_true.copy()
    assert relative_l2(y_pred, y_true) == pytest.approx(0.0, abs=1e-7)


# ---------------------------------------------------------------------------
# max_error
# ---------------------------------------------------------------------------

def test_max_error_perfect():
    y_pred, y_true = _ones()
    assert max_error(y_pred, y_true) == pytest.approx(0.0, abs=1e-8)


def test_max_error_known():
    y_pred = torch.tensor([[0.0], [5.0], [1.0]])
    y_true = torch.tensor([[0.0], [2.0], [1.0]])
    assert max_error(y_pred, y_true) == pytest.approx(3.0, rel=1e-6)


def test_max_error_returns_float():
    y_pred, y_true = _noisy()
    assert isinstance(max_error(y_pred, y_true), float)


# ---------------------------------------------------------------------------
# r2_score
# ---------------------------------------------------------------------------

def test_r2_score_is_one_for_perfect():
    y_pred, y_true = _ones()
    assert r2_score(y_pred, y_true) == pytest.approx(1.0, rel=1e-5)


def test_r2_score_negative_for_constant_prediction():
    y_true = torch.tensor([[1.0], [2.0], [3.0]])
    y_pred = torch.tensor([[0.0], [0.0], [0.0]])  # bad predictor
    assert r2_score(y_pred, y_true) < 0.0


def test_r2_score_returns_float():
    y_pred, y_true = _noisy()
    assert isinstance(r2_score(y_pred, y_true), float)


# ---------------------------------------------------------------------------
# MetricsBundle
# ---------------------------------------------------------------------------

def test_metrics_bundle_attributes():
    y_pred, y_true = _noisy()
    mb = MetricsBundle(y_pred, y_true)
    assert hasattr(mb, "rmse")
    assert hasattr(mb, "mae")
    assert hasattr(mb, "rel_l2")
    assert hasattr(mb, "max_err")
    assert hasattr(mb, "r2")


def test_metrics_bundle_perfect_prediction():
    y_pred, y_true = _ones()
    mb = MetricsBundle(y_pred, y_true)
    assert mb.rmse == pytest.approx(0.0, abs=1e-7)
    assert mb.rel_l2 == pytest.approx(0.0, abs=1e-7)
    assert mb.r2 == pytest.approx(1.0, rel=1e-5)


def test_metrics_bundle_to_dict():
    y_pred, y_true = _noisy()
    mb = MetricsBundle(y_pred, y_true)
    d = mb.to_dict()
    assert set(d) == {"rmse", "mae", "rel_l2", "max_err", "r2"}
    assert all(isinstance(v, float) for v in d.values())


def test_metrics_bundle_repr():
    y_pred, y_true = _ones()
    mb = MetricsBundle(y_pred, y_true)
    r = repr(mb)
    assert "MetricsBundle" in r
    assert "rmse" in r
    assert "r2" in r


def test_metrics_bundle_numpy_input():
    n = 30
    y_true = np.linspace(0.0, 1.0, n).reshape(-1, 1).astype(np.float32)
    y_pred = y_true + 0.05
    mb = MetricsBundle(y_pred, y_true)
    assert mb.rmse > 0.0
    assert mb.r2 < 1.0

