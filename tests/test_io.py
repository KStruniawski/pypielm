"""Tests for pypielm.io checkpoint and export."""

from __future__ import annotations

import pytest
import torch

from pypielm.core.solver import WeightedLinearSystem
from pypielm.data.dataset import PIELMDataset
from pypielm.io.checkpoint import load_model, save_model
from pypielm.io.export import to_torchscript
from pypielm.models.vanilla import CorePIELM
from pypielm.pde.operators import AnalyticLaplacian

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_core_model() -> tuple[CorePIELM, PIELMDataset]:
    X = torch.linspace(0.0, 1.0, 60, dtype=torch.float64).unsqueeze(1)
    X_bc = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    y_bc = torch.zeros(2, 1, dtype=torch.float64)
    ds = PIELMDataset(X_colloc=X, X_bc=X_bc, y_bc=y_bc)

    def op(fm, Xc):
        H = AnalyticLaplacian(fm)(Xc)
        f = -2.0 * torch.ones(Xc.shape[0], 1, dtype=Xc.dtype)
        return WeightedLinearSystem(H, f)

    model = CorePIELM(hidden_dim=50, seed=0).fit(ds, pde_operator=op)
    return model, ds


# ---------------------------------------------------------------------------
# save_model / load_model
# ---------------------------------------------------------------------------

def test_save_creates_file(tmp_path):
    model, _ = _make_core_model()
    p = tmp_path / "model.pt"
    save_model(model, p)
    assert p.exists()


def test_save_load_roundtrip(tmp_path):
    model, ds = _make_core_model()
    X_test = torch.linspace(0.0, 1.0, 30, dtype=torch.float64).unsqueeze(1)
    preds_before = model.predict(X_test)

    p = tmp_path / "model.pt"
    save_model(model, p)
    loaded = load_model(p)

    preds_after = loaded.predict(X_test)
    assert torch.allclose(preds_before, preds_after, atol=1e-10), \
        f"max diff = {(preds_before - preds_after).abs().max().item():.2e}"


def test_load_model_type(tmp_path):
    model, _ = _make_core_model()
    p = tmp_path / "model.pt"
    save_model(model, p)
    loaded = load_model(p)
    assert isinstance(loaded, CorePIELM)


def test_save_raises_if_exists(tmp_path):
    model, _ = _make_core_model()
    p = tmp_path / "model.pt"
    save_model(model, p)
    with pytest.raises(FileExistsError):
        save_model(model, p, overwrite=False)


def test_save_overwrite_flag(tmp_path):
    model, _ = _make_core_model()
    p = tmp_path / "model.pt"
    save_model(model, p)
    save_model(model, p, overwrite=True)  # should not raise
    assert p.exists()


def test_load_model_wrong_path_raises():
    with pytest.raises(FileNotFoundError):
        load_model("/nonexistent/path/model.pt")


def test_save_load_without_config(tmp_path):
    model, ds = _make_core_model()
    X_test = torch.linspace(0.0, 1.0, 20, dtype=torch.float64).unsqueeze(1)
    preds_before = model.predict(X_test)

    p = tmp_path / "model_noconfig.pt"
    save_model(model, p, include_config=False)
    # Load with explicit model_class since config is absent
    loaded = load_model(p, model_class=CorePIELM)
    preds_after = loaded.predict(X_test)
    assert torch.allclose(preds_before, preds_after, atol=1e-10)


# ---------------------------------------------------------------------------
# to_torchscript
# ---------------------------------------------------------------------------

def test_to_torchscript_creates_file(tmp_path):
    model, _ = _make_core_model()
    p = tmp_path / "model_ts.pt"
    x = torch.zeros(1, 1, dtype=torch.float64)
    to_torchscript(model, p, example_input=x, method="trace")
    assert p.exists()


def test_to_torchscript_inference_matches(tmp_path):
    model, _ = _make_core_model()
    X_test = torch.linspace(0.0, 1.0, 20, dtype=torch.float64).unsqueeze(1)
    preds_before = model.predict(X_test)

    p = tmp_path / "model_ts.pt"
    scripted = to_torchscript(model, p, example_input=X_test[:1], method="trace")
    preds_ts = scripted(X_test)
    assert torch.allclose(preds_before, preds_ts, atol=1e-10)


def test_to_torchscript_bad_method_raises(tmp_path):
    model, _ = _make_core_model()
    with pytest.raises(ValueError, match="method must be"):
        to_torchscript(model, tmp_path / "x.pt", method="invalid")


def test_to_onnx_skips_without_onnx(tmp_path):
    """to_onnx raises ImportError if onnx is not installed."""
    pytest.importorskip("onnx", reason="onnx not installed; skipping ONNX tests")
    from pypielm.io.export import to_onnx
    model, _ = _make_core_model()
    to_onnx(model, tmp_path / "model.onnx", input_dim=1)
    assert (tmp_path / "model.onnx").exists()

