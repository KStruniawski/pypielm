"""Comprehensive tests for the PyPIELM data layer.

Coverage:
  - PIELMDataset construction, from_arrays, to(device), __repr__
  - Normalizer (minmax + zscore), FeatureExpander, Pipeline
  - CSVAdapter round-trip
  - NPZAdapter round-trip
  - PINNacleAdapter (.dat, .npz, .csv)
  - TorchDatasetAdapter (all roles)
  - auto_load routing for every supported extension
  - Edge-cases: 1-D inputs, no-header CSVs, empty column maps, etc.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.utils.data

from pypielm.data import (
    CSVAdapter,
    FeatureExpander,
    Normalizer,
    NPZAdapter,
    PDEBenchAdapter,
    PIELMDataset,
    PINNacleAdapter,
    Pipeline,
    TorchDatasetAdapter,
    auto_load,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def _rand_tensor(n: int, d: int, dtype=torch.float64) -> torch.Tensor:
    return torch.randn(n, d, dtype=dtype)


# ---------------------------------------------------------------------------
# PIELMDataset
# ---------------------------------------------------------------------------

class TestPIELMDatasetConstruction:
    def test_minimal(self):
        X = torch.ones(10, 2, dtype=torch.float64)
        ds = PIELMDataset(X_colloc=X)
        assert ds.X_colloc is X
        assert ds.X_bc is None
        assert ds.y_bc is None
        assert ds.X_ic is None
        assert ds.y_ic is None
        assert ds.X_data is None
        assert ds.y_data is None
        assert ds.meta == {}

    def test_full_fields(self):
        X = torch.ones(20, 3, dtype=torch.float64)
        Xbc = torch.zeros(5, 3, dtype=torch.float64)
        ybc = torch.zeros(5, 1, dtype=torch.float64)
        Xic = torch.zeros(4, 3, dtype=torch.float64)
        yic = torch.ones(4, 1, dtype=torch.float64)
        Xd = torch.rand(8, 3, dtype=torch.float64)
        yd = torch.rand(8, 1, dtype=torch.float64)
        ds = PIELMDataset(
            X_colloc=X, X_bc=Xbc, y_bc=ybc,
            X_ic=Xic, y_ic=yic,
            X_data=Xd, y_data=yd,
            meta={"task": "test"},
        )
        assert ds.X_bc.shape == (5, 3)
        assert ds.y_bc.shape == (5, 1)
        assert ds.meta["task"] == "test"

    def test_repr_counts(self):
        X = torch.ones(30, 2)
        Xbc = torch.zeros(5, 2)
        ybc = torch.zeros(5, 1)
        ds = PIELMDataset(X_colloc=X, X_bc=Xbc, y_bc=ybc)
        r = repr(ds)
        assert "colloc=30" in r
        assert "bc=5" in r
        assert "ic=0" in r
        assert "obs=0" in r

    def test_repr_all_counts(self):
        ds = PIELMDataset(
            X_colloc=torch.ones(10, 1),
            X_bc=torch.ones(4, 1),
            y_bc=torch.ones(4, 1),
            X_ic=torch.ones(3, 1),
            y_ic=torch.ones(3, 1),
            X_data=torch.ones(7, 1),
            y_data=torch.ones(7, 1),
        )
        r = repr(ds)
        assert "colloc=10" in r
        assert "bc=4" in r
        assert "ic=3" in r
        assert "obs=7" in r


class TestPIELMDatasetFromArrays:
    def test_numpy_array(self):
        X = np.linspace(0, 1, 20).reshape(-1, 1)
        y = np.sin(X)
        ds = PIELMDataset.from_arrays(X, y_data=y)
        assert ds.X_colloc.dtype == torch.float64
        assert ds.X_colloc.shape == (20, 1)
        assert ds.y_data.shape == (20, 1)

    def test_list_input(self):
        X = [[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]]
        ds = PIELMDataset.from_arrays(X)
        assert ds.X_colloc.shape == (3, 2)

    def test_1d_input_becomes_2d(self):
        X = np.array([0.1, 0.2, 0.3])
        ds = PIELMDataset.from_arrays(X)
        assert ds.X_colloc.ndim == 2
        assert ds.X_colloc.shape == (3, 1)

    def test_custom_dtype(self):
        X = np.ones((5, 2))
        ds = PIELMDataset.from_arrays(X, dtype=torch.float32)
        assert ds.X_colloc.dtype == torch.float32

    def test_meta_passed_through(self):
        X = np.ones((4, 1))
        ds = PIELMDataset.from_arrays(X, meta={"src": "unit_test"})
        assert ds.meta["src"] == "unit_test"

    def test_meta_default_empty_dict(self):
        X = np.ones((4, 1))
        ds = PIELMDataset.from_arrays(X)
        assert ds.meta == {}

    def test_bc_ic_data_all_optional(self):
        X = np.ones((6, 2))
        ds = PIELMDataset.from_arrays(X)
        assert ds.X_bc is None and ds.y_bc is None
        assert ds.X_ic is None and ds.y_ic is None
        assert ds.X_data is None and ds.y_data is None

    def test_y_bc_1d_becomes_2d(self):
        X = np.zeros((5, 2))
        Xbc = np.zeros((3, 2))
        ybc = np.zeros(3)  # 1-D
        ds = PIELMDataset.from_arrays(X, X_bc=Xbc, y_bc=ybc)
        assert ds.y_bc.ndim == 2
        assert ds.y_bc.shape == (3, 1)

    def test_tensor_input_passthrough(self):
        X = torch.ones(8, 3, dtype=torch.float64)
        ds = PIELMDataset.from_arrays(X)
        assert ds.X_colloc.dtype == torch.float64


class TestPIELMDatasetTo:
    def test_to_same_device_noop(self):
        X = torch.ones(5, 2, dtype=torch.float64)
        ds = PIELMDataset(X_colloc=X)
        ds2 = ds.to("cpu")
        assert ds2.X_colloc.device.type == "cpu"

    def test_to_dtype_cast(self):
        X = torch.ones(5, 2, dtype=torch.float64)
        ds = PIELMDataset(X_colloc=X)
        ds2 = ds.to("cpu", dtype=torch.float32)
        assert ds2.X_colloc.dtype == torch.float32

    def test_to_preserves_none_fields(self):
        X = torch.ones(5, 2, dtype=torch.float64)
        ds = PIELMDataset(X_colloc=X)
        ds2 = ds.to("cpu")
        assert ds2.X_bc is None

    def test_to_moves_all_non_none_fields(self):
        X = torch.ones(5, 2, dtype=torch.float64)
        Xbc = torch.zeros(3, 2, dtype=torch.float64)
        ybc = torch.zeros(3, 1, dtype=torch.float64)
        ds = PIELMDataset(X_colloc=X, X_bc=Xbc, y_bc=ybc)
        ds2 = ds.to("cpu", dtype=torch.float32)
        assert ds2.X_bc.dtype == torch.float32
        assert ds2.y_bc.dtype == torch.float32

    def test_original_unchanged(self):
        X = torch.ones(5, 2, dtype=torch.float64)
        ds = PIELMDataset(X_colloc=X)
        ds.to("cpu", dtype=torch.float32)
        assert ds.X_colloc.dtype == torch.float64

    @pytest.mark.mps
    def test_to_mps(self, mps_device: torch.device):
        # MPS does not support float64; use float32
        X = torch.ones(5, 2, dtype=torch.float32)
        ds = PIELMDataset(X_colloc=X)
        ds_mps = ds.to(mps_device)
        assert ds_mps.X_colloc.device.type == "mps"


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------

class TestNormalizerMinMax:
    def test_range(self):
        X = torch.arange(10, dtype=torch.float64).unsqueeze(1)
        n = Normalizer("minmax")
        Xn = n.fit_transform(X)
        assert Xn.min().item() == pytest.approx(0.0)
        assert Xn.max().item() == pytest.approx(1.0)

    def test_inverse_roundtrip(self):
        X = torch.randn(50, 3, dtype=torch.float64)
        n = Normalizer("minmax")
        Xn = n.fit_transform(X)
        Xr = n.inverse_transform(Xn)
        assert torch.allclose(X, Xr, atol=1e-12)

    def test_multicolumn(self):
        X = torch.tensor([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]], dtype=torch.float64)
        n = Normalizer("minmax")
        Xn = n.fit_transform(X)
        assert Xn[:, 0].min().item() == pytest.approx(0.0)
        assert Xn[:, 0].max().item() == pytest.approx(1.0)
        assert Xn[:, 1].min().item() == pytest.approx(0.0)
        assert Xn[:, 1].max().item() == pytest.approx(1.0)

    def test_transform_before_fit_raises(self):
        n = Normalizer("minmax")
        with pytest.raises(RuntimeError, match="fit"):
            n.transform(torch.ones(3, 1))

    def test_1d_input(self):
        X = torch.arange(5, dtype=torch.float64)
        n = Normalizer("minmax")
        Xn = n.fit_transform(X)
        assert Xn.ndim == 2

    def test_no_train_contamination(self):
        """Test data is not used to compute statistics."""
        train = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        test = torch.tensor([[2.0], [-1.0]], dtype=torch.float64)
        n = Normalizer("minmax")
        n.fit(train)
        Xtest = n.transform(test)
        # test values outside [0,1] are expected (no clamping)
        assert Xtest[0, 0].item() == pytest.approx(2.0)
        assert Xtest[1, 0].item() == pytest.approx(-1.0)

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method"):
            Normalizer("l2")


class TestNormalizerZScore:
    def test_mean_std(self):
        X = torch.randn(100, 2, dtype=torch.float64) * 5 + 3
        n = Normalizer("zscore")
        Xn = n.fit_transform(X)
        assert Xn.mean(dim=0).abs().max().item() < 1e-6
        # std close to 1
        assert (Xn.std(dim=0) - 1).abs().max().item() < 0.05

    def test_inverse_roundtrip(self):
        X = torch.randn(60, 4, dtype=torch.float64)
        n = Normalizer("zscore")
        Xr = n.inverse_transform(n.fit_transform(X))
        assert torch.allclose(X, Xr, atol=1e-12)


# ---------------------------------------------------------------------------
# FeatureExpander
# ---------------------------------------------------------------------------

class TestFeatureExpander:
    def test_degree1_unchanged(self):
        X = torch.randn(10, 3, dtype=torch.float64)
        fe = FeatureExpander(degree=1)
        Xout = fe.fit_transform(X)
        assert torch.allclose(Xout, X)

    def test_degree2_shape(self):
        X = torch.ones(5, 2, dtype=torch.float64)
        fe = FeatureExpander(degree=2)
        Xout = fe.fit_transform(X)
        # degree=2 → original(2) + squared(2) = 4 columns
        assert Xout.shape == (5, 4)

    def test_degree3_shape(self):
        X = torch.ones(4, 3, dtype=torch.float64)
        fe = FeatureExpander(degree=3)
        Xout = fe.fit_transform(X)
        # 3 + 3 + 3 = 9
        assert Xout.shape == (4, 9)

    def test_trig_shape(self):
        X = torch.ones(5, 2, dtype=torch.float64)
        fe = FeatureExpander(degree=1, trig=True)
        Xout = fe.fit_transform(X)
        # original(2) + sin(2) + cos(2) = 6
        assert Xout.shape == (5, 6)

    def test_trig_values(self):
        X = torch.zeros(1, 1, dtype=torch.float64)  # x = 0
        fe = FeatureExpander(degree=1, trig=True)
        Xout = fe.fit_transform(X)
        # [0, sin(0)=0, cos(0)=1]
        assert Xout[0, 0].item() == pytest.approx(0.0)
        assert Xout[0, 1].item() == pytest.approx(0.0)   # sin(0)
        assert Xout[0, 2].item() == pytest.approx(1.0)   # cos(0)

    def test_transform_equals_fit_transform(self):
        X = torch.randn(8, 2, dtype=torch.float64)
        fe = FeatureExpander(degree=2, trig=True)
        assert torch.allclose(fe.fit_transform(X), fe.transform(X))

    def test_degree_less_than_1_raises(self):
        with pytest.raises(ValueError, match="degree"):
            FeatureExpander(degree=0)

    def test_1d_input(self):
        X = torch.arange(5, dtype=torch.float64)
        fe = FeatureExpander(degree=2)
        Xout = fe.fit_transform(X)
        assert Xout.ndim == 2


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_single_step(self):
        X = torch.arange(10, dtype=torch.float64).reshape(10, 1) * 1.0
        n = Normalizer("minmax")
        p = Pipeline([n])
        Xout = p.fit_transform(X)
        assert Xout.min().item() == pytest.approx(0.0)
        assert Xout.max().item() == pytest.approx(1.0)

    def test_two_steps(self):
        X = torch.randn(20, 2, dtype=torch.float64)
        p = Pipeline([Normalizer("zscore"), FeatureExpander(degree=2)])
        Xout = p.fit_transform(X)
        # 2 original + 2 squared = 4 cols
        assert Xout.shape == (20, 4)

    def test_transform_after_fit(self):
        X_train = torch.randn(20, 2, dtype=torch.float64)
        X_test = torch.randn(5, 2, dtype=torch.float64)
        p = Pipeline([Normalizer("zscore"), FeatureExpander(degree=1)])
        p.fit_transform(X_train)
        Xout = p.transform(X_test)
        assert Xout.shape == (5, 2)

    def test_empty_steps_raises(self):
        with pytest.raises(ValueError, match="step"):
            Pipeline([])


# ---------------------------------------------------------------------------
# CSVAdapter
# ---------------------------------------------------------------------------

class TestCSVAdapter:
    def test_basic_no_header(self, tmp_path: Path):
        data = np.column_stack([
            np.linspace(0, 1, 20),
            np.linspace(0, 1, 20) ** 2,
        ])
        np.savetxt(tmp_path / "data.csv", data, delimiter=",")
        ds = CSVAdapter(tmp_path / "data.csv").load()
        assert ds.X_colloc.shape == (20, 1)
        assert ds.y_data.shape == (20, 1)
        assert ds.X_colloc.dtype == torch.float64

    def test_with_header(self, tmp_path: Path):
        rows = [(float(i), float(i ** 2)) for i in range(10)]
        csv_path = tmp_path / "named.csv"
        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["x", "y"])
            w.writerows(rows)
        ds = CSVAdapter(csv_path).load()
        assert ds.X_colloc.shape == (10, 1)
        assert ds.y_data.shape == (10, 1)
        np.testing.assert_allclose(
            ds.X_colloc[:, 0].numpy(),
            np.arange(10, dtype=float),
        )

    def test_column_map_by_name(self, tmp_path: Path):
        csv_path = tmp_path / "map.csv"
        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["a", "b", "c"])
            for i in range(5):
                w.writerow([i, i * 2, i * 3])
        ds = CSVAdapter(
            csv_path,
            column_map={"X_colloc": ["a", "b"], "y_data": ["c"]},
        ).load()
        assert ds.X_colloc.shape == (5, 2)
        assert ds.y_data.shape == (5, 1)

    def test_column_map_missing_x_colloc_raises(self, tmp_path: Path):
        csv_path = tmp_path / "bad.csv"
        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["x", "y"])
            w.writerow([1.0, 2.0])
        with pytest.raises(ValueError, match="X_colloc"):
            CSVAdapter(csv_path, column_map={"y_data": ["y"]}).load()

    def test_auto_load_routing(self, tmp_path: Path):
        data = np.column_stack([
            np.linspace(0, 1, 15),
            np.linspace(0, 1, 15),
        ])
        path = tmp_path / "routed.csv"
        np.savetxt(path, data, delimiter=",")
        ds = auto_load(path)
        assert isinstance(ds, PIELMDataset)
        assert ds.X_colloc.shape[0] == 15

    def test_round_trip_values(self, tmp_path: Path):
        X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        y = np.array([[1.0], [2.0], [3.0]])
        data = np.hstack([X, y])
        path = tmp_path / "rt.csv"
        np.savetxt(path, data, delimiter=",")
        ds = CSVAdapter(path).load()
        np.testing.assert_allclose(ds.X_colloc.numpy(), X, atol=1e-10)
        np.testing.assert_allclose(ds.y_data.numpy(), y, atol=1e-10)

    def test_dtype_float32(self, tmp_path: Path):
        data = np.ones((5, 2))
        path = tmp_path / "f32.csv"
        np.savetxt(path, data, delimiter=",")
        ds = CSVAdapter(path, dtype=torch.float32).load()
        assert ds.X_colloc.dtype == torch.float32


# ---------------------------------------------------------------------------
# NPZAdapter
# ---------------------------------------------------------------------------

class TestNPZAdapter:
    def test_standard_keys(self, tmp_path: Path):
        rng = _rng()
        X = rng.standard_normal((30, 2))
        y = rng.standard_normal((30, 1))
        path = tmp_path / "data.npz"
        np.savez(path, X_colloc=X, y_data=y)
        ds = NPZAdapter(path).load()
        assert ds.X_colloc.shape == (30, 2)
        assert ds.y_data.shape == (30, 1)
        assert ds.X_bc is None

    def test_all_fields(self, tmp_path: Path):
        rng = _rng()
        data = {
            "X_colloc": rng.standard_normal((20, 3)),
            "X_bc": rng.standard_normal((5, 3)),
            "y_bc": rng.standard_normal((5, 1)),
            "X_ic": rng.standard_normal((4, 3)),
            "y_ic": rng.standard_normal((4, 1)),
            "X_data": rng.standard_normal((8, 3)),
            "y_data": rng.standard_normal((8, 1)),
        }
        path = tmp_path / "full.npz"
        np.savez(path, **data)
        ds = NPZAdapter(path).load()
        assert ds.X_bc.shape == (5, 3)
        assert ds.y_bc.shape == (5, 1)
        assert ds.X_ic.shape == (4, 3)

    def test_two_array_fallback(self, tmp_path: Path):
        X = np.linspace(0, 1, 10).reshape(-1, 1)
        y = (X ** 2)
        path = tmp_path / "fallback.npz"
        np.savez(path, arr0=X, arr1=y)
        ds = NPZAdapter(path).load()
        assert ds.X_colloc.shape[0] == 10
        assert ds.y_data.shape[0] == 10

    def test_unknown_keys_raises(self, tmp_path: Path):
        path = tmp_path / "bad.npz"
        np.savez(path, A=np.ones(5), B=np.ones(5), C=np.ones(5))
        with pytest.raises(ValueError, match="X_colloc"):
            NPZAdapter(path).load()

    def test_round_trip_values(self, tmp_path: Path):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        path = tmp_path / "rt.npz"
        np.savez(path, X_colloc=X)
        ds = NPZAdapter(path).load()
        np.testing.assert_allclose(ds.X_colloc.numpy(), X)

    def test_auto_load_routing(self, tmp_path: Path):
        X = np.ones((10, 2))
        path = tmp_path / "rt.npz"
        np.savez(path, X_colloc=X)
        ds = auto_load(path)
        assert isinstance(ds, PIELMDataset)
        assert ds.X_colloc.shape == (10, 2)

    def test_dtype_float32(self, tmp_path: Path):
        X = np.ones((5, 2))
        path = tmp_path / "f32.npz"
        np.savez(path, X_colloc=X)
        ds = NPZAdapter(path, dtype=torch.float32).load()
        assert ds.X_colloc.dtype == torch.float32


# ---------------------------------------------------------------------------
# PINNacleAdapter
# ---------------------------------------------------------------------------

class TestPINNacleAdapter:
    def test_dat_file(self, tmp_path: Path):
        data = np.column_stack([
            np.linspace(0, 1, 25),
            np.linspace(0, 1, 25) ** 2,
        ])
        path = tmp_path / "poisson_classic.dat"
        np.savetxt(path, data)
        ds = PINNacleAdapter(tmp_path, "poisson_classic").load()
        assert ds.X_colloc.shape == (25, 1)
        assert ds.y_data is not None
        assert ds.y_data.shape == (25, 1)
        assert ds.meta["source"] == "pinnacle"

    def test_npz_file(self, tmp_path: Path):
        X = np.random.randn(20, 2)
        y = np.random.randn(20, 1)
        path = tmp_path / "heat_task.npz"
        np.savez(path, X=X, y=y)
        ds = PINNacleAdapter(tmp_path, "heat_task").load()
        assert ds.X_colloc.shape == (20, 2)

    def test_csv_file(self, tmp_path: Path):
        rows = [(float(i), float(i * 0.1)) for i in range(15)]
        path = tmp_path / "wave.csv"
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["x", "u"])
            w.writerows(rows)
        ds = PINNacleAdapter(tmp_path, "wave").load()
        assert ds.X_colloc.shape == (15, 1)

    def test_task_not_found_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            PINNacleAdapter(tmp_path, "nonexistent_task").load()

    def test_meta_contains_task(self, tmp_path: Path):
        data = np.column_stack([np.linspace(0, 1, 10), np.zeros(10)])
        np.savetxt(tmp_path / "mytask.dat", data)
        ds = PINNacleAdapter(tmp_path, "mytask").load()
        assert ds.meta["task"] == "mytask"

    def test_auto_load_dat_routing(self, tmp_path: Path):
        data = np.column_stack([np.linspace(0, 1, 12), np.zeros(12)])
        path = tmp_path / "direct.dat"
        np.savetxt(path, data)
        ds = auto_load(path)
        assert isinstance(ds, PIELMDataset)
        assert ds.X_colloc.shape[0] == 12

    def test_auto_load_txt_routing(self, tmp_path: Path):
        data = np.column_stack([np.linspace(0, 1, 8), np.ones(8)])
        path = tmp_path / "data.txt"
        np.savetxt(path, data)
        ds = auto_load(path)
        assert isinstance(ds, PIELMDataset)
        assert ds.X_colloc.shape[0] == 8

    def test_dtype_preserved(self, tmp_path: Path):
        data = np.column_stack([np.linspace(0, 1, 5), np.zeros(5)])
        np.savetxt(tmp_path / "task.dat", data)
        ds = PINNacleAdapter(tmp_path, "task", dtype=torch.float32).load()
        assert ds.X_colloc.dtype == torch.float32


# ---------------------------------------------------------------------------
# PDEBenchAdapter  (unit-tests without h5py — skip gracefully)
# ---------------------------------------------------------------------------

class TestPDEBenchAdapter:
    @pytest.fixture(autouse=True)
    def _check_h5py(self):
        pytest.importorskip("h5py", reason="h5py not installed")

    def _make_h5_file(self, tmp_path: Path) -> Path:
        import h5py

        path = tmp_path / "1d_adv.h5"
        x = np.linspace(0, 1, 50)
        u = np.stack([np.sin(k * np.pi * x) for k in range(4)])  # (4, 50)
        with h5py.File(path, "w") as f:
            grp = f.create_group("1D_Advection")
            grp.create_dataset("x", data=x)
            grp.create_dataset("u", data=u)  # (n_samples, space)
        return path

    def test_load_basic(self, tmp_path: Path):
        path = self._make_h5_file(tmp_path)
        ds = PDEBenchAdapter(path, equation="1D_Advection", sample_idx=0).load()
        assert ds.X_colloc.shape == (50, 1)
        assert ds.y_data.shape == (50, 1)
        assert ds.X_colloc.dtype == torch.float64

    def test_sample_idx(self, tmp_path: Path):
        path = self._make_h5_file(tmp_path)
        ds0 = PDEBenchAdapter(path, equation="1D_Advection", sample_idx=0).load()
        ds2 = PDEBenchAdapter(path, equation="1D_Advection", sample_idx=2).load()
        assert not torch.allclose(ds0.y_data, ds2.y_data)

    def test_missing_equation_raises(self, tmp_path: Path):
        path = self._make_h5_file(tmp_path)
        with pytest.raises(KeyError, match="2D_Heat"):
            PDEBenchAdapter(path, equation="2D_Heat").load()

    def test_no_equation_uses_first_group(self, tmp_path: Path):
        path = self._make_h5_file(tmp_path)
        ds = PDEBenchAdapter(path).load()
        assert ds.X_colloc.shape[0] == 50

    def test_auto_load_h5_routing(self, tmp_path: Path):
        import h5py

        path = tmp_path / "task.h5"
        x = np.linspace(0, 1, 20)
        u = np.stack([x] * 3)
        with h5py.File(path, "w") as f:
            grp = f.create_group("eq")
            grp.create_dataset("x", data=x)
            grp.create_dataset("u", data=u)
        ds = auto_load(path)
        assert isinstance(ds, PIELMDataset)

    def test_no_h5py_raises_importerror(self, tmp_path: Path, monkeypatch):
        monkeypatch.setitem(sys.modules, "h5py", None)
        with pytest.raises((ImportError, TypeError)):
            PDEBenchAdapter(tmp_path / "fake.h5", equation="x").load()


# ---------------------------------------------------------------------------
# TorchDatasetAdapter
# ---------------------------------------------------------------------------

class _SimpleTorchDS(torch.utils.data.Dataset):
    def __init__(self, n: int = 10, d: int = 2):
        self.X = torch.randn(n, d, dtype=torch.float64)
        self.y = torch.randn(n, 1, dtype=torch.float64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class TestTorchDatasetAdapter:
    def test_data_role_shapes(self):
        ds_torch = _SimpleTorchDS(15, 3)
        ds = TorchDatasetAdapter(ds_torch, role="data").load()
        assert ds.X_colloc.shape == (15, 3)
        assert ds.y_data.shape == (15, 1)
        assert ds.X_bc is None

    def test_colloc_role(self):
        ds_torch = _SimpleTorchDS(10, 2)
        ds = TorchDatasetAdapter(ds_torch, role="colloc").load()
        assert ds.X_colloc.shape == (10, 2)
        assert ds.y_data is None

    def test_bc_role(self):
        ds_torch = _SimpleTorchDS(8, 2)
        ds = TorchDatasetAdapter(ds_torch, role="bc").load()
        assert ds.X_bc.shape == (8, 2)
        assert ds.y_bc.shape == (8, 1)

    def test_ic_role(self):
        ds_torch = _SimpleTorchDS(6, 1)
        ds = TorchDatasetAdapter(ds_torch, role="ic").load()
        assert ds.X_ic.shape == (6, 1)
        assert ds.y_ic.shape == (6, 1)

    def test_invalid_role_raises(self):
        with pytest.raises(ValueError, match="role"):
            TorchDatasetAdapter(_SimpleTorchDS(), role="invalid")

    def test_dtype(self):
        ds_torch = _SimpleTorchDS()
        ds = TorchDatasetAdapter(ds_torch, dtype=torch.float32).load()
        assert ds.X_colloc.dtype == torch.float32

    def test_values_match_source(self):
        ds_torch = _SimpleTorchDS(5, 2)
        ds = TorchDatasetAdapter(ds_torch, role="data").load()
        for i in range(5):
            xi, yi = ds_torch[i]
            assert torch.allclose(ds.X_colloc[i], xi.to(torch.float64))

    def test_empty_dataset_raises(self):
        class EmptyDS(torch.utils.data.Dataset):
            def __len__(self): return 0
            def __getitem__(self, i): raise IndexError

        with pytest.raises((ValueError, StopIteration, IndexError)):
            TorchDatasetAdapter(EmptyDS()).load()


# ---------------------------------------------------------------------------
# auto_load routing
# ---------------------------------------------------------------------------

class TestAutoLoad:
    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            auto_load(tmp_path / "ghost.csv")

    def test_unsupported_extension_raises(self, tmp_path: Path):
        path = tmp_path / "data.xyz"
        path.write_text("1 2 3\n")
        with pytest.raises(ValueError, match="extension"):
            auto_load(path)

    def test_string_path(self, tmp_path: Path):
        data = np.ones((5, 2))
        path = tmp_path / "str.npz"
        np.savez(path, X_colloc=data)
        ds = auto_load(str(path))
        assert ds.X_colloc.shape == (5, 2)

    def test_dtype_forwarded(self, tmp_path: Path):
        data = np.ones((4, 2))
        path = tmp_path / "dtype.npz"
        np.savez(path, X_colloc=data)
        ds = auto_load(path, dtype=torch.float32)
        assert ds.X_colloc.dtype == torch.float32


# ---------------------------------------------------------------------------
# Device-aware tests (MPS)
# ---------------------------------------------------------------------------

class TestDeviceAware:
    @pytest.mark.mps
    def test_from_arrays_mps(self, mps_device: torch.device):
        # MPS does not support float64; use float32
        X = np.ones((10, 2))
        ds = PIELMDataset.from_arrays(X, dtype=torch.float32, device=mps_device)
        assert ds.X_colloc.device.type == "mps"

    @pytest.mark.mps
    def test_normalizer_transform_on_mps(self, mps_device: torch.device):
        # MPS does not support float64; fit on CPU float32, transform on MPS
        X = torch.randn(20, 3, dtype=torch.float32)
        n = Normalizer("minmax")
        n.fit(X)
        X_mps = X.to(mps_device)
        # Normalizer moves its stats to the input tensor's device
        Xn = n.transform(X_mps)
        assert Xn.device.type == "mps"

    @pytest.mark.mps
    def test_feature_expander_mps(self, mps_device: torch.device):
        X = torch.randn(10, 2, dtype=torch.float32, device=mps_device)
        fe = FeatureExpander(degree=2, trig=True)
        Xout = fe.fit_transform(X)
        assert Xout.device.type == "mps"
        # 2 + 2 + 2 + 2 = 8 cols
        assert Xout.shape == (10, 8)

    @pytest.mark.mps
    def test_npz_adapter_mps(self, mps_device: torch.device, tmp_path: Path):
        # MPS does not support float64; use float32
        X = np.ones((8, 2))
        path = tmp_path / "mps.npz"
        np.savez(path, X_colloc=X)
        ds = NPZAdapter(path, dtype=torch.float32, device=mps_device).load()
        assert ds.X_colloc.device.type == "mps"


