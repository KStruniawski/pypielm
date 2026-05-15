"""Extended coverage tests for the PINNacle data adapter.

Covers:
* PINNacleAdapter.load() on the real poisson_classic.dat file
* _load_plain_table  (whitespace-delimited .dat)
* _load_array_dict   (.npz, .npy, .csv variants)
* _coerce_xy         (pre-split X/y, "y" key, 2-column, feature-only)
* _resolve_data_file (task stem matching)
* auto_load(..., source="pinnacle") integration
* Error paths (file-not-found, unsupported format)

Skips gracefully when the PINNacle data directory is not present.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

# Real PINNacle data
_REPO_ROOT   = Path(__file__).parents[1].parent
_DAT_FILE    = _REPO_ROOT / "Benchmarking/Papers/PINNacle-main/ref/poisson_classic.dat"
_PINNACLE_ROOT = _DAT_FILE.parent if _DAT_FILE.exists() else None

_HAS_DAT = _DAT_FILE.exists()
DTYPE = torch.float64


# ===========================================================================
# Helper: write a temp file and load it
# ===========================================================================

def _write_tmp(tmp_path: Path, filename: str, content: str) -> Path:
    p = tmp_path / filename
    p.write_text(textwrap.dedent(content))
    return p


# ===========================================================================
# _load_plain_table (internal)
# ===========================================================================

class TestLoadPlainTable:
    def test_basic_whitespace_delimited(self, tmp_path):
        from pypielm.data.adapters.pinnacle_adapter import _load_plain_table
        p = _write_tmp(tmp_path, "test.dat", """\
            0.0 0.5 1.0
            0.1 0.6 1.1
            0.2 0.7 1.2
        """)
        data = _load_plain_table(p)
        assert "X" in data
        assert "y" in data
        assert data["X"].shape[1] == 2
        assert data["y"].shape[1] == 1

    def test_single_column(self, tmp_path):
        from pypielm.data.adapters.pinnacle_adapter import _load_plain_table
        p = _write_tmp(tmp_path, "1col.dat", "1.0\n2.0\n3.0\n")
        data = _load_plain_table(p)
        assert "X" in data
        assert data["X"].shape[1] == 1

    def test_comment_lines_ignored(self, tmp_path):
        from pypielm.data.adapters.pinnacle_adapter import _load_plain_table
        p = _write_tmp(tmp_path, "commented.dat", """\
            % This is a comment
            0.0 1.0
            % Another comment
            0.5 2.0
        """)
        data = _load_plain_table(p)
        assert data["X"].shape[0] == 2

    def test_column_keys_present(self, tmp_path):
        from pypielm.data.adapters.pinnacle_adapter import _load_plain_table
        p = _write_tmp(tmp_path, "cols.dat", "1.0 2.0\n3.0 4.0\n")
        data = _load_plain_table(p)
        assert "col0" in data
        assert "col1" in data


# ===========================================================================
# _load_array_dict (internal)
# ===========================================================================

class TestLoadArrayDict:
    def test_npz_round_trip(self, tmp_path):
        from pypielm.data.adapters.pinnacle_adapter import _load_array_dict
        rng = np.random.default_rng(0)
        X = rng.random((50, 2))
        y = rng.random((50, 1))
        p = tmp_path / "data.npz"
        np.savez(p, X=X, y=y)
        data = _load_array_dict(p)
        assert "X" in data
        assert "y" in data
        assert data["X"].shape == (50, 2)
        assert data["y"].shape == (50, 1)

    def test_npy_2d_xy(self, tmp_path):
        from pypielm.data.adapters.pinnacle_adapter import _load_array_dict
        rng = np.random.default_rng(1)
        arr = rng.random((40, 3))  # 2 features + 1 target
        p = tmp_path / "data.npy"
        np.save(p, arr)
        data = _load_array_dict(p)
        assert "X" in data
        assert data["X"].shape[1] == 2

    def test_npy_1d(self, tmp_path):
        from pypielm.data.adapters.pinnacle_adapter import _load_array_dict
        arr = np.array([1.0, 2.0, 3.0])
        p = tmp_path / "vec.npy"
        np.save(p, arr)
        data = _load_array_dict(p)
        assert "X" in data

    def test_csv_with_header(self, tmp_path):
        from pypielm.data.adapters.pinnacle_adapter import _load_array_dict
        p = _write_tmp(tmp_path, "data.csv", """\
            x0,x1,y
            0.1,0.2,0.3
            0.4,0.5,0.6
        """)
        data = _load_array_dict(p)
        assert "x0" in data or "X" in data

    def test_csv_no_header(self, tmp_path):
        from pypielm.data.adapters.pinnacle_adapter import _load_array_dict
        # Write a CSV file that clearly has no header (all numeric)
        # genfromtxt may produce named columns using the first row as names
        # OR produce "X"/"y" split — either is acceptable
        p = _write_tmp(tmp_path, "noheader.csv", "0.1,0.2\n0.3,0.4\n0.5,0.6\n")
        data = _load_array_dict(p)
        # The result must be a non-empty dict with at least one array
        assert len(data) >= 1
        # All values must be numpy arrays
        import numpy as np
        for v in data.values():
            assert isinstance(v, np.ndarray)

    def test_dat_extension(self, tmp_path):
        from pypielm.data.adapters.pinnacle_adapter import _load_array_dict
        p = _write_tmp(tmp_path, "data.dat", "0.1 0.2 0.3\n0.4 0.5 0.6\n")
        data = _load_array_dict(p)
        assert "X" in data

    def test_txt_extension(self, tmp_path):
        from pypielm.data.adapters.pinnacle_adapter import _load_array_dict
        p = _write_tmp(tmp_path, "data.txt", "1.0 2.0\n3.0 4.0\n")
        data = _load_array_dict(p)
        assert "X" in data

    def test_unsupported_extension_raises(self, tmp_path):
        from pypielm.data.adapters.pinnacle_adapter import _load_array_dict
        p = tmp_path / "data.h5"
        p.write_bytes(b"\x00\x01\x02")
        with pytest.raises(ValueError, match="Unsupported"):
            _load_array_dict(p)


# ===========================================================================
# _coerce_xy (internal)
# ===========================================================================

class TestCoerceXY:
    def test_prebuilt_XY_keys(self):
        from pypielm.data.adapters.pinnacle_adapter import _coerce_xy
        rng = np.random.default_rng(0)
        data = {"X": rng.random((30, 2)), "y": rng.random(30)}
        X, y = _coerce_xy(data)
        assert X.shape == (30, 2)
        assert y.shape == (30, 1)

    def test_prebuilt_X_no_y(self):
        from pypielm.data.adapters.pinnacle_adapter import _coerce_xy
        data = {"X": np.arange(10).reshape(10, 1)}
        X, y = _coerce_xy(data)
        assert X.shape == (10, 1)
        assert y is None

    def test_explicit_y_key(self):
        from pypielm.data.adapters.pinnacle_adapter import _coerce_xy
        rng = np.random.default_rng(0)
        data = {"x0": rng.random(20), "x1": rng.random(20), "y": rng.random(20)}
        X, y = _coerce_xy(data)
        assert X.shape[1] == 2
        assert y.shape == (20, 1)

    def test_two_columns_no_y(self):
        from pypielm.data.adapters.pinnacle_adapter import _coerce_xy
        data = {"a": np.arange(10.0), "b": np.arange(10.0, 20.0)}
        X, y = _coerce_xy(data)
        assert X.shape[1] == 1  # first column
        assert y is not None

    def test_all_feature_columns(self):
        from pypielm.data.adapters.pinnacle_adapter import _coerce_xy
        rng = np.random.default_rng(0)
        data = {"a": rng.random(10), "b": rng.random(10), "c": rng.random(10)}
        X, y = _coerce_xy(data)
        assert X.shape[1] == 3
        assert y is None


# ===========================================================================
# _resolve_data_file (internal)
# ===========================================================================

class TestResolveDataFile:
    def test_absolute_existing_path(self, tmp_path):
        from pypielm.data.adapters.pinnacle_adapter import _resolve_data_file
        p = tmp_path / "task.dat"
        p.touch()
        result = _resolve_data_file(tmp_path, str(p))
        assert result == p

    def test_direct_child(self, tmp_path):
        from pypielm.data.adapters.pinnacle_adapter import _resolve_data_file
        p = tmp_path / "mydata.dat"
        p.touch()
        result = _resolve_data_file(tmp_path, "mydata.dat")
        assert result == p

    def test_stem_matching(self, tmp_path):
        from pypielm.data.adapters.pinnacle_adapter import _resolve_data_file
        sub = tmp_path / "subdir"
        sub.mkdir()
        p = sub / "poisson_classic.dat"
        p.touch()
        result = _resolve_data_file(tmp_path, "poisson_classic")
        assert result == p

    def test_file_not_found_raises(self, tmp_path):
        from pypielm.data.adapters.pinnacle_adapter import _resolve_data_file
        with pytest.raises(FileNotFoundError):
            _resolve_data_file(tmp_path, "nonexistent_task")


# ===========================================================================
# PINNacleAdapter (full class)
# ===========================================================================

class TestPINNacleAdapterSynthetic:
    def test_load_from_npz(self, tmp_path):
        from pypielm.data.adapters.pinnacle_adapter import PINNacleAdapter
        rng = np.random.default_rng(42)
        X = rng.random((100, 2))
        y = rng.random((100, 1))
        p = tmp_path / "testdata.npz"
        np.savez(p, X=X, y=y)
        adapter = PINNacleAdapter(root=tmp_path, task="testdata.npz")
        ds = adapter.load()
        assert ds.X_colloc.shape[0] == 100
        assert ds.y_data is not None
        assert ds.y_data.shape == (100, 1)

    def test_load_from_dat(self, tmp_path):
        from pypielm.data.adapters.pinnacle_adapter import PINNacleAdapter
        p = tmp_path / "simple.dat"
        p.write_text("0.1 0.2 0.3\n0.4 0.5 0.6\n0.7 0.8 0.9\n")
        adapter = PINNacleAdapter(root=tmp_path, task="simple.dat")
        ds = adapter.load()
        assert ds.X_colloc.shape[0] == 3

    def test_dataset_dtype(self, tmp_path):
        from pypielm.data.adapters.pinnacle_adapter import PINNacleAdapter
        p = tmp_path / "data.dat"
        p.write_text("0.1 0.2\n0.3 0.4\n")
        adapter = PINNacleAdapter(root=tmp_path, task="data.dat", dtype=torch.float64)
        ds = adapter.load()
        assert ds.X_colloc.dtype == torch.float64

    def test_meta_populated(self, tmp_path):
        from pypielm.data.adapters.pinnacle_adapter import PINNacleAdapter
        p = tmp_path / "info.dat"
        p.write_text("0.1 0.2\n0.3 0.4\n")
        adapter = PINNacleAdapter(root=tmp_path, task="info.dat")
        ds = adapter.load()
        assert ds.meta is not None
        assert ds.meta.get("source") == "pinnacle"


@pytest.mark.skipif(not _HAS_DAT, reason="PINNacle data not available")
class TestPINNacleAdapterRealData:
    def test_load_poisson_classic(self):
        from pypielm.data.adapters.pinnacle_adapter import PINNacleAdapter
        adapter = PINNacleAdapter(
            root=_DAT_FILE.parent,
            task=_DAT_FILE.name,
        )
        ds = adapter.load()
        assert ds.X_colloc.shape[0] > 0
        assert ds.X_colloc.shape[1] >= 1

    def test_shape_matches_reference(self):
        """poisson_classic.dat should have ~1255 total rows (benchmark split)."""
        from pypielm.data.adapters.pinnacle_adapter import PINNacleAdapter
        adapter = PINNacleAdapter(root=_DAT_FILE.parent, task=_DAT_FILE.name)
        ds = adapter.load()
        n = ds.X_colloc.shape[0]
        # Reference benchmark uses train=879, val=125, test=251 → 1255 total
        assert 1000 <= n <= 2000, f"Unexpected row count: {n}"

    def test_auto_load_pinnacle(self):
        from pypielm.data import auto_load
        # auto_load dispatches on .dat extension — no explicit source kwarg needed
        ds = auto_load(_DAT_FILE)
        assert ds.X_colloc.shape[0] > 0

    def test_dtype_float64(self):
        from pypielm.data.adapters.pinnacle_adapter import PINNacleAdapter
        adapter = PINNacleAdapter(root=_DAT_FILE.parent, task=_DAT_FILE.name,
                                   dtype=torch.float64)
        ds = adapter.load()
        assert ds.X_colloc.dtype == torch.float64
