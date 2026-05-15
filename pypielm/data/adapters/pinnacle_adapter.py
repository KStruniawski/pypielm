"""PINNacle dataset adapter.

PINNacle (https://github.com/lu-group/pinn-benchmark) distributes datasets
as HDF5 / npz archives with a specific directory structure.  This adapter
ports the logic from the reference benchmark_framework implementation to
operate directly on the raw files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pypielm.data.dataset import PIELMDataset

_SUPPORTED_EXT = {".npz", ".npy", ".csv", ".json", ".dat", ".txt"}


def _resolve_data_file(root: Path, task: str) -> Path:
    """Find the data file for *task* under *root*."""
    task_path = Path(task)
    if task_path.is_absolute() and task_path.exists():
        return task_path
    candidate = root / task
    if candidate.exists():
        return candidate
    matches = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in _SUPPORTED_EXT and task in p.stem
    ]
    if not matches:
        raise FileNotFoundError(
            f"Could not resolve data file for task '{task}' under '{root}'."
        )
    return sorted(matches)[0]


def _load_plain_table(path: Path) -> dict[str, np.ndarray]:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            arr = np.loadtxt(fh, comments="%")
    except ValueError as exc:
        if "inconsistent" in str(exc).lower() or "columns" in str(exc).lower():
            rows: list[list[float]] = []
            max_cols = 0
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("%"):
                        continue
                    vals = [float(v) for v in line.split()]
                    max_cols = max(max_cols, len(vals))
                    rows.append(vals)
            arr = np.full((len(rows), max_cols), np.nan)
            for i, row in enumerate(rows):
                arr[i, : len(row)] = row
            arr = arr[~np.any(np.isnan(arr), axis=1)]
        else:
            raise
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    out: dict[str, np.ndarray] = {f"col{i}": arr[:, i] for i in range(arr.shape[1])}
    if arr.shape[1] > 1:
        out["X"] = arr[:, :-1]
        out["y"] = arr[:, -1:]
    else:
        out["X"] = arr
    return out


def _load_array_dict(path: Path) -> dict[str, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as f:
            return {k: np.asarray(f[k]) for k in f.files}
    if suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
        if arr.ndim == 1:
            arr = arr[:, None]
        return {"X": arr[:, :-1], "y": arr[:, -1:]} if arr.shape[1] > 1 else {"X": arr}
    if suffix == ".csv":
        raw = np.genfromtxt(path, delimiter=",", names=True)
        if raw.dtype.names:
            return {n: np.asarray(raw[n]) for n in raw.dtype.names}  # type: ignore[call-overload]
        # No header — fall through to plain table
        raw2 = np.genfromtxt(path, delimiter=",", dtype=float)
        if raw2.ndim == 1:
            raw2 = raw2.reshape(-1, 1)
        return {"X": raw2[:, :-1], "y": raw2[:, -1:]} if raw2.shape[1] > 1 else {"X": raw2}
    if suffix in {".dat", ".txt"}:
        return _load_plain_table(path)
    raise ValueError(f"Unsupported file format: {path.suffix}")


def _coerce_xy(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray | None]:
    if "X" in data:
        X = np.asarray(data["X"])
        if X.ndim == 1:
            X = X[:, None]
        y: np.ndarray | None = None
        if "y" in data:
            y = np.asarray(data["y"])
            if y.ndim == 1:
                y = y[:, None]
        return X, y

    # No pre-built "X" array: collect scalar-per-row columns
    keys = list(data.keys())

    # Prefer explicit "y" key as target
    if "y" in data:
        feature_keys = [k for k in keys if k != "y"]
        if not feature_keys:
            raise ValueError("No feature arrays found in data.")
        columns = [np.asarray(data[k]).reshape(-1, 1) for k in feature_keys]
        X = np.concatenate(columns, axis=1) if len(columns) > 1 else columns[0]
        y_arr = np.asarray(data["y"])
        if y_arr.ndim == 1:
            y_arr = y_arr[:, None]
        return X, y_arr

    # Exactly 2 columns with arbitrary names → first is feature, last is target
    if len(keys) == 2:
        X = np.asarray(data[keys[0]])
        if X.ndim == 1:
            X = X[:, None]
        y_col = np.asarray(data[keys[1]])
        if y_col.ndim == 1:
            y_col = y_col[:, None]
        return X, y_col

    # More than 2 unnamed columns — treat all as features, no target
    columns = [np.asarray(data[k]).reshape(-1, 1) for k in keys]
    X = np.concatenate(columns, axis=1)
    return X, None


class PINNacleAdapter:
    """Load a task dataset in PINNacle format.

    Args:
        root: Root directory of the PINNacle data folder.
        task: Task identifier string, e.g. ``'poisson_classic'``, or a path
            to the data file itself.
        dtype: Tensor dtype.
        device: Target device.
    """

    def __init__(
        self,
        root: str | Path,
        task: str,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> None:
        self.root = Path(root)
        self.task = task
        self.dtype = dtype
        self.device = device

    def load(self) -> PIELMDataset:
        """Load the task dataset and return a :class:`~pypielm.data.dataset.PIELMDataset`."""
        data_file = _resolve_data_file(self.root, self.task)
        data = _load_array_dict(data_file)
        X, y = _coerce_xy(data)

        def _t(arr: np.ndarray) -> torch.Tensor:
            t = torch.tensor(arr, dtype=self.dtype, device=self.device)
            if t.ndim == 1:
                t = t.unsqueeze(1)
            return t

        return PIELMDataset(
            X_colloc=_t(X),
            y_data=_t(y) if y is not None else None,
            meta={"source": "pinnacle", "task": self.task, "file": str(data_file)},
        )
