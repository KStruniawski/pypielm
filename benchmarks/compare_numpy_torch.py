"""Cross-framework comparison: PyPIELM (PyTorch) vs reference NumPy ELM.

This script runs the **same** Poisson-1D task through:

1. **PyPIELM** — ``VanillaPIELM`` using ``ridge_solve`` on the PyTorch backend
2. **NumPy ELM** — a pure-NumPy reference implementation (no PyTorch dependency)

It measures:
* relative L² error
* fit wall-clock time
* CPU/memory usage (via tracemalloc)

Results are saved to ``benchmarks/results/<timestamp>_compare_numpy_torch.json``.

Usage::

    cd PyPIELM
    python benchmarks/compare_numpy_torch.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))

from perf_profile import TASK_MAP, _rel_l2, _save, RESULTS_DIR, get_results_dir, _platform_name
from device_utils import (
    resolve_device,
    dtype_for_device,
    device_info,
    to_device_dataset,
    to_device_tensor,
)

SEEDS = [42, 0, 1, 2, 3]
HIDDEN_DIMS = [50, 100, 200, 500]
N_DATA = 300
DEFAULT_TASK = "poisson_1d"


# ---------------------------------------------------------------------------
# Pure NumPy ELM reference
# ---------------------------------------------------------------------------

class NumpyELM:
    """Minimal ELM in NumPy: tanh random features + ridge regression.

    Used as a cross-framework baseline.  No PDE physics — purely data-driven.
    """

    def __init__(self, hidden_dim: int = 200, ridge_lambda: float = 1e-8,
                 seed: int = 42) -> None:
        self.hidden_dim = hidden_dim
        self.ridge_lambda = ridge_lambda
        self.seed = seed
        self._W: np.ndarray | None = None
        self._b: np.ndarray | None = None
        self._beta: np.ndarray | None = None

    def _build_features(self, X: np.ndarray) -> np.ndarray:
        """Compute tanh random-feature matrix (N, H)."""
        return np.tanh(X @ self._W + self._b)  # (N, H)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NumpyELM":
        rng = np.random.default_rng(self.seed)
        d = X.shape[1]
        # Random weights: W ~ N(0,1), b ~ U(-π, π)
        self._W = rng.standard_normal((d, self.hidden_dim))
        self._b = rng.uniform(-np.pi, np.pi, (1, self.hidden_dim))
        H = self._build_features(X)                     # (N, H)
        HtH = H.T @ H                                   # (H, H)
        Hty = H.T @ y.reshape(-1, 1)                   # (H, 1)
        A = HtH + self.ridge_lambda * np.eye(self.hidden_dim)
        self._beta = np.linalg.solve(A, Hty)           # (H, 1)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        H = self._build_features(X)
        return (H @ self._beta).squeeze()


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_comparison(
    task_name: str = DEFAULT_TASK,
    dims: list[int] = HIDDEN_DIMS,
    seeds: list[int] = SEEDS,
    n_data: int = N_DATA,
    save: bool = True,
    device: str = "cpu",
    results_dir=None,
) -> dict[str, Any]:
    """Compare PyPIELM vs NumPy ELM.

    Returns::

        results[hidden_dim] = {
            "pypielm": {"rel_l2_mean", "rel_l2_std", "fit_time_mean_s", "peak_ram_mb_mean"},
            "numpy_elm": { ... },
            "speedup_fit": float,   # numpy_time / pypielm_time
        }
    """
    import torch
    from pypielm.models.vanilla import VanillaPIELM

    dev = resolve_device(device)
    dtype = dtype_for_device(dev)
    task = TASK_MAP[task_name]
    results: dict[str, Any] = {
        "task": task_name, "dims": dims, "seeds": seeds,
        "device": device_info(dev),
    }

    for hd in dims:
        print(f"\n[compare] hidden_dim={hd}")
        results[hd] = {}

        py_rl2, py_ft, py_ram = [], [], []
        np_rl2, np_ft, np_ram = [], [], []

        for seed in seeds:
            dataset = task.build_dataset(n_data=n_data, seed=seed)
            X_train = dataset.X_data.numpy()
            y_train = dataset.y_data.numpy()
            X_test = task.get_test_points()
            y_test = task.u_exact(X_test)

            # --- PyPIELM ---
            try:
                model = VanillaPIELM(hidden_dim=hd, seed=seed, dtype=dtype,
                                     device=str(dev))
                ds_dev = to_device_dataset(dataset, dev)
                tracemalloc.start()
                t0 = time.perf_counter()
                model.fit(ds_dev)
                py_ft.append(time.perf_counter() - t0)
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                py_ram.append(peak / 1e6)
                X_test_t = to_device_tensor(X_test, dev)
                u_pred = model.predict(X_test_t).cpu().detach().numpy().squeeze()
                py_rl2.append(_rel_l2(y_test, u_pred))
                print(f"  PyPIELM   seed={seed}: "
                      f"rel_l2={py_rl2[-1]:.3e}  "
                      f"fit={py_ft[-1]*1000:.1f}ms  "
                      f"ram={py_ram[-1]:.1f}MB")
            except Exception as exc:
                tracemalloc.stop()
                py_rl2.append(float("nan")); py_ft.append(float("nan"))
                py_ram.append(float("nan"))
                print(f"  PyPIELM   seed={seed}: ERROR {exc}")

            # --- NumPy ELM ---
            try:
                np_elm = NumpyELM(hidden_dim=hd, seed=seed)
                tracemalloc.start()
                t0 = time.perf_counter()
                np_elm.fit(X_train, y_train)
                np_ft.append(time.perf_counter() - t0)
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                np_ram.append(peak / 1e6)
                u_pred_np = np_elm.predict(X_test)
                np_rl2.append(_rel_l2(y_test, u_pred_np))
                print(f"  NumPyELM  seed={seed}: "
                      f"rel_l2={np_rl2[-1]:.3e}  "
                      f"fit={np_ft[-1]*1000:.1f}ms  "
                      f"ram={np_ram[-1]:.1f}MB")
            except Exception as exc:
                tracemalloc.stop()
                np_rl2.append(float("nan")); np_ft.append(float("nan"))
                np_ram.append(float("nan"))
                print(f"  NumPyELM  seed={seed}: ERROR {exc}")

        py_ft_mean = float(np.nanmean(py_ft))
        np_ft_mean = float(np.nanmean(np_ft))
        speedup = np_ft_mean / (py_ft_mean + 1e-12)

        results[hd] = {
            "pypielm": {
                "rel_l2_mean": float(np.nanmean(py_rl2)),
                "rel_l2_std":  float(np.nanstd(py_rl2)),
                "fit_time_mean_s": py_ft_mean,
                "peak_ram_mb_mean": float(np.nanmean(py_ram)),
            },
            "numpy_elm": {
                "rel_l2_mean": float(np.nanmean(np_rl2)),
                "rel_l2_std":  float(np.nanstd(np_rl2)),
                "fit_time_mean_s": np_ft_mean,
                "peak_ram_mb_mean": float(float(np.nanmean(np_ram))),
            },
            "speedup_fit_numpy_over_pypielm": speedup,
        }
        print(f"  Speedup (NumPy/PyPIELM fit time ratio): {speedup:.2f}x")

    if save:
        _save(results, "compare_numpy_torch", results_dir)
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NumPy vs PyPIELM comparison")
    parser.add_argument("--task", default=DEFAULT_TASK, choices=list(TASK_MAP))
    parser.add_argument("--dims", nargs="+", type=int, default=HIDDEN_DIMS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--n-data", type=int, default=N_DATA)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Compute device for PyPIELM: cpu | mps | cuda | cuda:N  "
             "(NumPy ELM always runs on CPU)",
    )
    parser.add_argument(
        "--platform", type=str, default=None,
        help="Results sub-folder tag (default: derived from --device).",
    )
    args = parser.parse_args()

    dev = resolve_device(args.device)
    platform = args.platform or _platform_name(dev)
    rdir = get_results_dir(platform) if not args.no_save else None

    run_comparison(
        task_name=args.task,
        dims=args.dims,
        seeds=args.seeds,
        n_data=args.n_data,
        save=not args.no_save,
        device=args.device,
        results_dir=rdir,
    )
    print("\nDone.")
