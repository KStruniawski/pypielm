"""Performance benchmarks: timing, memory, and accuracy profiles.

Run all benchmarks::

    cd PyPIELM
    python benchmarks/perf_profile.py

Run on MPS (Apple Silicon)::

    python benchmarks/perf_profile.py --device mps

Run on CUDA::

    python benchmarks/perf_profile.py --device cuda

Results are written to ``benchmarks/results/<timestamp>.json``.
"""

from __future__ import annotations

import os
import sys

# Must be set before torch is imported so CuBLAS allows deterministic ops on CUDA >= 10.2.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# Ensure Unicode characters (arrows, Greek letters) print safely on Windows.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import math
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent))
from device_utils import (  # noqa: E402
    resolve_device,
    dtype_for_device,
    device_info,
    to_device_dataset,
    to_device_tensor,
)

# ---------------------------------------------------------------------------
# Analytic PDE tasks
# ---------------------------------------------------------------------------
# Each task exposes:
#   .name          str
#   .input_dim     int
#   .build_dataset(n_data, n_colloc, n_bc, seed) -> PIELMDataset
#   .u_exact(X)    -> np.ndarray  (N,)


class _Poisson1D:
    """1D Poisson: u'' = -2, u = x(1-x), x ∈ [0,1]."""

    name = "poisson_1d"
    input_dim = 1

    def build_dataset(
        self,
        n_data: int = 200,
        n_colloc: int = 100,
        n_bc: int = 0,
        seed: int = 42,
    ):
        from pypielm.data.dataset import PIELMDataset

        rng = np.random.default_rng(seed)
        X = rng.uniform(0, 1, (n_data, 1)).astype(np.float64)
        y = (X * (1 - X)).squeeze()

        # Boundary points
        X_bc = np.array([[0.0], [1.0]])
        y_bc = np.array([0.0, 0.0])

        # Interior collocation
        X_c = rng.uniform(0, 1, (n_colloc, 1)).astype(np.float64)

        # Test points (separate from training)
        X_test = np.linspace(0, 1, 80).reshape(-1, 1)
        self._X_test = X_test

        return PIELMDataset.from_arrays(
            X_c,
            X_data=X, y_data=y,
            X_bc=X_bc, y_bc=y_bc,
        )

    def u_exact(self, X: np.ndarray) -> np.ndarray:
        x = X[:, 0] if X.ndim == 2 else X
        return x * (1 - x)

    def get_test_points(self) -> np.ndarray:
        return getattr(self, "_X_test",
                       np.linspace(0, 1, 80).reshape(-1, 1))

    def pde_operator(self, fm, X):
        """ΔH = Σ d²H/dx², RHS = -2."""
        from pypielm.core.solver import WeightedLinearSystem
        lap_H = fm.d2(X, 0)
        rhs = -2.0 * torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)
        return WeightedLinearSystem(H=lap_H, y=rhs, weight=1.0)


class _Poisson2D:
    """2D Poisson: Δu = -2π²sin(πx)sin(πy), u = sin(πx)sin(πy), [0,1]²."""

    name = "poisson_2d"
    input_dim = 2

    def build_dataset(
        self,
        n_data: int = 400,
        n_colloc: int = 200,
        n_bc: int = 0,
        seed: int = 42,
    ):
        from pypielm.data.dataset import PIELMDataset

        rng = np.random.default_rng(seed)
        X = rng.uniform(0, 1, (n_data, 2)).astype(np.float64)
        y = np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])

        # BCs: u=0 on all edges
        n_edge = 20
        t = np.linspace(0, 1, n_edge)
        X_bc = np.vstack([
            np.column_stack([np.zeros(n_edge), t]),
            np.column_stack([np.ones(n_edge), t]),
            np.column_stack([t, np.zeros(n_edge)]),
            np.column_stack([t, np.ones(n_edge)]),
        ])
        y_bc = np.zeros(len(X_bc))

        X_c = rng.uniform(0, 1, (n_colloc, 2)).astype(np.float64)

        # Test grid
        t1d = np.linspace(0.05, 0.95, 10)
        gx, gy = np.meshgrid(t1d, t1d)
        self._X_test = np.column_stack([gx.ravel(), gy.ravel()])

        return PIELMDataset.from_arrays(
            X_c,
            X_data=X, y_data=y,
            X_bc=X_bc, y_bc=y_bc,
        )

    def u_exact(self, X: np.ndarray) -> np.ndarray:
        return np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])

    def get_test_points(self) -> np.ndarray:
        if not hasattr(self, "_X_test"):
            t1d = np.linspace(0.05, 0.95, 10)
            gx, gy = np.meshgrid(t1d, t1d)
            self._X_test = np.column_stack([gx.ravel(), gy.ravel()])
        return self._X_test

    def pde_operator(self, fm, X):
        """ΔH = d²H/dx² + d²H/dy², RHS = -2π²sin(πx)sin(πy)."""
        from pypielm.core.solver import WeightedLinearSystem
        lap_H = fm.d2(X, 0) + fm.d2(X, 1)
        rhs_vals = (-2 * math.pi ** 2
                    * torch.sin(math.pi * X[:, 0:1])
                    * torch.sin(math.pi * X[:, 1:2]))
        return WeightedLinearSystem(H=lap_H, y=rhs_vals, weight=1.0)


class _Heat1D:
    """1D Heat: u_t = u_xx, u = exp(-π²t)sin(πx), (x,t) ∈ [0,1]×[0,1]."""

    name = "heat_1d"
    input_dim = 2  # (x, t)

    def build_dataset(
        self,
        n_data: int = 400,
        n_colloc: int = 200,
        n_bc: int = 0,
        seed: int = 42,
    ):
        from pypielm.data.dataset import PIELMDataset

        rng = np.random.default_rng(seed)
        X = rng.uniform(0, 1, (n_data, 2)).astype(np.float64)
        y = np.exp(-np.pi ** 2 * X[:, 1]) * np.sin(np.pi * X[:, 0])

        # BCs: u=0 at x=0 and x=1 for all t
        n_bc_pts = 20
        t_bc = np.linspace(0, 1, n_bc_pts)
        X_bc = np.vstack([
            np.column_stack([np.zeros(n_bc_pts), t_bc]),
            np.column_stack([np.ones(n_bc_pts), t_bc]),
        ])
        y_bc = np.zeros(len(X_bc))

        # IC: u = sin(πx) at t=0
        x_ic = np.linspace(0, 1, n_bc_pts).reshape(-1, 1)
        X_ic = np.column_stack([x_ic, np.zeros(n_bc_pts)])
        y_ic = np.sin(np.pi * x_ic.squeeze())

        X_c = rng.uniform(0, 1, (n_colloc, 2)).astype(np.float64)

        # Test points
        x1d = np.linspace(0.05, 0.95, 10)
        t1d = np.linspace(0.05, 0.95, 10)
        gx, gt = np.meshgrid(x1d, t1d)
        self._X_test = np.column_stack([gx.ravel(), gt.ravel()])

        return PIELMDataset.from_arrays(
            X_c,
            X_data=X, y_data=y,
            X_bc=X_bc, y_bc=y_bc,
            X_ic=X_ic, y_ic=y_ic,
        )

    def u_exact(self, X: np.ndarray) -> np.ndarray:
        return np.exp(-np.pi ** 2 * X[:, 1]) * np.sin(np.pi * X[:, 0])

    def get_test_points(self) -> np.ndarray:
        if not hasattr(self, "_X_test"):
            x1d = np.linspace(0.05, 0.95, 10)
            t1d = np.linspace(0.05, 0.95, 10)
            gx, gt = np.meshgrid(x1d, t1d)
            self._X_test = np.column_stack([gx.ravel(), gt.ravel()])
        return self._X_test

    def pde_operator(self, fm, X):
        """u_t - u_xx = 0 → dH/dt - d²H/dx² = 0, RHS = 0."""
        from pypielm.core.solver import WeightedLinearSystem
        # d = (x=0, t=1): dH/dt - d²H/dx²
        H_t = fm.d1(X, 1)   # dH/dt
        H_xx = fm.d2(X, 0)  # d²H/dx²
        H_pde = H_t - H_xx
        rhs = torch.zeros(X.shape[0], 1, dtype=X.dtype, device=X.device)
        return WeightedLinearSystem(H=H_pde, y=rhs, weight=1.0)


class _Burgers1D:
    """Steady 1D advection-diffusion (Burgers-like boundary-layer test).

    PDE: u_x - ε·u_xx = 0 on [0,1], ε = 0.05
    BCs: u(0)=0, u(1)=1
    Exact: u = (exp(x/ε) - 1) / (exp(1/ε) - 1)
    """

    name = "burgers_1d"
    input_dim = 1
    EPS = 0.05  # diffusivity (creates sharp boundary layer near x=1)

    def build_dataset(
        self,
        n_data: int = 200,
        n_colloc: int = 100,
        n_bc: int = 0,
        seed: int = 42,
    ):
        from pypielm.data.dataset import PIELMDataset

        rng = np.random.default_rng(seed)
        X = rng.uniform(0, 1, (n_data, 1)).astype(np.float64)
        y = self._exact(X[:, 0])

        X_bc = np.array([[0.0], [1.0]])
        y_bc = np.array([0.0, 1.0])

        X_c = rng.uniform(0, 1, (n_colloc, 1)).astype(np.float64)
        self._X_test = np.linspace(0, 1, 80).reshape(-1, 1)

        return PIELMDataset.from_arrays(
            X_c,
            X_data=X, y_data=y,
            X_bc=X_bc, y_bc=y_bc,
        )

    def _exact(self, x: np.ndarray) -> np.ndarray:
        eps = self.EPS
        return (np.exp(x / eps) - 1.0) / (np.exp(1.0 / eps) - 1.0)

    def u_exact(self, X: np.ndarray) -> np.ndarray:
        x = X[:, 0] if X.ndim == 2 else X
        return self._exact(x)

    def get_test_points(self) -> np.ndarray:
        return getattr(self, "_X_test",
                       np.linspace(0, 1, 80).reshape(-1, 1))

    def pde_operator(self, fm, X):
        """u_x - ε·u_xx = 0  →  H_x - ε·H_xx = 0,  RHS = 0."""
        from pypielm.core.solver import WeightedLinearSystem
        H_x  = fm.d1(X, 0)
        H_xx = fm.d2(X, 0)
        H_pde = H_x - self.EPS * H_xx
        rhs = torch.zeros(X.shape[0], 1, dtype=X.dtype, device=X.device)
        return WeightedLinearSystem(H=H_pde, y=rhs, weight=1.0)


class _Darcy2D:
    """2D Darcy / Poisson: -Δu = f on [0,1]².

    Exact: u = cos(πx/2)·cos(πy/2)
    RHS:   f = (π²/2)·cos(πx/2)·cos(πy/2)
    BCs: u prescribed on all four edges (Dirichlet).
    """

    name = "darcy_2d"
    input_dim = 2

    def build_dataset(
        self,
        n_data: int = 400,
        n_colloc: int = 200,
        n_bc: int = 0,
        seed: int = 42,
    ):
        from pypielm.data.dataset import PIELMDataset

        rng = np.random.default_rng(seed)
        X = rng.uniform(0, 1, (n_data, 2)).astype(np.float64)
        y = self._exact(X)

        n_edge = 20
        t = np.linspace(0, 1, n_edge)
        X_bc = np.vstack([
            np.column_stack([np.zeros(n_edge), t]),   # x=0
            np.column_stack([np.ones(n_edge), t]),    # x=1 (u=0)
            np.column_stack([t, np.zeros(n_edge)]),   # y=0
            np.column_stack([t, np.ones(n_edge)]),    # y=1 (u=0)
        ])
        y_bc = self._exact(X_bc)

        X_c = rng.uniform(0, 1, (n_colloc, 2)).astype(np.float64)

        t1d = np.linspace(0.05, 0.95, 10)
        gx, gy = np.meshgrid(t1d, t1d)
        self._X_test = np.column_stack([gx.ravel(), gy.ravel()])

        return PIELMDataset.from_arrays(
            X_c,
            X_data=X, y_data=y,
            X_bc=X_bc, y_bc=y_bc,
        )

    def _exact(self, X: np.ndarray) -> np.ndarray:
        return np.cos(math.pi * X[:, 0] / 2) * np.cos(math.pi * X[:, 1] / 2)

    def u_exact(self, X: np.ndarray) -> np.ndarray:
        return self._exact(X)

    def get_test_points(self) -> np.ndarray:
        if not hasattr(self, "_X_test"):
            t1d = np.linspace(0.05, 0.95, 10)
            gx, gy = np.meshgrid(t1d, t1d)
            self._X_test = np.column_stack([gx.ravel(), gy.ravel()])
        return self._X_test

    def pde_operator(self, fm, X):
        """-Δu = (π²/2)·cos(πx/2)·cos(πy/2)."""
        from pypielm.core.solver import WeightedLinearSystem
        lap_H = fm.d2(X, 0) + fm.d2(X, 1)
        rhs_vals = (
            (math.pi ** 2 / 2.0)
            * torch.cos(math.pi * X[:, 0:1] / 2.0)
            * torch.cos(math.pi * X[:, 1:2] / 2.0)
        ).to(X.dtype)
        return WeightedLinearSystem(H=-lap_H, y=rhs_vals, weight=1.0)


TASKS = [_Poisson1D(), _Poisson2D(), _Heat1D(), _Burgers1D(), _Darcy2D()]
TASK_MAP = {t.name: t for t in TASKS}

# ---------------------------------------------------------------------------
# Model constructors for each benchmark variant
# ---------------------------------------------------------------------------

MODEL_NAMES = [
    # --- Vanilla / Core ELM ---
    "vanilla_pielm",
    "core_pielm",
    "bayesian_pielm",
    "gff_pielm",
    "curriculum_pielm",
    # --- Constrained ELM ---
    "nullspace_pielm",
    "eig_pielm",
    "lseelm",
    "stefan_pielm",
    # --- CorePIELM thin wrappers (_make_core_variant) ---
    "normal_equation_elm",
    "parameter_retention_elm",
    "piecewise_elm",
    "delm",
    "fpielm",
    "sgepielm",
    "rinn",
    "rann_pielm",
    "xpielm",
    "pielm_rvds",
    "tspielm",
    "kapielm",
    "soft_partition_kapielm",
    # --- Domain decomposition ---
    "dpielm",
    "locelm",
    "ddelm_coarse",
    # --- Gradient-based PINNs ---
    "vanilla_pinn",
    "adaptive_pinn",
    "fourier_pinn",
    "muon_pinn",
    "residual_adaptive_pinn",
]

# PINN models use different constructor kwargs than ELM-based models.
_VANILLA_PINN_NAMES = frozenset({
    "vanilla_pinn", "adaptive_pinn", "fourier_pinn", "muon_pinn",
})
_RESNET_PINN_NAMES = frozenset({"residual_adaptive_pinn"})
_PINN_NAMES = _VANILLA_PINN_NAMES | _RESNET_PINN_NAMES


def _build_model(
    name: str,
    input_dim: int,
    hidden_dim: int = 200,
    seed: int = 42,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    pinn_epochs: int = 500,
):
    """Instantiate a registered model with appropriate defaults."""
    import pypielm.models  # ensure all models are registered  # noqa: F401
    from pypielm.models.registry import get_model

    if dtype is None:
        dtype = dtype_for_device(device)
    if name in _VANILLA_PINN_NAMES:
        # VanillaPINN / AdaptivePINN / FourierPINN / MuonPINN use layer_dims
        common = dict(
            layer_dims=[hidden_dim, hidden_dim, hidden_dim],
            seed=seed, dtype=dtype, device=str(device),
            max_epochs=pinn_epochs,
        )
    elif name in _RESNET_PINN_NAMES:
        # ResidualAdaptivePINN uses `width` instead of layer_dims
        common = dict(
            width=hidden_dim,
            seed=seed, dtype=dtype, device=str(device),
            max_epochs=pinn_epochs,
        )
    else:
        common = dict(hidden_dim=hidden_dim, seed=seed, dtype=dtype,
                      device=str(device))
    return get_model(name, **common)


def _fit_model(model, task, dataset, pde=True):
    """Fit model, passing pde_operator when available.

    All models accept ``pde_operator`` either as an explicit keyword argument
    or via ``**kwargs``, so we always pass it.  Data-driven models silently
    ignore it; physics-informed models use it.
    """
    model.fit(dataset, pde_operator=task.pde_operator if pde else None)
    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent / "results"


def _platform_name(device: str | torch.device) -> str:
    """Derive a short platform tag from a device string.

    Examples: ``"cpu"`` → ``"cpu"``, ``"mps"`` → ``"mps"``,
    ``"cuda"`` / ``"cuda:0"`` → ``"cuda"``.
    """
    s = str(device).lower().split(":")[0]  # strip ordinal (cuda:0 → cuda)
    return s


def get_results_dir(platform: str | None = None) -> Path:
    """Return the platform-specific results directory.

    ``benchmarks/results/<platform>/``  (e.g. ``results/mps/``)

    Args:
        platform: Tag string such as ``"mps"``, ``"cpu"``, ``"cuda"``.
            When *None* results go to the legacy flat ``results/`` directory
            (backwards-compatible behaviour).
    """
    if platform is None:
        return RESULTS_DIR
    return RESULTS_DIR / platform


def _timestamp() -> str:
    # Microsecond precision prevents collisions when multiple saves occur
    # within the same second (e.g. accuracy + timing + memory in one run).
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _save(data: dict, name: str, results_dir: Path | None = None) -> Path:
    out = results_dir if results_dir is not None else RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{_timestamp()}_{name}.json"
    path.write_text(json.dumps(data, indent=2))
    print(f"  Saved -> {path}")
    return path


def _rel_l2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    num = np.linalg.norm(y_true - y_pred)
    den = np.linalg.norm(y_true) + 1e-12
    return float(num / den)


# ---------------------------------------------------------------------------
# TimingBenchmark
# ---------------------------------------------------------------------------

class TimingBenchmark:
    """Measure wall-clock fit and predict times across model x task pairs."""

    def __init__(
        self,
        model_names: list[str] = MODEL_NAMES,
        task_names: list[str] | None = None,
        hidden_dim: int = 200,
        seeds: list[int] | None = None,
        n_data: int = 300,
        save: bool = True,
        device: str | torch.device = "cpu",
        results_dir: Path | None = None,
        pinn_epochs: int = 500,
    ) -> None:
        self.model_names = model_names
        self.task_names = task_names or [t.name for t in TASKS]
        self.hidden_dim = hidden_dim
        self.seeds = seeds or [42, 0, 1]
        self.n_data = n_data
        self.save = save
        self.device = resolve_device(str(device))
        self.dtype = dtype_for_device(self.device)
        self.results_dir = results_dir
        self.pinn_epochs = pinn_epochs

    def run(self) -> dict[str, Any]:
        results: dict[str, Any] = {"device": device_info(self.device)}
        for task_name in self.task_names:
            task = TASK_MAP[task_name]
            print(f"[Timing] Task: {task_name}  device={self.device}")
            results[task_name] = {}
            dataset = to_device_dataset(
                task.build_dataset(n_data=self.n_data, seed=42), self.device
            )
            for mname in self.model_names:
                fit_times, pred_times = [], []
                print(f"  Model: {mname}", end="", flush=True)
                for seed in self.seeds:
                    try:
                        model = _build_model(mname, task.input_dim,
                                             self.hidden_dim, seed,
                                             device=self.device, dtype=self.dtype,
                                             pinn_epochs=self.pinn_epochs)
                        t0 = time.perf_counter()
                        _fit_model(model, task, dataset)
                        fit_times.append(time.perf_counter() - t0)
                        X_test_t = to_device_tensor(task.get_test_points(), self.device)
                        t1 = time.perf_counter()
                        model.predict(X_test_t)
                        pred_times.append(time.perf_counter() - t1)
                        print(".", end="", flush=True)
                    except Exception as exc:
                        print(f"E({exc.__class__.__name__})", end="", flush=True)
                        fit_times.append(float("nan"))
                        pred_times.append(float("nan"))
                print()
                results[task_name][mname] = {
                    "fit_time_mean_s": float(np.nanmean(fit_times)),
                    "fit_time_std_s":  float(np.nanstd(fit_times)),
                    "pred_time_mean_s": float(np.nanmean(pred_times)),
                    "seeds": self.seeds,
                }
        if self.save:
            _save(results, "timing", self.results_dir)
        return results


# ---------------------------------------------------------------------------
# MemoryBenchmark
# ---------------------------------------------------------------------------

class MemoryBenchmark:
    """Measure peak RAM usage during fit via tracemalloc."""

    def __init__(
        self,
        model_names: list[str] = MODEL_NAMES,
        task_names: list[str] | None = None,
        hidden_dim: int = 200,
        seed: int = 42,
        n_data: int = 300,
        save: bool = True,
        device: str | torch.device = "cpu",
        results_dir: Path | None = None,
        pinn_epochs: int = 500,
    ) -> None:
        self.model_names = model_names
        self.task_names = task_names or [t.name for t in TASKS]
        self.hidden_dim = hidden_dim
        self.seed = seed
        self.n_data = n_data
        self.save = save
        self.device = resolve_device(str(device))
        self.dtype = dtype_for_device(self.device)
        self.results_dir = results_dir
        self.pinn_epochs = pinn_epochs

    def run(self) -> dict[str, Any]:
        results: dict[str, Any] = {"device": device_info(self.device)}
        for task_name in self.task_names:
            task = TASK_MAP[task_name]
            print(f"[Memory] Task: {task_name}")
            results[task_name] = {}
            dataset = to_device_dataset(
                task.build_dataset(n_data=self.n_data, seed=self.seed), self.device
            )
            for mname in self.model_names:
                print(f"  Model: {mname}", end="", flush=True)
                try:
                    model = _build_model(mname, task.input_dim,
                                         self.hidden_dim, self.seed,
                                         device=self.device, dtype=self.dtype,
                                         pinn_epochs=self.pinn_epochs)
                    tracemalloc.start()
                    _fit_model(model, task, dataset)
                    _, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    peak_mb = peak / 1e6
                    print(f" {peak_mb:.2f} MB")
                except Exception as exc:
                    tracemalloc.stop()
                    peak_mb = float("nan")
                    print(f" ERROR({exc.__class__.__name__})")
                results[task_name][mname] = {"peak_ram_mb": float(peak_mb)}
        if self.save:
            _save(results, "memory", self.results_dir)
        return results


# ---------------------------------------------------------------------------
# AccuracyBenchmark
# ---------------------------------------------------------------------------

class AccuracyBenchmark:
    """Evaluate relative L2 accuracy across model x task x seed."""

    def __init__(
        self,
        model_names: list[str] = MODEL_NAMES,
        task_names: list[str] | None = None,
        hidden_dim: int = 200,
        seeds: list[int] | None = None,
        n_data: int = 300,
        save: bool = True,
        device: str | torch.device = "cpu",
        results_dir: Path | None = None,
        pinn_epochs: int = 500,
    ) -> None:
        self.model_names = model_names
        self.task_names = task_names or [t.name for t in TASKS]
        self.hidden_dim = hidden_dim
        self.seeds = seeds or [42, 0, 1, 2, 3]
        self.n_data = n_data
        self.save = save
        self.device = resolve_device(str(device))
        self.dtype = dtype_for_device(self.device)
        self.results_dir = results_dir
        self.pinn_epochs = pinn_epochs

    def run(self) -> dict[str, Any]:
        results: dict[str, Any] = {"device": device_info(self.device)}
        for task_name in self.task_names:
            task = TASK_MAP[task_name]
            print(f"[Accuracy] Task: {task_name}  device={self.device}")
            results[task_name] = {}
            for mname in self.model_names:
                rel_l2s = []
                print(f"  Model: {mname}", end="", flush=True)
                for seed in self.seeds:
                    try:
                        dataset = to_device_dataset(
                            task.build_dataset(n_data=self.n_data, seed=seed),
                            self.device,
                        )
                        model = _build_model(mname, task.input_dim,
                                             self.hidden_dim, seed,
                                             device=self.device, dtype=self.dtype,
                                             pinn_epochs=self.pinn_epochs)
                        _fit_model(model, task, dataset)
                        X_test = task.get_test_points()
                        u_true = task.u_exact(X_test)
                        X_test_t = to_device_tensor(X_test, self.device)
                        u_pred = model.predict(X_test_t).cpu().detach().numpy().squeeze()
                        rel_l2s.append(_rel_l2(u_true, u_pred))
                        print(".", end="", flush=True)
                    except Exception as exc:
                        rel_l2s.append(float("nan"))
                        print(f"E({exc.__class__.__name__})", end="", flush=True)
                print(f"  mean rel_l2={np.nanmean(rel_l2s):.4e}")
                results[task_name][mname] = {
                    "rel_l2_per_seed": rel_l2s,
                    "rel_l2_mean": float(np.nanmean(rel_l2s)),
                    "rel_l2_std":  float(np.nanstd(rel_l2s)),
                    "seeds": self.seeds,
                }
        if self.save:
            _save(results, "accuracy", self.results_dir)
        return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PyPIELM performance benchmarks")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model names (default: all)")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Task names (default: all 5 tasks)")
    parser.add_argument("--hidden-dim", type=int, default=200)
    parser.add_argument("--n-data", type=int, default=300)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 0, 1, 2, 3])
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--skip-memory", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--pinn-epochs", type=int, default=500,
                        help="Max epochs for PINN models (default: 500).")
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Compute device: cpu | mps | cuda | cuda:N  (default: cpu)",
    )
    parser.add_argument(
        "--platform", type=str, default=None,
        help="Results sub-folder tag (default: derived from --device, e.g. 'mps')."
             " Use to label cross-platform runs: cpu_windows, cuda_windows, etc.",
    )
    args = parser.parse_args()

    from device_utils import resolve_device, device_info
    dev = resolve_device(args.device)
    platform = args.platform or _platform_name(dev)
    rdir = get_results_dir(platform)

    kw: dict[str, Any] = dict(
        model_names=args.models or MODEL_NAMES,
        task_names=args.tasks,
        hidden_dim=args.hidden_dim,
        n_data=args.n_data,
        save=not args.no_save,
        device=args.device,
        results_dir=rdir,
        pinn_epochs=args.pinn_epochs,
    )

    print("=" * 60)
    print("PyPIELM Performance Benchmark")
    print(f"Device:   {dev}  ({device_info(dev)})")
    print(f"Platform: {platform}")
    print(f"Results:  {rdir}")
    print("=" * 60)

    if not args.skip_accuracy:
        print("\n--- Accuracy ---")
        AccuracyBenchmark(seeds=args.seeds, **kw).run()

    if not args.skip_timing:
        print("\n--- Timing ---")
        TimingBenchmark(seeds=args.seeds[:3], **kw).run()

    if not args.skip_memory:
        print("\n--- Memory ---")
        MemoryBenchmark(**{k: v for k, v in kw.items()
                           if k != "seeds"}).run()

    print("\nDone.")

