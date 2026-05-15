"""Solver comparison: ridge vs RRQR vs Bayesian.

Compares three linear solvers on the same PDE tasks:

* **ridge** — closed-form (HᵀH + λI)⁻¹Hᵀy
* **rrqr** — rank-revealing QR / minimum-norm least-squares
* **bayesian** — sequential Bayesian update (posterior mean)

For each solver × task × seed × condition-number scenario the script records:

    rel_l2       — relative L² error on the analytic solution
    solve_time_s — wall-clock solve time
    cond         — estimated condition number of the assembled design matrix

Results are saved to ``benchmarks/results/<timestamp>_sweep_solver.json``.

Usage::

    cd PyPIELM
    python benchmarks/sweep_solver.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))

from perf_profile import TASK_MAP, RESULTS_DIR, _rel_l2, _save, _timestamp, get_results_dir, _platform_name
from device_utils import (
    resolve_device,
    dtype_for_device,
    device_info,
    to_device_tensor,
)

SEEDS = [42, 0, 1, 2, 3]
HIDDEN_DIMS = [100, 200, 500]          # wider → higher condition number
DEFAULT_TASKS = ["poisson_1d", "poisson_2d"]
N_DATA = 300

# Ridge regularisation lambdas to sweep (ill-conditioning study)
RIDGE_LAMBDAS = [1e-10, 1e-8, 1e-6, 1e-4]


def _assemble_matrix(
    hidden_dim: int,
    input_dim: int,
    n_data: int,
    seed: int,
    task,
    dtype=torch.float64,
    device: torch.device | None = None,
):
    """Build a weighted design matrix from the data + BC blocks."""
    from pypielm.core.feature_maps import RandomFeatureMap
    from pypielm.core.solver import WeightedLinearSystem, bayesian_solve
    from pypielm.data.dataset import PIELMDataset

    if device is None:
        device = torch.device("cpu")

    dataset = task.build_dataset(n_data=n_data, seed=seed)
    fm = RandomFeatureMap(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        activation="tanh",
        seed=seed,
        dtype=dtype,
    ).to(str(device))
    # Data block
    X_d = dataset.X_data.to(device=device, dtype=dtype)
    y_d = dataset.y_data.to(device=device, dtype=dtype)
    H_d = fm(X_d)
    y_d = y_d.reshape(-1, 1)

    # Boundary block (if available)
    blocks_H = [H_d]
    blocks_y = [y_d]

    if dataset.X_bc is not None:
        X_bc = dataset.X_bc.to(device=device, dtype=dtype)
        y_bc = dataset.y_bc.to(device=device, dtype=dtype).reshape(-1, 1)
        H_bc = fm(X_bc)
        blocks_H.append(H_bc)
        blocks_y.append(y_bc)

    H_full = torch.cat(blocks_H, dim=0)
    y_full = torch.cat(blocks_y, dim=0)

    X_test = to_device_tensor(task.get_test_points(), device).to(dtype=dtype)
    y_test = task.u_exact(task.get_test_points())

    return fm, H_full, y_full, X_test, y_test, dataset


def _cond(H: torch.Tensor) -> float:
    try:
        sv = torch.linalg.svdvals(H)
        cond = float(sv.max() / (sv.min() + 1e-16))
    except Exception:
        cond = float("nan")
    return cond


def run_sweep(
    task_names: list[str] = DEFAULT_TASKS,
    dims: list[int] = HIDDEN_DIMS,
    ridge_lambdas: list[float] = RIDGE_LAMBDAS,
    seeds: list[int] = SEEDS,
    n_data: int = N_DATA,
    save: bool = True,
    device: str | torch.device = "cpu",
    results_dir=None,
) -> dict[str, Any]:
    """Run solver comparison sweep.

    Returns nested results dict keyed by task → solver → hidden_dim.
    """
    from pypielm.core.solver import (
        ridge_solve,
        rrqr_solve,
        bayesian_solve,
        WeightedLinearSystem,
    )

    dev = resolve_device(str(device))
    dtype = dtype_for_device(dev)

    results: dict[str, Any] = {
        "dims": dims,
        "ridge_lambdas": ridge_lambdas,
        "seeds": seeds,
        "device": device_info(dev),
    }

    for task_name in task_names:
        task = TASK_MAP[task_name]
        print(f"\n[sweep_solver] Task: {task_name}  device={dev}")
        results[task_name] = {}

        for hd in dims:
            print(f"  hidden_dim={hd}")
            results[task_name][hd] = {}

            # --- Ridge: sweep lambda ---
            ridge_entries: dict[float, dict] = {}
            for lam in ridge_lambdas:
                rel_l2s, times, conds = [], [], []
                for seed in seeds:
                    try:
                        fm, H, y, X_test, y_test, ds = _assemble_matrix(
                            hd, task.input_dim, n_data, seed, task,
                            dtype=dtype, device=dev,
                        )
                        conds.append(_cond(H))
                        t0 = time.perf_counter()
                        beta = ridge_solve(H, y, lam=lam)
                        times.append(time.perf_counter() - t0)
                        u_pred = (fm(X_test) @ beta).cpu().detach().numpy().squeeze()
                        rel_l2s.append(_rel_l2(y_test, u_pred))
                    except Exception as exc:
                        rel_l2s.append(float("nan"))
                        times.append(float("nan"))
                        conds.append(float("nan"))
                        print(f"    ridge lam={lam:.0e} seed={seed}: "
                              f"ERROR {exc.__class__.__name__}")
                ridge_entries[lam] = {
                    "rel_l2_mean": float(np.nanmean(rel_l2s)),
                    "rel_l2_std":  float(np.nanstd(rel_l2s)),
                    "solve_time_mean_s": float(np.nanmean(times)),
                    "cond_mean": float(np.nanmean(conds)),
                }
                print(f"    ridge λ={lam:.0e}: "
                      f"rel_l2={np.nanmean(rel_l2s):.3e}  "
                      f"cond={np.nanmean(conds):.2e}")
            results[task_name][hd]["ridge"] = ridge_entries

            # --- RRQR ---
            rel_l2s, times = [], []
            for seed in seeds:
                try:
                    fm, H, y, X_test, y_test, ds = _assemble_matrix(
                        hd, task.input_dim, n_data, seed, task,
                        dtype=dtype, device=dev,
                    )
                    t0 = time.perf_counter()
                    beta = rrqr_solve(H, y)
                    times.append(time.perf_counter() - t0)
                    u_pred = (fm(X_test) @ beta).cpu().detach().numpy().squeeze()
                    rel_l2s.append(_rel_l2(y_test, u_pred))
                except Exception as exc:
                    rel_l2s.append(float("nan"))
                    times.append(float("nan"))
                    print(f"    rrqr seed={seed}: ERROR {exc.__class__.__name__}")
            results[task_name][hd]["rrqr"] = {
                "rel_l2_mean": float(np.nanmean(rel_l2s)),
                "rel_l2_std":  float(np.nanstd(rel_l2s)),
                "solve_time_mean_s": float(np.nanmean(times)),
            }
            print(f"    rrqr:         rel_l2={np.nanmean(rel_l2s):.3e}")

            # --- Bayesian ---
            rel_l2s, times = [], []
            for seed in seeds:
                try:
                    fm, H, y, X_test, y_test, ds = _assemble_matrix(
                        hd, task.input_dim, n_data, seed, task,
                        dtype=dtype, device=dev,
                    )
                    blk = WeightedLinearSystem(H=H, y=y, weight=1.0)
                    t0 = time.perf_counter()
                    result = bayesian_solve([blk], prior_precision=1e-4)
                    times.append(time.perf_counter() - t0)
                    beta_mean = result.beta_mean
                    u_pred = (fm(X_test) @ beta_mean).cpu().detach().numpy().squeeze()
                    rel_l2s.append(_rel_l2(y_test, u_pred))
                except Exception as exc:
                    rel_l2s.append(float("nan"))
                    times.append(float("nan"))
                    print(f"    bayesian seed={seed}: ERROR {exc.__class__.__name__}")
            results[task_name][hd]["bayesian"] = {
                "rel_l2_mean": float(np.nanmean(rel_l2s)),
                "rel_l2_std":  float(np.nanstd(rel_l2s)),
                "solve_time_mean_s": float(np.nanmean(times)),
            }
            print(f"    bayesian:     rel_l2={np.nanmean(rel_l2s):.3e}")

    if save:
        _save(results, "sweep_solver", results_dir)
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Solver comparison benchmark")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS,
                        choices=list(TASK_MAP))
    parser.add_argument("--dims", nargs="+", type=int, default=HIDDEN_DIMS)
    parser.add_argument("--lambdas", nargs="+", type=float, default=RIDGE_LAMBDAS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--n-data", type=int, default=N_DATA)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Compute device: cpu | mps | cuda | cuda:N",
    )
    parser.add_argument(
        "--platform", type=str, default=None,
        help="Results sub-folder tag (default: derived from --device).",
    )
    args = parser.parse_args()

    dev = resolve_device(args.device)
    platform = args.platform or _platform_name(dev)
    rdir = get_results_dir(platform) if not args.no_save else None

    run_sweep(
        task_names=args.tasks,
        dims=args.dims,
        ridge_lambdas=args.lambdas,
        seeds=args.seeds,
        n_data=args.n_data,
        save=not args.no_save,
        device=args.device,
        results_dir=rdir,
    )
    print("Done.")
