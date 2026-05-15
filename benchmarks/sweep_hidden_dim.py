"""Hidden-dimension sweep: accuracy vs training time (Pareto curve).

For each model in ``MODEL_NAMES``, sweeps ``hidden_dim`` ∈ {50, 100, 200,
500, 1000} over multiple seeds on the Poisson-1D task and records::

    rel_l2   — relative L² error on test set
    fit_time — wall-clock seconds

Results are saved to ``benchmarks/results/<timestamp>_sweep_hidden_dim.json``
and a Pareto figure is saved to
``benchmarks/results/<timestamp>_pareto_hidden_dim.png``.

Usage::

    cd PyPIELM
    python benchmarks/sweep_hidden_dim.py
    python benchmarks/sweep_hidden_dim.py --task poisson_2d --dims 50 100 200
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Import shared utilities from perf_profile
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).parent))

from perf_profile import (
    TASK_MAP,
    MODEL_NAMES,
    _build_model,
    _fit_model,
    _rel_l2,
    _save,
    _timestamp,
    RESULTS_DIR,
    get_results_dir,
    _platform_name,
)
from device_utils import (
    resolve_device,
    dtype_for_device,
    device_info,
    to_device_dataset,
    to_device_tensor,
)

HIDDEN_DIMS = [50, 100, 200, 500, 1000]
SEEDS = [42, 0, 1, 2, 3]
DEFAULT_TASK = "poisson_1d"
N_DATA = 300


def run_sweep(
    model_names: list[str] = MODEL_NAMES,
    task_name: str = DEFAULT_TASK,
    dims: list[int] = HIDDEN_DIMS,
    seeds: list[int] = SEEDS,
    n_data: int = N_DATA,
    save: bool = True,
    device: str | torch.device = "cpu",
    results_dir=None,
) -> dict[str, Any]:
    """Run the hidden-dim accuracy/timing sweep.

    Returns nested dict::

        results[model_name][hidden_dim] = {
            "rel_l2_mean": float,
            "rel_l2_std": float,
            "fit_time_mean_s": float,
            "fit_time_std_s": float,
            "seeds": list[int],
        }
    """
    dev = resolve_device(str(device))
    dtype = dtype_for_device(dev)
    task = TASK_MAP[task_name]
    results: dict[str, Any] = {
        "task": task_name, "dims": dims, "seeds": seeds,
        "device": device_info(dev),
    }

    for mname in model_names:
        print(f"[sweep_hidden_dim] model={mname}  device={dev}")
        results[mname] = {}
        for hd in dims:
            rel_l2s, fit_times = [], []
            print(f"  hidden_dim={hd}", end="", flush=True)
            for seed in seeds:
                try:
                    dataset = to_device_dataset(
                        task.build_dataset(n_data=n_data, seed=seed), dev
                    )
                    model = _build_model(mname, task.input_dim, hd, seed,
                                        device=dev, dtype=dtype)

                    t0 = time.perf_counter()
                    _fit_model(model, task, dataset)
                    fit_times.append(time.perf_counter() - t0)

                    X_test = task.get_test_points()
                    u_true = task.u_exact(X_test)
                    X_test_t = to_device_tensor(X_test, dev)
                    u_pred = model.predict(X_test_t).cpu().detach().numpy().squeeze()
                    rel_l2s.append(_rel_l2(u_true, u_pred))
                    print(".", end="", flush=True)
                except Exception as exc:
                    rel_l2s.append(float("nan"))
                    fit_times.append(float("nan"))
                    print(f"E({exc.__class__.__name__})", end="", flush=True)
            rl2_mean = float(np.nanmean(rel_l2s))
            ft_mean = float(np.nanmean(fit_times))
            print(f"  rel_l2={rl2_mean:.3e}  fit_time={ft_mean:.3f}s")
            results[mname][hd] = {
                "rel_l2_mean": rl2_mean,
                "rel_l2_std": float(np.nanstd(rel_l2s)),
                "fit_time_mean_s": ft_mean,
                "fit_time_std_s": float(np.nanstd(fit_times)),
                "seeds": seeds,
            }

    if save:
        _save(results, "sweep_hidden_dim", results_dir)

    return results


# ---------------------------------------------------------------------------
# Plot Pareto front
# ---------------------------------------------------------------------------

def plot_sweep(results: dict[str, Any], save: bool = True, results_dir=None) -> None:
    """Plot accuracy–time Pareto front for the hidden-dim sweep."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dims = results["dims"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Axis 1: accuracy vs hidden_dim
    ax_acc = axes[0]
    ax_acc.set_title("Accuracy vs Hidden Dim")
    ax_acc.set_xlabel("Hidden dim")
    ax_acc.set_ylabel("Relative $L^2$ error")
    ax_acc.set_yscale("log")

    # Axis 2: timing vs hidden_dim
    ax_tim = axes[1]
    ax_tim.set_title("Fit Time vs Hidden Dim")
    ax_tim.set_xlabel("Hidden dim")
    ax_tim.set_ylabel("Fit time (s)")

    model_keys = [k for k in results if k not in ("task", "dims", "seeds")]
    for mname in model_keys:
        rd = results[mname]
        x = [int(d) for d in dims if int(d) in {int(k) for k in rd}]
        rl2  = [rd[d]["rel_l2_mean"]    for d in x]
        rl2e = [rd[d]["rel_l2_std"]     for d in x]
        ft   = [rd[d]["fit_time_mean_s"] for d in x]
        fte  = [rd[d]["fit_time_std_s"]  for d in x]

        ax_acc.errorbar(x, rl2, yerr=rl2e, marker="o", label=mname, capsize=3)
        ax_tim.errorbar(x, ft,  yerr=fte,  marker="o", label=mname, capsize=3)

    ax_acc.legend(fontsize=7)
    ax_tim.legend(fontsize=7)
    plt.tight_layout()

    if save:
        rdir = results_dir or RESULTS_DIR
        rdir.mkdir(parents=True, exist_ok=True)
        path = rdir / f"{_timestamp()}_pareto_hidden_dim.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"  Figure -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hidden-dim sweep benchmark")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--task", default=DEFAULT_TASK, choices=list(TASK_MAP))
    parser.add_argument("--dims", nargs="+", type=int, default=HIDDEN_DIMS)
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

    from device_utils import resolve_device
    dev = resolve_device(args.device)
    platform = args.platform or _platform_name(dev)
    rdir = get_results_dir(platform) if not args.no_save else None

    res = run_sweep(
        model_names=args.models or MODEL_NAMES,
        task_name=args.task,
        dims=args.dims,
        seeds=args.seeds,
        n_data=args.n_data,
        save=not args.no_save,
        device=args.device,
        results_dir=rdir,
    )
    plot_sweep(res, save=not args.no_save, results_dir=rdir)
    print("Done.")
