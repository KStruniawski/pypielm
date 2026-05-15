"""Device comparison benchmark: CPU vs MPS/CUDA side-by-side.

Runs accuracy and timing benchmarks for each requested device and computes
speedup ratios relative to CPU.  Intended for the publication's hardware
comparison section.

Usage::

    cd PyPIELM
    # Compare CPU against Apple MPS
    python benchmarks/compare_devices.py --devices cpu mps

    # Quick smoke-test with small dims / fewer seeds
    python benchmarks/compare_devices.py --devices cpu mps \\
        --dims 100 200 --seeds 42 0 1 --n-data 200

    # Windows with CUDA
    python benchmarks/compare_devices.py --devices cpu cuda

Outputs (in benchmarks/results/)::

    <ts>_device_comparison.json  — per-device timing + accuracy table
    <ts>_speedup_plot.png        — speedup bar chart (GPU/CPU)
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
    available_devices,
    to_device_dataset,
    to_device_tensor,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

HIDDEN_DIMS = [100, 200, 500]
SEEDS = [42, 0, 1, 2, 3]
N_DATA = 300
DEFAULT_TASKS = ["poisson_1d", "poisson_2d", "heat_1d"]


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_device_comparison(
    devices: list[str] | None = None,
    model_names: list[str] = MODEL_NAMES,
    task_names: list[str] = DEFAULT_TASKS,
    dims: list[int] = HIDDEN_DIMS,
    seeds: list[int] = SEEDS,
    n_data: int = N_DATA,
    save: bool = True,
    results_dir=None,
) -> dict[str, Any]:
    """Run timing + accuracy for each device and collate results.

    Returns a nested dict::

        results["by_device"][device_str][task][model_name][hidden_dim] = {
            "rel_l2_mean": float,
            "rel_l2_std": float,
            "fit_time_mean_s": float,
            "fit_time_std_s": float,
            "pred_time_mean_s": float,
        }
        results["speedup"][task][model_name][hidden_dim] = {
            "<gpu_device>_vs_cpu": float,  # cpu_time / gpu_time
        }
    """
    if devices is None:
        devices = ["cpu"] + [d for d in available_devices() if d != "cpu"]

    # Always resolve + validate each requested device
    dev_objects = {}
    for dname in devices:
        try:
            dev_objects[dname] = resolve_device(dname)
        except ValueError as exc:
            print(f"  [compare_devices] Skipping {dname!r}: {exc}")

    if not dev_objects:
        raise RuntimeError("No valid devices available.")

    print(f"Devices to compare: {list(dev_objects.keys())}")
    print(f"Tasks: {task_names}")
    print(f"Models: {model_names}")
    print(f"Hidden dims: {dims}")
    print(f"Seeds: {seeds}")
    print("=" * 60)

    by_device: dict[str, Any] = {}

    for dname, dev in dev_objects.items():
        dtype = dtype_for_device(dev)
        print(f"\n{'='*60}")
        print(f"Device: {dev}  dtype={dtype}  info={device_info(dev)}")
        print(f"{'='*60}")
        by_device[dname] = {"device_info": device_info(dev)}

        for task_name in task_names:
            task = TASK_MAP[task_name]
            by_device[dname][task_name] = {}

            for mname in model_names:
                by_device[dname][task_name][mname] = {}

                for hd in dims:
                    rel_l2s, fit_times, pred_times = [], [], []
                    print(
                        f"  [{dname}] task={task_name}  model={mname}  "
                        f"hidden_dim={hd}",
                        end="",
                        flush=True,
                    )
                    for seed in seeds:
                        try:
                            dataset = to_device_dataset(
                                task.build_dataset(n_data=n_data, seed=seed), dev
                            )
                            model = _build_model(
                                mname, task.input_dim, hd, seed,
                                device=dev, dtype=dtype,
                            )

                            t0 = time.perf_counter()
                            _fit_model(model, task, dataset)
                            fit_times.append(time.perf_counter() - t0)

                            X_test_t = to_device_tensor(task.get_test_points(), dev)
                            t1 = time.perf_counter()
                            u_pred_raw = model.predict(X_test_t)
                            pred_times.append(time.perf_counter() - t1)

                            u_pred = u_pred_raw.cpu().detach().numpy().squeeze()
                            u_true = task.u_exact(task.get_test_points())
                            rel_l2s.append(_rel_l2(u_true, u_pred))
                            print(".", end="", flush=True)
                        except Exception as exc:
                            rel_l2s.append(float("nan"))
                            fit_times.append(float("nan"))
                            pred_times.append(float("nan"))
                            print(f"E({exc.__class__.__name__})", end="", flush=True)
                    print(
                        f"  rel_l2={np.nanmean(rel_l2s):.3e}  "
                        f"fit={np.nanmean(fit_times)*1000:.1f}ms"
                    )
                    by_device[dname][task_name][mname][hd] = {
                        "rel_l2_mean": float(np.nanmean(rel_l2s)),
                        "rel_l2_std":  float(np.nanstd(rel_l2s)),
                        "fit_time_mean_s": float(np.nanmean(fit_times)),
                        "fit_time_std_s":  float(np.nanstd(fit_times)),
                        "pred_time_mean_s": float(np.nanmean(pred_times)),
                        "seeds": seeds,
                    }

    # --- Compute speedup ratios (each non-CPU device vs CPU) ---
    speedup: dict[str, Any] = {}
    cpu_data = by_device.get("cpu")
    if cpu_data is not None:
        for dname, dev_data in by_device.items():
            if dname == "cpu":
                continue
            speedup[dname] = {}
            for task_name in task_names:
                if task_name not in dev_data or task_name not in cpu_data:
                    continue
                speedup[dname][task_name] = {}
                for mname in model_names:
                    if mname not in dev_data[task_name] or mname not in cpu_data[task_name]:
                        continue
                    speedup[dname][task_name][mname] = {}
                    for hd in dims:
                        cpu_fit = cpu_data[task_name][mname].get(hd, {}).get(
                            "fit_time_mean_s", float("nan")
                        )
                        gpu_fit = dev_data[task_name][mname].get(hd, {}).get(
                            "fit_time_mean_s", float("nan")
                        )
                        ratio = cpu_fit / (gpu_fit + 1e-12)
                        speedup[dname][task_name][mname][hd] = {
                            "speedup_fit": float(ratio),
                            "cpu_fit_s": float(cpu_fit),
                            "gpu_fit_s": float(gpu_fit),
                        }
    else:
        print("  [compare_devices] No CPU baseline found — speedup not computed.")

    results: dict[str, Any] = {
        "devices": list(dev_objects.keys()),
        "tasks": task_names,
        "model_names": model_names,
        "dims": dims,
        "seeds": seeds,
        "by_device": by_device,
        "speedup": speedup,
    }

    if save:
        _save(results, "device_comparison", results_dir)

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_speedup(results: dict[str, Any], save: bool = True, results_dir=None) -> None:
    """Bar chart: speedup of each non-CPU device vs CPU, aggregated over tasks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    speedup = results.get("speedup", {})
    if not speedup:
        print("  [plot_speedup] No speedup data to plot.")
        return

    dims = results["dims"]
    model_names = results["model_names"]

    for dname, task_data in speedup.items():
        # Average speedup per model × dim across tasks
        speedup_matrix: dict[str, dict[int, list[float]]] = {
            m: {hd: [] for hd in dims} for m in model_names
        }
        for task_name, mdict in task_data.items():
            for mname, hddict in mdict.items():
                for hd, info in hddict.items():
                    v = info.get("speedup_fit", float("nan"))
                    if not np.isnan(v):
                        speedup_matrix[mname][hd].append(v)

        # Collapse to mean per model × dim
        x = np.arange(len(dims))
        width = 0.8 / max(len(model_names), 1)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axhline(1.0, color="k", linewidth=0.8, linestyle="--", label="CPU baseline")
        for i, mname in enumerate(model_names):
            means = [np.nanmean(speedup_matrix[mname][hd]) for hd in dims]
            offset = (i - len(model_names) / 2 + 0.5) * width
            ax.bar(x + offset, means, width, label=mname)

        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in dims])
        ax.set_xlabel("Hidden dim")
        ax.set_ylabel(f"Speedup vs CPU  ({dname})")
        ax.set_title(f"Fit-time speedup: {dname} vs CPU")
        ax.legend(fontsize=7, ncol=3)
        plt.tight_layout()

        if save:
            rdir = results_dir or RESULTS_DIR
            rdir.mkdir(parents=True, exist_ok=True)
            path = rdir / f"{_timestamp()}_speedup_{dname}_vs_cpu.png"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            print(f"  Speedup figure -> {path}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    avail = available_devices()
    parser = argparse.ArgumentParser(
        description="CPU vs GPU/MPS device comparison benchmark"
    )
    parser.add_argument(
        "--devices", nargs="+", default=None,
        help=f"Devices to benchmark (default: all available = {avail})",
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS,
                        choices=list(TASK_MAP))
    parser.add_argument("--dims", nargs="+", type=int, default=HIDDEN_DIMS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--n-data", type=int, default=N_DATA)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument(
        "--platform", type=str, default=None,
        help="Results sub-folder tag (default: derived from first non-CPU device).",
    )
    args = parser.parse_args()

    devs = args.devices or avail
    # Derive platform from first non-cpu device (or cpu if only cpu)
    first_non_cpu = next((d for d in devs if d != "cpu"), devs[0] if devs else "cpu")
    platform = args.platform or _platform_name(first_non_cpu)
    rdir = get_results_dir(platform) if not args.no_save else None

    res = run_device_comparison(
        devices=args.devices,
        model_names=args.models or MODEL_NAMES,
        task_names=args.tasks,
        dims=args.dims,
        seeds=args.seeds,
        n_data=args.n_data,
        save=not args.no_save,
        results_dir=rdir,
    )
    plot_speedup(res, save=not args.no_save, results_dir=rdir)
    print("\nDone.")
