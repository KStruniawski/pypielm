#!/usr/bin/env python3
"""
WP2 — Fig. 4: Fit-time speedup — CUDA vs CPU (all PIELM variants).

Loads the most recent ``*_device_comparison.json`` from
``PyPIELM/benchmarks/results/cuda/`` and produces:

  fig4_speedup_cuda.pdf   — vector, for LaTeX inclusion
  fig4_speedup_cuda.png   — 300 dpi raster

Speedup is averaged across tasks (poisson_1d, poisson_2d, heat_1d) for each
model × hidden_dim combination.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
REPO   = Path(__file__).resolve().parent.parent.parent
BENCH  = REPO / "PyPIELM" / "benchmarks" / "results"
OUT    = Path(__file__).resolve().parent

# ── Find most-recent cuda device_comparison JSON ──────────────────────────────
src_files = sorted((BENCH / "cuda").glob("*device_comparison.json"))
if not src_files:
    # Fall back to any platform folder
    src_files = sorted(BENCH.rglob("*device_comparison.json"))
if not src_files:
    raise FileNotFoundError(
        f"No device_comparison.json found under {BENCH}. "
        "Run PyPIELM/benchmarks/compare_devices.py first."
    )
src = src_files[-1]
print(f"Loading: {src.relative_to(REPO)}")

with open(src) as fh:
    data = json.load(fh)

model_names = data["model_names"]
dims_raw    = data["dims"]                 # e.g. [100, 200]
dims        = [int(d) for d in dims_raw]

# ── Identify the accelerator device name (first non-cpu key in speedup) ───────
speedup_all = data.get("speedup", {})
if not speedup_all:
    raise ValueError("No 'speedup' key in the loaded JSON.")
accel_name = next(iter(speedup_all))       # e.g. "cuda"
task_data  = speedup_all[accel_name]       # {task: {model: {dim: {speedup_fit}}}}

# ── Compute mean speedup per model × dim, averaged over tasks ─────────────────
# Shape: speedup_mean[model][dim_int] = float (NaN if no data)
speedup_mean: dict[str, dict[int, float]] = {}
for mname in model_names:
    speedup_mean[mname] = {}
    for hd in dims:
        vals = [
            task_data[tk][mname][str(hd)]["speedup_fit"]
            for tk in task_data
            if mname in task_data[tk] and str(hd) in task_data[tk][mname]
        ]
        speedup_mean[mname][hd] = float(np.nanmean(vals)) if vals else float("nan")

# ── Pretty display names (same palette as Fig 3) ─────────────────────────────
DISPLAY = {
    "vanilla_pielm":          "Vanilla PIELM",
    "core_pielm":             "CorePIELM",
    "bayesian_pielm":         "Bayesian PIELM",
    "gff_pielm":              "GFF-PIELM",
    "curriculum_pielm":       "Curriculum PIELM",
    "nullspace_pielm":        "Nullspace PIELM",
    "eig_pielm":              "Eig-PIELM",
    "lseelm":                 "LSE-ELM",
    "stefan_pielm":           "Stefan-PIELM",
    "normal_equation_elm":    "NormalEq-ELM",
    "parameter_retention_elm":"ParamRet-ELM",
    "piecewise_elm":          "Piecewise-ELM",
    "delm":                   "D-ELM",
    "fpielm":                 "F-PIELM",
    "sgepielm":               "SGE-PIELM",
    "rinn":                   "RINN",
    "rann_pielm":             "RANN-PIELM",
    "xpielm":                 "X-PIELM",
    "pielm_rvds":             "PIELM-RVDS",
    "tspielm":                "TS-PIELM",
    "kapielm":                "KA-PIELM",
    "soft_partition_kapielm": "SK-PIELM",
    "dpielm":                 "D-PIELM",
    "locelm":                 "LOC-ELM",
    "ddelm_coarse":           "DDe-LM (coarse)",
}

# ── Colour palette (consistent with Fig 3) ───────────────────────────────────
cmap20b = plt.get_cmap("tab20b")
cmap20c = plt.get_cmap("tab20c")
colours = [cmap20b(i / 20) for i in range(20)] + \
          [cmap20c(i / 20) for i in range(10)]

# ── Figure layout ─────────────────────────────────────────────────────────────
# Double-column journal width: 178 mm ≈ 7.01 in
FS  = 10
FW  = 7.01
FH  = 3.8

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         FS,
    "axes.labelsize":    FS,
    "xtick.labelsize":   FS - 1,
    "ytick.labelsize":   FS - 1,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "axes.grid":         False,   # ← no background grid
    "figure.dpi":        300,
})

fig, ax = plt.subplots(figsize=(FW, FH))

# ── Grouped bar chart ─────────────────────────────────────────────────────────
n_models = len(model_names)
n_dims   = len(dims)
group_w  = 0.82                          # fraction of inter-group space used
bar_w    = group_w / n_models
x_base   = np.arange(n_dims, dtype=float)

bar_handles = []
for i, mname in enumerate(model_names):
    c      = colours[i % len(colours)]
    label  = DISPLAY.get(mname, mname)
    offset = (i - n_models / 2 + 0.5) * bar_w
    vals   = [speedup_mean[mname].get(hd, float("nan")) for hd in dims]
    bars   = ax.bar(x_base + offset, vals, bar_w,
                    color=c, label=label, linewidth=0)
    bar_handles.append(bars)

# CPU = 1.0 reference line
baseline = ax.axhline(1.0, color="black", linewidth=1.0,
                      linestyle="--", label="CPU baseline (1×)", zorder=5)

# ── Axes ─────────────────────────────────────────────────────────────────────
ax.set_xticks(x_base)
ax.set_xticklabels([str(d) for d in dims])
ax.set_xlabel("Hidden dimension")
ax.set_ylabel(f"Speedup vs. CPU ({accel_name.upper()})")
ax.set_ylim(bottom=0)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
ax.tick_params(which="both", direction="in")

# ── 2-column legend outside the axes, below the plot ─────────────────────────
# Collect handles/labels; put CPU baseline first
all_handles, all_labels = ax.get_legend_handles_labels()
# Reorder so 'CPU baseline' is first
bl_idx = all_labels.index("CPU baseline (1×)")
ordered_h = [all_handles[bl_idx]] + [h for j, h in enumerate(all_handles) if j != bl_idx]
ordered_l = [all_labels[bl_idx]]  + [l for j, l in enumerate(all_labels)  if j != bl_idx]

fig.tight_layout(pad=0.5)

# 2-column legend anchored below the axes (bbox_to_anchor in axes coordinates;
# bbox_inches="tight" on save captures it even when it falls outside the figure).
leg = ax.legend(
    ordered_h, ordered_l,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.14),    # just below x-axis label
    ncol=2,                          # ← 2 columns as requested
    framealpha=0.9,
    edgecolor="#CCCCCC",
    fontsize=7.5,
    handlelength=1.4,
    handleheight=0.9,
    columnspacing=1.0,
    handletextpad=0.4,
    labelspacing=0.25,
)

# ── Save ──────────────────────────────────────────────────────────────────────
for fmt in ("pdf", "png"):
    out = OUT / f"fig4_speedup_cuda.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved: {out}")

plt.close(fig)
