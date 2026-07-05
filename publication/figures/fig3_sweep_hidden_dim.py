#!/usr/bin/env python3
"""
WP2 — Fig. 3: Accuracy and fit-time vs. hidden_dim for all PIELM variants.

Loads the most recent ``*_sweep_hidden_dim.json`` from
``PyPIELM/benchmarks/results/`` (any sub-folder) and produces:

  fig3_sweep_hidden_dim.pdf   — vector, for LaTeX inclusion
  fig3_sweep_hidden_dim.png   — 300 dpi raster
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
REPO     = Path(__file__).resolve().parent.parent.parent
BENCH    = REPO / "PyPIELM" / "benchmarks" / "results"
OUT      = Path(__file__).resolve().parent

# ── Find most-recent sweep file (search all platform sub-folders) ─────────────
sweep_files = sorted(BENCH.rglob("*sweep_hidden_dim.json"))
if not sweep_files:
    raise FileNotFoundError(
        f"No sweep_hidden_dim.json found under {BENCH}. "
        "Run PyPIELM/benchmarks/sweep_hidden_dim.py first."
    )
src = sweep_files[-1]          # lexicographic timestamp → most recent
print(f"Loading: {src.relative_to(REPO)}")

with open(src) as fh:
    data = json.load(fh)

model_keys = [k for k in data if k not in ("task", "dims", "seeds", "device")]
dims_raw   = data.get("dims", [])
dims       = [int(d) for d in dims_raw]

# Filter models that have any valid (non-NaN) accuracy data
def _valid(mname: str) -> bool:
    rd = data[mname]
    return any(
        not np.isnan(rd[str(d)]["rel_l2_mean"])
        for d in dims_raw
        if str(d) in rd
    )

model_keys = [m for m in model_keys if _valid(m)]

# ── Pretty display names ──────────────────────────────────────────────────────
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
    "vanilla_pinn":           "PINN (vanilla)",
    "adaptive_pinn":          "PINN (adaptive)",
    "fourier_pinn":           "PINN (Fourier)",
    "muon_pinn":              "PINN (Muon)",
    "residual_adaptive_pinn": "PINN (res-adap.)",
}

# ── Figure layout ─────────────────────────────────────────────────────────────
# Double-column journal width: 190 mm ≈ 7.48 in
FS  = 10
FW  = 7.48
FH  = 3.6

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         FS,
    "axes.labelsize":    FS,
    "xtick.labelsize":   FS - 1,
    "ytick.labelsize":   FS - 1,
    "legend.fontsize":   7.5,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth":   1.4,
    "lines.markersize":  4.5,
    "axes.grid":         False,      # ← no background grid
    "figure.dpi":        300,
})

fig, (ax_acc, ax_tim) = plt.subplots(1, 2, figsize=(FW, FH))

# ── Colour cycle ──────────────────────────────────────────────────────────────
# Use a perceptually-uniform 30-colour palette derived from the tab20 pair
cmap20b = plt.get_cmap("tab20b")
cmap20c = plt.get_cmap("tab20c")
colours = [cmap20b(i / 20) for i in range(20)] + \
          [cmap20c(i / 20) for i in range(10)]
markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h",
           "H", "+", "x", "X", "d", "|", "_", "P", "8", "1",
           "2", "3", "4", "o", "s", "^", "D", "v", "<", ">"]

for idx, mname in enumerate(model_keys):
    rd    = data[mname]
    c     = colours[idx % len(colours)]
    mk    = markers[idx % len(markers)]
    label = DISPLAY.get(mname, mname)

    x_vals, rl2, rl2e, ft, fte = [], [], [], [], []
    for d in dims_raw:
        key = str(d)
        if key not in rd:
            continue
        v = rd[key]
        if np.isnan(v["rel_l2_mean"]):
            continue
        x_vals.append(int(d))
        rl2.append(v["rel_l2_mean"])
        rl2e.append(v["rel_l2_std"])
        ft.append(v["fit_time_mean_s"])
        fte.append(v["fit_time_std_s"])

    if not x_vals:
        continue

    kw = dict(color=c, marker=mk, capsize=2, capthick=0.6,
              elinewidth=0.6, linewidth=1.2)
    ax_acc.errorbar(x_vals, rl2, yerr=rl2e, label=label, **kw)
    ax_tim.errorbar(x_vals, ft,  yerr=fte,  label=label, **kw)

# ── Accuracy panel ────────────────────────────────────────────────────────────
ax_acc.set_yscale("log")
ax_acc.set_xlabel(r"Hidden dimension")
ax_acc.set_ylabel(r"Relative $L^2$ error")
ax_acc.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax_acc.tick_params(which="both", direction="in")
ax_acc.set_xlim(left=min(dims) * 0.85, right=max(dims) * 1.10)

# ── Timing panel ─────────────────────────────────────────────────────────────
ax_tim.set_xlabel(r"Hidden dimension")
ax_tim.set_ylabel(r"Fit time (s)")
ax_tim.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax_tim.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
ax_tim.tick_params(which="both", direction="in")
ax_tim.set_xlim(left=min(dims) * 0.85, right=max(dims) * 1.10)
ax_tim.set_ylim(bottom=0)

# ── Shared legend below both panels ──────────────────────────────────────────
n_cols = min(5, max(3, len(model_keys) // 6 + 1))
handles, labels = ax_acc.get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.01),
    ncol=n_cols,
    framealpha=0.9,
    edgecolor="#CCCCCC",
    fontsize=7.0,
    handlelength=1.6,
    columnspacing=0.8,
    handletextpad=0.4,
)

# Extra bottom margin for the legend
fig.tight_layout(pad=0.5)
fig.subplots_adjust(bottom=0.38)

# ── Save ──────────────────────────────────────────────────────────────────────
for fmt in ("pdf", "png"):
    out = OUT / f"fig3_sweep_hidden_dim.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved: {out}")

plt.close(fig)
