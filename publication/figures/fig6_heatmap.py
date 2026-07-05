#!/usr/bin/env python3
"""
Fig 6 — Leaderboard heatmap: relative L² error across all models and tasks.

Models on y-axis (grouped by paradigm, sorted best-to-worst within group),
tasks on x-axis.  Colormap: blue-red (RdBu_r), log₁₀ scale, TwoSlopeNorm
centred at log₁₀(1) = 0 (blue = low error, red = high error).

SoftwareX double-column width (7.01 in), 300 dpi, serif 9 pt.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[2]
DATA    = ROOT / "PyPIELM-SoftwareX" / "scripts" / "output" / "paper_stats_canonical.csv"
OUT_DIR = Path(__file__).resolve().parent

# ── display names ─────────────────────────────────────────────────────────────
DISPLAY_MODEL = {
    "BayesianPIELMRegressor":         "Bayesian PIELM",
    "CorePIELMRegressor":             "Core PIELM",
    "CurriculumPIELMRegressor":       "Curriculum PIELM",
    "DDELMCoarseRegressor":           "DD-ELM",
    "DELMRegressor":                  "D-ELM",
    "DPIELMRegressor":                "DP-IELM",
    "EigPIELMRegressor":              "Eig-PIELM",
    "FPIELMRegressor":                "FP-IELM",
    "GFFPIELMRegressor":              "GFF-PIELM",
    "KAPIELMRegressor":               "KAP-IELM",
    "LSEELMRegressor":                "LSE-ELM",
    "LocELMRegressor":                "Loc-ELM",
    "NormalEquationELMRegressor":     "NE-ELM",
    "NullSpacePIELMRegressor":        "NS-PIELM",
    "PIELMRVDSRegressor":             "PIELM-RVDS",
    "ParameterRetentionELMRegressor": "PR-ELM",
    "PiecewiseELMRegressor":          "Piecewise ELM",
    "RINNRegressor":                  "RINN",
    "RRQRELMFBPINNRegressor":         "RRQR-ELM+FBPINN",
    "RaNNPIELMRegressor":             "RaNN-PIELM",
    "SGEPIELMRegressor":              "SGE-PIELM",
    "SoftPartitionKAPIELMRegressor":  "SoftPart-KAPIELM",
    "StefanPIELMRegressor":           "Stefan PIELM",
    "TSPIELMRegressor":               "TS-PIELM",
    "VanillaPIELMRegressor":          "Vanilla PIELM",
    "VanillaPIENNRegressor":          "Vanilla PINN",
    "XPIELMRegressor":                "X-PIELM",
    "AdaptivePINNRegressor":          "Adaptive PINN",
    "FourierPINNRegressor":           "Fourier PINN",
    "MuonPINNRegressor":              "Muon PINN",
    "ResidualAdaptivePINNRegressor":  "Res.-Adapt. PINN",
    "VanillaPINNRegressor":           "Vanilla PINN",
    "FiniteDifferenceHeat1D":         "FD Heat-1D",
    "FiniteDifferencePoisson1D":      "FD Poisson-1D",
    "FiniteDifferencePoisson2D":      "FD Poisson-2D",
    "FiniteElementPoisson1D":         "FE Poisson-1D",
    "FiniteVolumeHeat1D":             "FV Heat-1D",
    "SpectralPoisson1D":              "Spectral Poisson-1D",
}

DISPLAY_TASK = {
    "poisson_classic":      "Poisson 1D",
    "burgers1d":            "Burgers 1D",
    "heat_longtime":        "Heat (long-time)",
    "Kuramoto_Sivashinsky": "Kuramoto\u2013Sivashinsky",
    "ns2d":                 "Navier\u2013Stokes 2D",
}

TASK_ORDER = [
    "poisson_classic",
    "burgers1d",
    "heat_longtime",
    "Kuramoto_Sivashinsky",
    "ns2d",
]

PARADIGM_ORDER = ["PIELM", "PINN", "Traditional"]

# ── load & pivot ──────────────────────────────────────────────────────────────
df    = pd.read_csv(DATA)
pivot = df.pivot_table(index="model", columns="task",
                       values="relative_l2_mean", aggfunc="first")
pivot = pivot[TASK_ORDER]

paradigm_map       = df.groupby("model")["paradigm"].first()
pivot["_paradigm"] = paradigm_map

with np.errstate(divide="ignore", invalid="ignore"):
    _log = np.log10(np.where(pivot[TASK_ORDER].values > 0,
                             pivot[TASK_ORDER].values, np.nan))
pivot["_log_mean"] = np.nanmean(_log, axis=1)

pivot = (
    pivot
    .assign(_paradigm_ord=pivot["_paradigm"].map(
        {p: i for i, p in enumerate(PARADIGM_ORDER)}))
    .sort_values(["_paradigm_ord", "_log_mean"])
)

# ── matrices ──────────────────────────────────────────────────────────────────
mat_raw  = pivot[TASK_ORDER].values.astype(float)
with np.errstate(divide="ignore", invalid="ignore"):
    mat_log = np.log10(np.where(mat_raw > 0, mat_raw, np.nan))

model_keys   = list(pivot.index)
model_labels = [DISPLAY_MODEL.get(m, m) for m in model_keys]
task_labels  = [DISPLAY_TASK[t] for t in TASK_ORDER]
paradigms    = list(pivot["_paradigm"].values)
n_models     = len(model_labels)
n_tasks      = len(task_labels)

# Row indices where a new paradigm begins (> 0)
group_boundaries = [i for i in range(1, n_models) if paradigms[i] != paradigms[i - 1]]

# (label, start_row, end_row) for each paradigm group
par_spans = []
starts = [0] + group_boundaries
ends   = group_boundaries + [n_models]
for p, s, e in zip(PARADIGM_ORDER, starts, ends):
    par_spans.append((p, s, e))

# ── rcParams ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       9,
    "axes.labelsize":  9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# ── figure ────────────────────────────────────────────────────────────────────
ROW_H = 0.215          # inches per model row
FIG_W = 7.01
FIG_H = n_models * ROW_H + 1.8   # extra for top x-labels and bottom colorbar

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

# ── colormap / normalisation ──────────────────────────────────────────────────
# Centre at 0 (log₁₀(1) = 0): blue → good (L² < 1), red → bad (L² ≥ 1).
VMIN, VCENTER, VMAX = -2.5, 0.0, 2.0
norm = mcolors.TwoSlopeNorm(vmin=VMIN, vcenter=VCENTER, vmax=VMAX)
cmap = plt.cm.RdBu_r

mat_plot = np.clip(mat_log, VMIN, VMAX)  # clip for rendering; annotations show true value
img = ax.imshow(mat_plot, aspect="auto", cmap=cmap, norm=norm,
                origin="upper", interpolation="nearest")

# ── cell annotations ──────────────────────────────────────────────────────────
for i in range(n_models):
    for j in range(n_tasks):
        val  = mat_raw[i, j]
        logv = mat_log[i, j]
        if np.isnan(val):
            ax.text(j, i, "\u2014", ha="center", va="center",
                    fontsize=7.5, color="#555")
            continue
        normed  = norm(np.clip(logv, VMIN, VMAX))   # 0–1 position in cmap
        txt_col = "white" if (normed < 0.20 or normed > 0.80) else "black"
        # Compact label
        if val >= 10:
            lbl = f"{val:.0f}"
        elif val >= 1.0:
            lbl = f"{val:.2f}"
        elif val >= 0.01:
            lbl = f"{val:.3f}"
        else:
            exp  = int(np.floor(np.log10(val)))
            coef = val / 10 ** exp
            lbl  = f"{coef:.1f}e{exp}"
        ax.text(j, i, lbl, ha="center", va="center",
                fontsize=6.5, color=txt_col)

# ── group separator lines ─────────────────────────────────────────────────────
for b in group_boundaries:
    ax.axhline(b - 0.5, color="black", lw=1.8, zorder=3)

# ── paradigm bracket labels (right of axes, clip_on=False) ───────────────────
for p_label, s, e in par_spans:
    mid = (s + e - 1) / 2.0
    ax.text(n_tasks - 0.5 + 0.25, mid, p_label,
            ha="left", va="center", fontsize=8.5, fontweight="bold",
            clip_on=False, transform=ax.transData, color="#1a1a1a")

# ── axes ticks ────────────────────────────────────────────────────────────────
ax.set_xticks(range(n_tasks))
ax.set_yticks(range(n_models))
ax.set_yticklabels(model_labels, fontsize=8.5)

ax.xaxis.set_ticks_position("top")
ax.xaxis.set_label_position("top")
ax.set_xticklabels(task_labels, rotation=35, ha="left", fontsize=9)

for sp in ax.spines.values():
    sp.set_visible(False)

# Fine cell grid
ax.set_xticks(np.arange(-0.5, n_tasks, 1), minor=True)
ax.set_yticks(np.arange(-0.5, n_models, 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
ax.tick_params(which="minor", bottom=False, left=False, top=False)

# ── colorbar ──────────────────────────────────────────────────────────────────
cbar = fig.colorbar(img, ax=ax, orientation="vertical",
                    pad=0.01, fraction=0.022, shrink=0.32, aspect=18,
                    extend="both")
cbar.set_label(r"$\log_{10}\!\left(\bar{e}_{L^2}\right)$", fontsize=9)
cbar.set_ticks([-2, -1, 0, 1, 2])
cbar.set_ticklabels(
    [r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$", r"$10^{2}$"],
    fontsize=8.5)

# ── save ──────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    path = OUT_DIR / f"fig6_heatmap.{ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved {path.relative_to(ROOT)}")

print(f"Done — {n_models} models × {n_tasks} tasks")
