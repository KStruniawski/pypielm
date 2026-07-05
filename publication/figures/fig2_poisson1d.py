#!/usr/bin/env python3
"""
WP2 — Fig. 2: 1D Poisson Equation — CorePIELM predicted vs. exact solution.

Problem:  -u''(x) = π² sin(πx),  x ∈ [0,1],  u(0)=u(1)=0
Exact:     u(x) = sin(πx)
Model:     CorePIELM, hidden_dim=300, ridge_lambda=1e-10, seed=42

Exports:
  fig2_poisson1d.pdf   — vector, for LaTeX inclusion
  fig2_poisson1d.png   — 300 dpi raster
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

# ── Make sure the local pypielm is importable ────────────────────────────────
REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "PyPIELM"))

from pypielm.data.dataset import PIELMDataset
from pypielm.models import CorePIELM
from pypielm.core.solver import WeightedLinearSystem

OUT = Path(__file__).resolve().parent

# ────────────────────────────────────────────────────────────────────────────
# Problem configuration
# ────────────────────────────────────────────────────────────────────────────
N_COLLOC = 200
N_TEST   = 500
HIDDEN   = 300
SEED     = 42

rng  = np.random.default_rng(SEED)
X_c  = rng.uniform(0.0, 1.0, (N_COLLOC, 1))
X_bc = np.array([[0.0], [1.0]])
y_bc = np.array([0.0, 0.0])

dataset = PIELMDataset.from_arrays(X_c, X_bc=X_bc, y_bc=y_bc)


def pde_operator(fm, X: torch.Tensor) -> WeightedLinearSystem:
    """-u'' = π² sin(πx)  encoded as a weighted linear block."""
    H_xx = fm.d2(X, 0)
    rhs  = (math.pi ** 2
            * torch.sin(math.pi * X[:, 0:1]).to(X.dtype))
    return WeightedLinearSystem(H=-H_xx, y=rhs, weight=1.0)


# ────────────────────────────────────────────────────────────────────────────
# Fit and evaluate
# ────────────────────────────────────────────────────────────────────────────
model = CorePIELM(hidden_dim=HIDDEN, ridge_lambda=1e-10, seed=SEED)
model.fit(dataset, pde_operator=pde_operator)

x_test = np.linspace(0.0, 1.0, N_TEST)
u_true = np.sin(math.pi * x_test)

with torch.no_grad():
    u_pred = (
        model.predict(torch.tensor(x_test.reshape(-1, 1), dtype=torch.float64))
        .numpy()
        .squeeze()
    )

rel_l2 = float(
    np.linalg.norm(u_true - u_pred) / (np.linalg.norm(u_true) + 1e-12)
)
print(f"Relative L² error: {rel_l2:.2e}")

# ────────────────────────────────────────────────────────────────────────────
# Publication figure
# Single-column journal width: 90 mm ≈ 3.54 in
# ────────────────────────────────────────────────────────────────────────────
FS   = 10          # base font size (pt)
FW   = 3.54        # figure width  (in)  — Elsevier single column
FH   = 2.70        # figure height (in)

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         FS,
    "axes.labelsize":    FS,
    "xtick.labelsize":   FS - 1,
    "ytick.labelsize":   FS - 1,
    "legend.fontsize":   FS - 1.5,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth":   1.6,
    "axes.grid":         False,
    "figure.dpi":        300,
})

fig, ax = plt.subplots(figsize=(FW, FH))

# ── Exact solution ────────────────────────────────────────────────────────
ax.plot(x_test, u_true,
        linestyle="--", color="#E67E22", linewidth=1.4,
        label=r"Exact $(\sin\pi x)$", zorder=2)

# ── CorePIELM prediction ─────────────────────────────────────────────────
ax.plot(x_test, u_pred,
        linestyle="-",  color="#2471A3", linewidth=1.8,
        label="CorePIELM (predicted)", zorder=3)

# ── Absolute error band ───────────────────────────────────────────────────
err = np.abs(u_pred - u_true)
ax.fill_between(x_test,
                u_pred - err, u_pred + err,
                alpha=0.20, color="#2471A3",
                label="Abs. error band", zorder=1)

# ── Axes ─────────────────────────────────────────────────────────────────
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$u(x)$")
ax.set_xlim(0.0, 1.0)
ax.set_ylim(-0.05, 1.10)

# Minor ticks
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
ax.tick_params(which="both", direction="in")

# ── Inset annotation: L² error ───────────────────────────────────────────
ax.text(0.97, 0.05,
        fr"Rel. $L^2 = {rel_l2:.2e}$",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=FS - 2, color="#333333",
        bbox=dict(boxstyle="round,pad=0.2",
                  facecolor="white", edgecolor="#BBBBBB",
                  linewidth=0.6, alpha=0.85))

# ── Legend ────────────────────────────────────────────────────────────────
leg = ax.legend(loc="upper left",
                framealpha=0.85, edgecolor="#CCCCCC",
                handlelength=1.8, labelspacing=0.3)
leg.get_frame().set_linewidth(0.6)

fig.tight_layout(pad=0.4)

# ────────────────────────────────────────────────────────────────────────────
# Save
# ────────────────────────────────────────────────────────────────────────────
for fmt in ("pdf", "png"):
    out = OUT / f"fig2_poisson1d.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved: {out}")

plt.close(fig)
