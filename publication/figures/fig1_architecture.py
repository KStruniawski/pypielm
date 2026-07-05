#!/usr/bin/env python3
"""
WP2 — Fig. 1: PyPIELM Architecture Diagram
Exports: fig1_architecture.pdf  and  fig1_architecture.png  (300 dpi)

Layout (top → bottom):
  Title banner
  ┌──────────────┐   ┌─────────────────────────────────────────┐   ┌─────────────────┐
  │  CLI / YAML  │──►│  pypielm.data  (Adapters · Dataset)      │◄──│  Data Sources   │
  └──────────────┘   └─────────────────────────────────────────┘   └─────────────────┘
                                         ↓
                      ┌─────────────────────────────────────────┐
                      │  pypielm.pde  (operators · BC/IC)        │
                      └─────────────────────────────────────────┘
                                         ↓
                      ┌─────────────────────────────────────────┐
                      │  pypielm.core  (ELMBase · solver)        │
                      └─────────────────────────────────────────┘
                                         ↓
              ┌──────────────────────────────────────────────────────┐
              │  pypielm.models  (26+ variants + 4 PINN baselines)    │
              │ [Vanilla][Bayesian][Fourier][Curriculum][Domain][…]   │
              └──────────────────────────────────────────────────────┘
                         ↓                ↓                ↓
               ┌───────────────┐ ┌──────────────┐ ┌──────────────────┐
               │pypielm.metrics│ │  pypielm.io  │ │pypielm.visualiz. │
               └───────────────┘ └──────────────┘ └──────────────────┘
                         ↓                ↓                ↓
              ┌──────────────────────────────────────────────────────┐
              │         fit()  ·  predict()  ·  score()  API          │
              └──────────────────────────────────────────────────────┘
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT = Path(__file__).resolve().parent

# ────────────────────────────────────────────────────────────────────────────
# Colour palette  (soft, publication-friendly)
# ────────────────────────────────────────────────────────────────────────────
C_TITLE  = "#1B4F72"   # dark navy   — title banner fill
C_DATA   = "#D6EAF8"   # sky blue    — data layer
C_PDE    = "#E8DAEF"   # lavender    — pde layer
C_CORE   = "#FDEBD0"   # peach       — core layer
C_MODEL  = "#D5F5E3"   # mint green  — models layer
C_OUT    = "#FEF9E7"   # cream       — output modules
C_VIZ    = "#FDEDEC"   # light rose  — visualization
C_CLI    = "#EBF5EB"   # pale green  — CLI / YAML
C_EXT    = "#EBF5FB"   # pale blue   — external sources
C_API    = "#F2F3F4"   # light grey  — API banner
C_EDGE   = "#4A4A4A"   # near-black  — default edge
C_ARROW  = "#2C3E50"   # dark slate  — main arrows

# Sub-box colours for model categories
SUB_COLS = ["#85C1E9", "#AED6F1", "#C39BD3", "#F8C471",
            "#7DCEA0", "#F1948A", "#A9CCE3"]

# ────────────────────────────────────────────────────────────────────────────
# Canvas
# ────────────────────────────────────────────────────────────────────────────
FW, FH = 8.5, 9.2          # inches
fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis("off")
fig.patch.set_facecolor("white")


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
def rbox(ax, cx, cy, w, h, fc, label, sub="",
         fs=9.0, sub_fs=7.5, lw=1.2, rad=0.22, alpha=0.96, zorder=3):
    """Draw a rounded-corner box centred at (cx, cy)."""
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={rad}",
        linewidth=lw, edgecolor=C_EDGE,
        facecolor=fc, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(patch)
    ty = cy + (0.14 if sub else 0)
    ax.text(cx, ty, label,
            ha="center", va="center", fontsize=fs,
            fontweight="bold", color="#1A1A2E", zorder=zorder + 1)
    if sub:
        ax.text(cx, cy - 0.20, sub,
                ha="center", va="center", fontsize=sub_fs,
                color="#444444", style="italic", zorder=zorder + 1)


def varrow(ax, x, y_top, y_bot, lw=1.8, color=C_ARROW):
    """Vertical downward arrow."""
    ax.annotate(
        "", xy=(x, y_bot), xytext=(x, y_top),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=12),
        zorder=6,
    )


def harrow(ax, x_start, x_end, y, lw=1.4, color=C_ARROW, rad=0.0):
    """Horizontal (or curved) arrow."""
    ax.annotate(
        "", xy=(x_end, y), xytext=(x_start, y),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=11,
                        connectionstyle=f"arc3,rad={rad}"),
        zorder=6,
    )


def diag_arrow(ax, x1, y1, x2, y2, lw=1.3, color=C_ARROW, rad=0.0):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=11,
                        connectionstyle=f"arc3,rad={rad}"),
        zorder=6,
    )


# ────────────────────────────────────────────────────────────────────────────
# 1. Title banner
# ────────────────────────────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch(
    (0.25, 11.10), 9.5, 0.72,
    boxstyle="round,pad=0.0,rounding_size=0.2",
    linewidth=0, facecolor=C_TITLE, alpha=1.0, zorder=3))
ax.text(5.0, 11.46, "PyPIELM  —  Package Architecture",
        ha="center", va="center", fontsize=12.5,
        fontweight="bold", color="white", zorder=4)

# ────────────────────────────────────────────────────────────────────────────
# 2. Main pipeline  (centre column, x = 5.0)
# ────────────────────────────────────────────────────────────────────────────
CX   = 5.0      # pipeline centre x
PW   = 5.4      # pipeline box width
PH   = 0.78     # pipeline box height (regular layers)
MH   = 1.55     # models box height

Y_DATA   = 9.65
Y_PDE    = 8.40
Y_CORE   = 7.15
Y_MODELS = 5.55   # centre of taller models box
Y_OUT    = 3.90
Y_API    = 2.75

# ── data layer ───────────────────────────────────────────────────────────────
rbox(ax, CX, Y_DATA, PW, PH, C_DATA,
     "pypielm.data",
     "Adapters (PINNacle · PDEBench · CSV · NPZ · Torch)   ·   "
     "Dataset   ·   auto_load()",
     fs=9.0, sub_fs=7.5)

# ── pde layer ────────────────────────────────────────────────────────────────
rbox(ax, CX, Y_PDE, PW, PH, C_PDE,
     "pypielm.pde",
     "autograd operators (∇, Δ, ∂ₜ)   ·   collocation points   ·   "
     "BC / IC constraints",
     fs=9.0, sub_fs=7.5)

# ── core layer ───────────────────────────────────────────────────────────────
rbox(ax, CX, Y_CORE, PW, PH, C_CORE,
     "pypielm.core",
     "ELMBase   ·   feature maps (tanh · Fourier · RBF)   ·   "
     "solver (ridge / lstsq)",
     fs=9.0, sub_fs=7.5)

# ── models layer  (taller, with sub-boxes) ──────────────────────────────────
# Draw the outer box (no label via rbox — we place the title manually)
ax.add_patch(FancyBboxPatch(
    (CX - PW / 2, Y_MODELS - MH / 2), PW, MH,
    boxstyle="round,pad=0.0,rounding_size=0.22",
    linewidth=1.5, edgecolor=C_EDGE,
    facecolor=C_MODEL, alpha=0.96, zorder=3))
# Title at the TOP of the models box (above sub-boxes)
ax.text(CX, Y_MODELS + MH / 2 - 0.27,
        "pypielm.models   (26+ variants  +  4 PINN baselines)",
        ha="center", va="center", fontsize=9.2,
        fontweight="bold", color="#1A1A2E", zorder=4)

# Sub-boxes for the 7 model categories inside the models box
SUB_LABELS = [
    "Vanilla\n& Core", "Bayesian", "Fourier\n(GFF)",
    "Curriculum", "Domain\nDecomp.", "Constrained",
    "PINN\nbaselines",
]
N_SUBS  = len(SUB_LABELS)
sub_margin = 0.25
sub_w  = (PW - sub_margin) / N_SUBS - 0.06
sub_h  = 0.68
# sub-boxes sit in the lower 2/3 of the models box
sub_y  = Y_MODELS - 0.25
sub_x0 = CX - PW / 2 + sub_margin / 2

for i, (lbl, col) in enumerate(zip(SUB_LABELS, SUB_COLS)):
    sx = sub_x0 + (i + 0.5) * (PW - sub_margin) / N_SUBS
    ax.add_patch(FancyBboxPatch(
        (sx - sub_w / 2, sub_y - sub_h / 2), sub_w, sub_h,
        boxstyle="round,pad=0.0,rounding_size=0.10",
        linewidth=0.8, edgecolor="#777777",
        facecolor=col, alpha=0.90, zorder=5))
    ax.text(sx, sub_y, lbl,
            ha="center", va="center", fontsize=6.3,
            color="#1A1A2E", zorder=6)

# ── output row  (metrics | io | visualization) ───────────────────────────────
OUT_W = 1.68
OUT_H = 0.68
OUT_XS = [CX - 1.80, CX, CX + 1.80]   # three boxes, evenly spaced

LAYER_LABELS = [
    ("pypielm.metrics",      "RMSE · Rel.L² · MAE · R²",      C_OUT),
    ("pypielm.io",           "ONNX · TorchScript · ckpt.", C_OUT),
    ("pypielm.visualization","solution · error · convergence",  C_VIZ),
]
for ox, (lbl, sub, col) in zip(OUT_XS, LAYER_LABELS):
    rbox(ax, ox, Y_OUT, OUT_W, OUT_H, col, lbl, sub, fs=7.8, sub_fs=6.5)

# ── unified API banner ────────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch(
    (CX - PW / 2, Y_API - 0.28), PW, 0.52,
    boxstyle="round,pad=0.0,rounding_size=0.15",
    linewidth=1.3, edgecolor="#7F8C8D",
    facecolor=C_API, alpha=0.97, zorder=3))
ax.text(CX, Y_API,
        "model.fit()     ·     model.predict()     ·     model.score()",
        ha="center", va="center", fontsize=8.4,
        color="#2C3E50", fontfamily="monospace", zorder=4)

# ────────────────────────────────────────────────────────────────────────────
# 3. External data sources  (right panel)
# ────────────────────────────────────────────────────────────────────────────
EX_CX  = 8.85
EX_W   = 1.85
EX_H   = 0.52
EX_FC  = C_EXT

ax.text(EX_CX, 10.40, "External Data Sources",
        ha="center", va="center", fontsize=7.8,
        fontweight="bold", color="#1A5276")

src_items = [
    (10.05, "PINNacle  (.dat)"),
    ( 9.55, "PDEBench  (.npz / .h5)"),
    ( 9.05, "CSV  /  NumPy  /  Torch"),
]
for ey, lbl in src_items:
    rbox(ax, EX_CX, ey, EX_W, EX_H * 0.85, EX_FC, lbl,
         fs=7.3, lw=0.9, rad=0.12)

# Bracket on left of external items
bx = EX_CX - EX_W / 2 - 0.12
ax.annotate("", xy=(bx, 8.78), xytext=(bx, 10.30),
            arrowprops=dict(arrowstyle="-", color="#1A5276", lw=1.0))

# Single arrow from group into right edge of data box
diag_arrow(ax,
           EX_CX - EX_W / 2 - 0.15, 9.55,
           CX + PW / 2 + 0.05, Y_DATA,
           color="#2980B9", lw=1.5)

# ────────────────────────────────────────────────────────────────────────────
# 4. CLI / YAML  (left panel)
# ────────────────────────────────────────────────────────────────────────────
CL_CX = 0.83
CL_W  = 1.45
CL_H  = 2.10

ax.add_patch(FancyBboxPatch(
    (CL_CX - CL_W / 2, Y_PDE - 0.42), CL_W, CL_H,
    boxstyle="round,pad=0.0,rounding_size=0.20",
    linewidth=1.3, edgecolor="#27AE60",
    facecolor=C_CLI, alpha=0.95, zorder=3))

ax.text(CL_CX, Y_PDE + 1.40, "CLI  /  YAML",
        ha="center", va="center", fontsize=8.5,
        fontweight="bold", color="#1E8449", zorder=4)
ax.text(CL_CX, Y_PDE + 1.05, "__main__.py",
        ha="center", va="center", fontsize=7.0,
        color="#555", style="italic", zorder=4)

for i, cmd in enumerate(["pypielm run", "pypielm sweep",
                          "pypielm export", "pypielm list-models"]):
    ax.text(CL_CX, Y_PDE + 0.68 - i * 0.29, cmd,
            ha="center", va="center", fontsize=6.7,
            color="#145A32", fontfamily="monospace", zorder=4)

# config.yaml chip
yaml_y = Y_PDE - 0.62
ax.add_patch(FancyBboxPatch(
    (CL_CX - 0.70, yaml_y - 0.20), 1.40, 0.38,
    boxstyle="round,pad=0.0,rounding_size=0.10",
    linewidth=0.9, edgecolor="#27AE60",
    facecolor="#D5F5E3", alpha=0.95, zorder=4))
ax.text(CL_CX, yaml_y, "config.yaml",
        ha="center", va="center", fontsize=7.2,
        fontfamily="monospace", color="#145A32", zorder=5)

# Arrow: CLI box → left edge of data box
harrow(ax,
       CL_CX + CL_W / 2 + 0.05, CX - PW / 2 - 0.05,
       Y_DATA,
       color="#27AE60", lw=1.5)

# ────────────────────────────────────────────────────────────────────────────
# 5. PyPIELM-App (Streamlit, bottom-left)
# ────────────────────────────────────────────────────────────────────────────
APP_CX = 1.45
APP_Y  = Y_OUT
rbox(ax, APP_CX, APP_Y, 1.75, 0.65, "#FADBD8",
     "PyPIELM-App", "Streamlit GUI",
     fs=8.0, sub_fs=7.0, lw=1.1, rad=0.18, zorder=3)

# Curved arrow from models bottom-left corner to app
diag_arrow(ax,
           CX - PW / 2, Y_MODELS - MH / 2,
           APP_CX + 0.88 + 0.05, APP_Y + 0.05,
           color="#C0392B", lw=1.2, rad=-0.25)

# ────────────────────────────────────────────────────────────────────────────
# 6. Pipeline arrows  (vertical flow)
# ────────────────────────────────────────────────────────────────────────────
varrow(ax, CX, Y_DATA   - PH / 2 - 0.02, Y_PDE   + PH / 2 + 0.02)
varrow(ax, CX, Y_PDE    - PH / 2 - 0.02, Y_CORE  + PH / 2 + 0.02)
varrow(ax, CX, Y_CORE   - PH / 2 - 0.02, Y_MODELS + MH / 2 + 0.02)

# Fan arrows from models bottom to each output box
for ox in OUT_XS:
    diag_arrow(ax,
               ox, Y_MODELS - MH / 2 - 0.02,
               ox, Y_OUT    + OUT_H / 2 + 0.02,
               lw=1.4)

# Arrows from output boxes down to API banner
for ox in OUT_XS:
    varrow(ax, ox, Y_OUT - OUT_H / 2 - 0.02, Y_API + 0.28 + 0.02,
           lw=1.1, color="#7F8C8D")

# ────────────────────────────────────────────────────────────────────────────
# 7. Layer labels  (left margin, rotated)
# ────────────────────────────────────────────────────────────────────────────
for y, txt, col in [
    (Y_DATA,   "DATA",    "#2471A3"),
    (Y_PDE,    "PDE",     "#7D3C98"),
    (Y_CORE,   "CORE",    "#BA4A00"),
    (Y_MODELS, "MODELS",  "#1E8449"),
    (Y_OUT,    "OUTPUT",  "#9A7D0A"),
]:
    ax.text(0.18, y, txt,
            ha="center", va="center", fontsize=6.2,
            color=col, fontweight="bold", rotation=90)

# ────────────────────────────────────────────────────────────────────────────
# 8. Model-category legend
# ────────────────────────────────────────────────────────────────────────────
ax.text(5.0, 2.08, "Model categories:",
        ha="center", va="center", fontsize=7.5, color="#444", style="italic")

leg_y   = 1.60
leg_w   = 1.05
leg_h   = 0.34
leg_x0  = CX - (N_SUBS * leg_w) / 2 + leg_w / 2

for i, (lbl, col) in enumerate(zip(SUB_LABELS, SUB_COLS)):
    lx = leg_x0 + i * leg_w
    # Squash newlines for the legend
    short = lbl.replace("\n", "/")
    ax.add_patch(FancyBboxPatch(
        (lx - leg_w / 2 + 0.04, leg_y - leg_h / 2), leg_w - 0.08, leg_h,
        boxstyle="round,pad=0.0,rounding_size=0.06",
        linewidth=0.7, edgecolor="#888",
        facecolor=col, alpha=0.88, zorder=3))
    ax.text(lx, leg_y, short,
            ha="center", va="center", fontsize=6.0,
            color="#1A1A2E", zorder=4)

# ────────────────────────────────────────────────────────────────────────────
# 9. Footer
# ────────────────────────────────────────────────────────────────────────────
ax.text(5.0, 1.03,
        "scikit-learn-compatible  fit / predict / score  interface  ·  "
        "YAML-driven reproducible experiments  ·  ONNX / TorchScript export",
        ha="center", va="center", fontsize=7.0,
        color="#666", style="italic")

# ────────────────────────────────────────────────────────────────────────────
# Save
# ────────────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0)

for fmt in ("pdf", "png"):
    out = OUT / f"fig1_architecture.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved: {out}")

plt.close(fig)
