"""fig5_cd_diagram.py
===================
WP2.5 — Figure 5: Critical-Difference Diagram (Nemenyi post-hoc test).

Two-panel stacked figure (full double-column width per panel):
  (a) PIELM variants   – MPS run, 5 models, n_tasks=3, n_seeds=3
  (b) PINN baselines   – CPU run, 5 models, n_tasks=5, n_seeds=3

Publication spec (SoftwareX):
  - Double-column width: 7.01 in  (each panel uses full width)
  - 12 pt serif font throughout
  - No figure title; panel labels (a)/(b) in bold
  - 300 dpi, exported as PDF + PNG
"""

import math
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[2]
STATS_PIELM = ROOT / "PyPIELM/benchmarks/results/mps/20260509T043038Z_stats_summary.json"
STATS_PINN  = ROOT / "PyPIELM/benchmarks/results/cpu/20260509T113532956883Z_stats_summary.json"
OUT_DIR     = Path(__file__).parent

# ---------------------------------------------------------------------------
# Display-name mapping
# ---------------------------------------------------------------------------
DISPLAY = {
    "bayesian_pielm":         "Bayesian PIELM",
    "curriculum_pielm":       "Curriculum PIELM",
    "ddelm_coarse":           "DD-ELM",
    "gff_pielm":              "GFF-PIELM",
    "vanilla_pielm":          "Vanilla PIELM",
    "vanilla_pinn":           "Vanilla PINN",
    "adaptive_pinn":          "Adaptive PINN",
    "residual_adaptive_pinn": "Res.-Adaptive PINN",
    "muon_pinn":              "Muon PINN",
    "fourier_pinn":           "Fourier PINN",
}

# ---------------------------------------------------------------------------
# Nemenyi CD formula
# ---------------------------------------------------------------------------
_Q_TABLE = {
    (0.05, 2): 1.960, (0.05, 3): 2.343, (0.05, 4): 2.569,
    (0.05, 5): 2.728, (0.05, 6): 2.850, (0.05, 7): 2.949,
    (0.05, 8): 3.031, (0.05, 9): 3.102, (0.05, 10): 3.164,
    (0.10, 2): 1.645, (0.10, 3): 2.052, (0.10, 4): 2.291,
    (0.10, 5): 2.459, (0.10, 6): 2.589, (0.10, 7): 2.693,
    (0.10, 8): 2.780, (0.10, 9): 2.855, (0.10, 10): 2.920,
}


def _compute_cd(n_models, n_tasks, alpha=0.05):
    q = _Q_TABLE.get(
        (alpha, min(n_models, 10)),
        1.960 + 0.47 * math.log(max(n_models, 2)),
    )
    return q * math.sqrt(n_models * (n_models + 1) / (6 * n_tasks))


def _maximal_ns_groups(ranks, cd):
    n = len(ranks)
    groups = []
    for i in range(n):
        for j in range(i + 1, n):
            if ranks[j] - ranks[i] < cd:
                groups.append((i, j))
            else:
                break
    maximal = []
    for (a, b) in groups:
        dominated = any(
            a2 <= a and b2 >= b and (a2, b2) != (a, b)
            for (a2, b2) in groups
        )
        if not dominated:
            maximal.append((a, b))
    return maximal


def _deflect_labels(ranks, half_widths):
    """Push label x-positions apart so no two overlap."""
    pos = list(ranks)
    for _ in range(500):
        changed = False
        for i in range(len(pos) - 1):
            gap_req = half_widths[i] + half_widths[i + 1] + 0.12
            if pos[i + 1] - pos[i] < gap_req:
                mid = (pos[i] + pos[i + 1]) / 2.0
                pos[i]     = mid - gap_req / 2.0
                pos[i + 1] = mid + gap_req / 2.0
                changed = True
        for i in range(len(pos) - 2, -1, -1):
            gap_req = half_widths[i] + half_widths[i + 1] + 0.12
            if pos[i + 1] - pos[i] < gap_req:
                mid = (pos[i] + pos[i + 1]) / 2.0
                pos[i]     = mid - gap_req / 2.0
                pos[i + 1] = mid + gap_req / 2.0
                changed = True
        if not changed:
            break
    return pos


# ---------------------------------------------------------------------------
# Core panel drawing
# ---------------------------------------------------------------------------

def draw_cd_panel(ax, avg_ranks, n_tasks, n_seeds, alpha=0.05,
                  panel_label="", fontsize=12, fig_width_in=7.01):
    models  = sorted(avg_ranks, key=avg_ranks.get)
    ranks   = [avg_ranks[m] for m in models]
    n_models = len(models)

    cd    = _compute_cd(n_models, n_tasks, alpha)
    r_min = min(ranks)
    r_max = max(ranks)
    pad   = 0.85

    xlim_lo    = r_min - pad
    xlim_hi    = r_max + pad
    xlim_range = xlim_hi - xlim_lo

    ax.set_xlim(xlim_lo, xlim_hi)
    ax.set_ylim(-2.6, 3.4)
    ax.axis("off")

    # rank axis
    ax.axhline(0.0, color="black", lw=0.9, zorder=2)

    # --- label half-width in rank units ---
    axes_w_in   = fig_width_in * 0.84
    rank_per_in = xlim_range / axes_w_in
    char_w_in   = (fontsize / 72.0) * 0.85

    def hw(label):
        return len(label) * char_w_in / 2.0 * rank_per_in

    # nudge identical ranks
    nudge_eps  = 0.06
    nudged     = list(ranks)
    for i in range(n_models):
        for j in range(i + 1, n_models):
            if abs(nudged[j] - nudged[i]) < nudge_eps * 0.4:
                nudged[i] -= nudge_eps / 2.0
                nudged[j] += nudge_eps / 2.0

    above_idx = [i for i in range(n_models) if i % 2 == 0]
    below_idx = [i for i in range(n_models) if i % 2 == 1]

    def place_group(indices):
        orig   = [nudged[i] for i in indices]
        labels = [DISPLAY.get(models[i], models[i]) for i in indices]
        hws    = [hw(l) for l in labels]
        order  = sorted(range(len(orig)), key=lambda k: orig[k])
        s_orig = [orig[order[k]] for k in range(len(orig))]
        s_hws  = [hws[order[k]] for k in range(len(hws))]
        defl   = _deflect_labels(s_orig, s_hws)
        result = {}
        for k, oi in enumerate(order):
            result[oi] = defl[k]
        return {indices[oi]: result[oi] for oi in range(len(indices))}

    above_lx = place_group(above_idx)
    below_lx = place_group(below_idx)

    label_info = {}
    for gi in above_idx:
        label_info[gi] = (nudged[gi], above_lx[gi], 0.90, "bottom")
    for gi in below_idx:
        label_info[gi] = (nudged[gi], below_lx[gi], -1.05, "top")

    # draw dots, connectors, labels
    arm_y_frac = 0.20
    for i, model in enumerate(models):
        dot_x, lx, ly, va = label_info[i]
        label = DISPLAY.get(model, model)

        ax.plot(dot_x, 0.0, "o", markersize=7, color="steelblue",
                zorder=5, clip_on=False)
        ax.plot([dot_x, dot_x], [-0.08, 0.08], color="black", lw=0.8, zorder=3)

        sign  = 1.0 if ly > 0 else -1.0
        arm_y = sign * arm_y_frac
        if abs(lx - dot_x) > 0.10:
            xs = [dot_x, dot_x, lx, lx]
            ys = [0.0, arm_y, arm_y, sign * abs(ly) * 0.72]
            ax.plot(xs, ys, color="gray", lw=0.7, zorder=1,
                    solid_capstyle="round", clip_on=False)
        else:
            ax.plot([dot_x, lx], [0.0, sign * abs(ly) * 0.72],
                    color="gray", lw=0.7, zorder=1, clip_on=False)

        ax.text(lx, ly, label, ha="center", va=va,
                fontsize=fontsize, fontfamily="serif", clip_on=False)

    # CD bracket
    cd_start  = r_min
    cd_end    = min(r_min + cd, xlim_hi - 0.05)
    bracket_y = 2.30
    ax.annotate("",
                xy=(cd_end, bracket_y), xytext=(cd_start, bracket_y),
                arrowprops={"arrowstyle": "<->", "color": "crimson", "lw": 1.6},
                annotation_clip=False)
    ax.text((cd_start + cd_end) / 2.0, bracket_y - 0.25,
            f"CD = {cd:.2f}",
            ha="center", va="top",
            fontsize=fontsize, fontfamily="serif", color="crimson")

    # non-significant clique bars
    ns_groups = _maximal_ns_groups(ranks, cd)
    bar_intervals = []
    cliq_y_base = 1.55
    for (i_start, i_end) in ns_groups:
        x0, x1 = ranks[i_start], ranks[i_end]
        y = cliq_y_base
        while any(abs(y - py) < 0.18 and not (x1 + 0.02 < px0 or x0 - 0.02 > px1)
                  for (px0, px1, py) in bar_intervals):
            y += 0.20
        bar_intervals.append((x0, x1, y))
        ax.plot([x0, x1], [y, y], color="gray", lw=4.5, alpha=0.50,
                solid_capstyle="round", zorder=3)

    # stats annotation
    ax.text(xlim_lo + 0.05, -2.35,
            f"$n_{{\\rm tasks}}={n_tasks}$, "
            f"$n_{{\\rm seeds}}={n_seeds}$, "
            f"$\\alpha = {alpha}$",
            fontsize=fontsize - 2, fontfamily="serif",
            color="dimgray", va="top", clip_on=False)

    # panel label
    if panel_label:
        ax.text(xlim_lo + 0.05, 3.25,
                panel_label,
                fontsize=fontsize, fontfamily="serif",
                fontweight="bold", va="top", clip_on=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with open(STATS_PIELM) as f:
        pielm = json.load(f)
    with open(STATS_PINN) as f:
        pinn = json.load(f)

    FS    = 12
    FIG_W = 7.01
    plt.rcParams.update({
        "font.family":    "serif",
        "font.size":       FS,
        "axes.labelsize":  FS,
        "xtick.labelsize": FS,
        "ytick.labelsize": FS,
        "legend.fontsize": FS,
    })

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(FIG_W, 5.2),
        gridspec_kw={"hspace": 0.12},
    )

    draw_cd_panel(ax_top,
                  avg_ranks=pielm["avg_ranks"],
                  n_tasks=pielm["n_tasks"],
                  n_seeds=pielm["n_seeds"],
                  panel_label="(a) PIELM variants",
                  fontsize=FS, fig_width_in=FIG_W)

    draw_cd_panel(ax_bot,
                  avg_ranks=pinn["avg_ranks"],
                  n_tasks=pinn["n_tasks"],
                  n_seeds=pinn["n_seeds"],
                  panel_label="(b) PINN baselines",
                  fontsize=FS, fig_width_in=FIG_W)

    for ext in ("pdf", "png"):
        path = OUT_DIR / f"fig5_cd_diagram.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
