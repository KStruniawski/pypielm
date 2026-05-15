"""Statistical analysis of benchmark results.

Loads JSON artefacts from ``benchmarks/results/`` and performs:

1. **Friedman test** — non-parametric rank-based test for differences across
   multiple models on multiple tasks (null: all models identical).
2. **Nemenyi post-hoc test** — pairwise comparison following a significant
   Friedman test, implemented via Wilcoxon rank-sum + Holm–Bonferroni
   correction (no external ``autorank``/``scikit-posthocs`` required).
3. **Critical-difference (CD) diagram** — ranked model comparison plot
   saved to ``benchmarks/results/<timestamp>_cd_diagram.png``.

Usage::

    cd PyPIELM

    # Analyse latest accuracy artefact automatically:
    python benchmarks/stats_analysis.py

    # Or specify explicit files:
    python benchmarks/stats_analysis.py \\
        --accuracy benchmarks/results/20240101T000000Z_accuracy.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

import sys
sys.path.insert(0, str(Path(__file__).parent))
from perf_profile import RESULTS_DIR, _save, _timestamp, get_results_dir, _platform_name

# ---------------------------------------------------------------------------
# Load utilities
# ---------------------------------------------------------------------------

def _latest(pattern: str, search_dir: Path | None = None) -> Path | None:
    """Return the most-recent file matching ``pattern`` in *search_dir* (default: RESULTS_DIR)."""
    d = search_dir or RESULTS_DIR
    matches = sorted(d.glob(pattern))
    return matches[-1] if matches else None


def load_accuracy(path: Path) -> dict[str, dict[str, list[float]]]:
    """Load accuracy JSON → ``{task: {model: [rel_l2_per_seed]}}``.

    Handles both ``AccuracyBenchmark`` output and sweep_hidden_dim output.
    """
    raw = json.loads(path.read_text())
    result: dict[str, dict[str, list[float]]] = {}
    for task_name, task_data in raw.items():
        if task_name in ("dims", "seeds", "task", "device"):
            continue
        if not isinstance(task_data, dict):
            continue
        result[task_name] = {}
        for model_name, mdata in task_data.items():
            if isinstance(mdata, dict) and "rel_l2_per_seed" in mdata:
                result[task_name][model_name] = [
                    v for v in mdata["rel_l2_per_seed"] if not math.isnan(v)
                ]
    return result


# ---------------------------------------------------------------------------
# Friedman test
# ---------------------------------------------------------------------------

def friedman_test(
    data: dict[str, dict[str, list[float]]],
) -> dict[str, Any]:
    """Run Friedman test across models for each task.

    ``data[task][model]`` must be a list of per-seed rel_l2 values.

    Returns::

        {task: {"statistic": float, "p_value": float, "significant": bool}}
    """
    results: dict[str, Any] = {}
    for task_name, models in data.items():
        model_names = sorted(models)
        # Align seeds: take minimum count to make rectangular
        min_seeds = min(len(models[m]) for m in model_names)
        if min_seeds < 2 or len(model_names) < 3:
            results[task_name] = {
                "error": "Need ≥3 models and ≥2 seeds for Friedman test",
                "n_models": len(model_names),
                "min_seeds": min_seeds,
            }
            continue
        matrix = np.array([models[m][:min_seeds] for m in model_names])  # (M, S)
        stat, p = friedmanchisquare(*[matrix[i] for i in range(len(model_names))])
        results[task_name] = {
            "models": model_names,
            "statistic": float(stat),
            "p_value": float(p),
            "significant_alpha05": bool(p < 0.05),
            "n_seeds": min_seeds,
        }
    return results


# ---------------------------------------------------------------------------
# Pairwise Wilcoxon + Holm–Bonferroni
# ---------------------------------------------------------------------------

def pairwise_wilcoxon(
    data: dict[str, dict[str, list[float]]],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Pairwise Wilcoxon signed-rank tests with Holm–Bonferroni correction.

    Returns::

        {task: {"{modelA} vs {modelB}": {"p_raw", "p_adj", "significant"}}}
    """
    from itertools import combinations

    results: dict[str, Any] = {}
    for task_name, models in data.items():
        model_names = sorted(models)
        min_seeds = min(len(models[m]) for m in model_names)
        pairs = list(combinations(model_names, 2))
        raw_ps = {}
        for a, b in pairs:
            xa = np.array(models[a][:min_seeds])
            xb = np.array(models[b][:min_seeds])
            try:
                _, p = wilcoxon(xa, xb)
            except Exception:
                p = 1.0
            raw_ps[f"{a} vs {b}"] = float(p)

        # Holm–Bonferroni correction
        sorted_pairs = sorted(raw_ps, key=raw_ps.get)
        n = len(sorted_pairs)
        task_results: dict[str, Any] = {}
        for rank, pair in enumerate(sorted_pairs):
            p_raw = raw_ps[pair]
            p_adj = min(p_raw * (n - rank), 1.0)
            task_results[pair] = {
                "p_raw": p_raw,
                "p_adj_holm": p_adj,
                "significant_alpha05": bool(p_adj < alpha),
            }
        results[task_name] = task_results
    return results


# ---------------------------------------------------------------------------
# Average ranks
# ---------------------------------------------------------------------------

def average_ranks(
    data: dict[str, dict[str, list[float]]],
) -> dict[str, float]:
    """Compute average cross-task rank for each model (lower = better).

    Uses mean rel_l2 per task to rank models (rank 1 = best).
    """
    all_models: set[str] = set()
    for models in data.values():
        all_models.update(models.keys())

    model_ranks: dict[str, list[float]] = {m: [] for m in all_models}
    for task_name, models in data.items():
        means = {m: float(np.nanmean(vs)) for m, vs in models.items() if vs}
        ranked = sorted(means, key=means.get)
        for r, m in enumerate(ranked, start=1):
            model_ranks[m].append(float(r))

    return {m: float(np.mean(vs)) for m, vs in model_ranks.items() if vs}


# ---------------------------------------------------------------------------
# CD diagram
# ---------------------------------------------------------------------------

def plot_cd_diagram(
    avg_ranks: dict[str, float],
    n_tasks: int,
    n_seeds: int,
    alpha: float = 0.05,
    save: bool = True,
    results_dir=None,
) -> None:
    """Plot a critical-difference diagram.

    Args:
        avg_ranks: ``{model_name: average_rank}`` (lower = better).
        n_tasks: Number of datasets/tasks (k in the Friedman test).
        n_seeds: Number of seeds used per model per task.
        alpha: Significance level for the CD threshold.
        save: Save PNG to ``benchmarks/results/``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import studentized_range

    models = sorted(avg_ranks, key=avg_ranks.get)
    ranks = [avg_ranks[m] for m in models]
    n_models = len(models)

    # Critical difference (Nemenyi test threshold)
    # CD = q_alpha * sqrt(k(k+1) / (6N))
    # q_alpha: critical value of the studentized range distribution
    # Approximation: q_alpha ≈ sqrt(2) * z_{alpha / (k(k-1)/2)} using Nemenyi
    # Standard approach: use tabulated q_alpha values
    _q_table = {
        (0.05, 2): 1.960, (0.05, 3): 2.343, (0.05, 4): 2.569,
        (0.05, 5): 2.728, (0.05, 6): 2.850, (0.05, 7): 2.949,
        (0.05, 8): 3.031, (0.05, 9): 3.102, (0.05, 10): 3.164,
        (0.10, 2): 1.645, (0.10, 3): 2.052, (0.10, 4): 2.291,
        (0.10, 5): 2.459, (0.10, 6): 2.589, (0.10, 7): 2.693,
        (0.10, 8): 2.780, (0.10, 9): 2.855, (0.10, 10): 2.920,
    }
    q_alpha = _q_table.get(
        (alpha, min(n_models, 10)),
        1.960 + 0.47 * math.log(max(n_models, 2))  # rough extrapolation
    )
    cd = q_alpha * math.sqrt(n_models * (n_models + 1) / (6 * n_tasks))

    fig, ax = plt.subplots(figsize=(max(8, n_models * 1.5), 4))
    ax.set_xlim(1 - 0.3, n_models + 0.3)
    ax.set_ylim(-1.5, 2.5)
    ax.set_xticks(range(1, n_models + 1))
    ax.set_xticklabels([str(i) for i in range(1, n_models + 1)])
    ax.set_xlabel("Average rank (lower is better)")
    ax.set_title("Critical-Difference Diagram (Nemenyi post-hoc test)")
    ax.axhline(0, color="black", linewidth=0.8)

    # Draw each model as a labeled point on the axis
    for i, (model, rank) in enumerate(zip(models, ranks)):
        ax.plot(rank, 0, "o", markersize=8, color="steelblue", zorder=5)
        # Alternate label height to avoid overlap
        y_txt = 0.6 if i % 2 == 0 else -0.8
        ax.annotate(
            model,
            xy=(rank, 0),
            xytext=(rank, y_txt),
            ha="center",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.7},
        )

    # Draw critical-difference bar at rank=1
    ax.annotate(
        "",
        xy=(1 + cd, 2.0),
        xytext=(1, 2.0),
        arrowprops={"arrowstyle": "<->", "color": "red", "lw": 1.5},
    )
    ax.text(1 + cd / 2, 2.2, f"CD={cd:.2f}", ha="center", color="red", fontsize=8)

    # Draw non-significant cliques (pairwise difference < CD)
    y_bar = 1.3
    for i, (ma, ra) in enumerate(zip(models, ranks)):
        for j, (mb, rb) in enumerate(zip(models, ranks)):
            if j <= i:
                continue
            if abs(ra - rb) < cd:
                ax.plot([ra, rb], [y_bar, y_bar], color="gray", lw=3, alpha=0.5)
        y_bar = 1.3  # keep on same row (simplification)

    ax.axis("off")
    ax.text(
        0.5, -1.3,
        f"n_tasks={n_tasks}, n_seeds={n_seeds}, α={alpha}",
        transform=ax.transData,
        ha="left",
        fontsize=8,
        color="dimgray",
    )

    plt.tight_layout()
    if save:
        rdir = results_dir or RESULTS_DIR
        rdir.mkdir(parents=True, exist_ok=True)
        path = rdir / f"{_timestamp()}_cd_diagram.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"CD diagram saved → {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def analyse(
    accuracy_path: Path | None = None,
    alpha: float = 0.05,
    save: bool = True,
    results_dir=None,
) -> dict[str, Any]:
    """Run the full statistical analysis pipeline.

    Args:
        accuracy_path: Path to an ``accuracy`` JSON artefact.  If ``None``,
            the most recent file matching ``*_accuracy.json`` is used.
        alpha: Significance level.
        save: Write JSON summary and CD-diagram PNG.
        results_dir: Directory for output files.  When *None*, uses ``RESULTS_DIR``.

    Returns:
        Dict with keys ``friedman``, ``pairwise``, ``avg_ranks``.
    """
    search_dir = results_dir or RESULTS_DIR
    if accuracy_path is None:
        accuracy_path = _latest("*_accuracy.json", search_dir)
    if accuracy_path is None:
        raise FileNotFoundError(
            f"No accuracy JSON found in {search_dir}. "
            "Run perf_profile.py first."
        )

    print(f"Loading accuracy data from: {accuracy_path}")
    data = load_accuracy(accuracy_path)

    if not data:
        raise ValueError("No task data found in accuracy file.")

    print(f"Tasks: {list(data.keys())}")
    all_models: set[str] = set()
    for td in data.values():
        all_models.update(td.keys())
    print(f"Models: {sorted(all_models)}")

    print("\n--- Friedman test ---")
    friedman = friedman_test(data)
    for task, res in friedman.items():
        if "error" in res:
            print(f"  {task}: {res['error']}")
        else:
            sig = "SIGNIFICANT" if res["significant_alpha05"] else "not significant"
            print(f"  {task}: χ²={res['statistic']:.3f}  p={res['p_value']:.4f}  "
                  f"[{sig}]")

    print("\n--- Pairwise Wilcoxon + Holm–Bonferroni ---")
    pairwise = pairwise_wilcoxon(data, alpha=alpha)
    for task, pairs in pairwise.items():
        sig_pairs = {p: v for p, v in pairs.items() if v["significant_alpha05"]}
        print(f"  {task}: {len(sig_pairs)}/{len(pairs)} pairs significant")
        for pair, v in sig_pairs.items():
            print(f"    * {pair}  p_adj={v['p_adj_holm']:.4f}")

    print("\n--- Average ranks ---")
    avg_r = average_ranks(data)
    for model, r in sorted(avg_r.items(), key=lambda x: x[1]):
        print(f"  rank {r:.2f}  {model}")

    # CD diagram
    n_tasks = len(data)
    n_seeds = min(
        min(len(vs) for vs in task_data.values())
        for task_data in data.values()
        if task_data
    )
    plot_cd_diagram(avg_r, n_tasks=n_tasks, n_seeds=n_seeds, alpha=alpha,
                    save=save, results_dir=results_dir)

    summary: dict[str, Any] = {
        "friedman": friedman,
        "pairwise_wilcoxon": pairwise,
        "avg_ranks": avg_r,
        "n_tasks": n_tasks,
        "n_seeds": n_seeds,
    }

    if save:
        _save(summary, "stats_summary", results_dir)

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Statistical analysis of benchmarks")
    parser.add_argument("--accuracy", type=Path, default=None,
                        help="Path to accuracy JSON (default: latest)")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument(
        "--platform", type=str, default=None,
        help="Results sub-folder to search and save into (e.g. 'mps', 'cpu', 'cuda').",
    )
    args = parser.parse_args()

    rdir = get_results_dir(args.platform) if args.platform else None

    analyse(
        accuracy_path=args.accuracy,
        alpha=args.alpha,
        save=not args.no_save,
        results_dir=rdir,
    )
    print("\nDone.")
