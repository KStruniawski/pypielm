"""Matplotlib-based visualisation utilities.

All functions accept :class:`torch.Tensor` or NumPy arrays and produce
:class:`matplotlib.figure.Figure` objects that can be shown interactively,
saved, or embedded in notebooks.

Requires: ``pip install pypielm[viz]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_numpy(a: Any) -> np.ndarray:
    """Convert Tensor, list, or ndarray to 1-D / N-D numpy array."""
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=float).squeeze()


def _import_matplotlib() -> Any:
    """Return the ``matplotlib.pyplot`` module (deferred import)."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless-safe; no-op if already set
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualisation. "
            "Install it with: pip install pypielm[viz]"
        ) from exc


# ---------------------------------------------------------------------------
# plot_solution_1d
# ---------------------------------------------------------------------------

def plot_solution_1d(
    x: Any,
    u_pred: Any,
    u_true: Any | None = None,
    *,
    xlabel: str = "x",
    ylabel: str = "u(x)",
    title: str = "Solution",
    figsize: tuple[float, float] = (7, 4),
    ax: Any = None,
) -> Any:
    """Plot a 1D PDE solution (predicted vs. reference).

    Args:
        x: Coordinate array, shape ``(N,)``.
        u_pred: Predicted solution, shape ``(N,)``.
        u_true: Optional reference solution for comparison.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Figure title.
        figsize: Figure size in inches (ignored when *ax* is provided).
        ax: Existing :class:`~matplotlib.axes.Axes` to draw on.
            If ``None`` a new figure is created.

    Returns:
        :class:`matplotlib.figure.Figure`.
    """
    plt = _import_matplotlib()

    x_np = _to_numpy(x)
    u_pred_np = _to_numpy(u_pred)

    sort_idx = np.argsort(x_np)
    x_np = x_np[sort_idx]
    u_pred_np = u_pred_np[sort_idx]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.plot(x_np, u_pred_np, label="Predicted", linewidth=2)

    if u_true is not None:
        u_true_np = _to_numpy(u_true)[sort_idx]
        ax.plot(x_np, u_true_np, "--", label="Reference", linewidth=2)
        err = np.abs(u_pred_np - u_true_np)
        ax.fill_between(x_np, u_pred_np - err, u_pred_np + err,
                        alpha=0.15, label="Abs. error band")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# plot_solution_2d
# ---------------------------------------------------------------------------

def plot_solution_2d(
    x: Any,
    u_pred: Any,
    u_true: Any | None = None,
    *,
    nx: int = 64,
    ny: int = 64,
    cmap: str = "viridis",
    title: str = "Solution",
    figsize: tuple[float, float] = (12, 4),
    fig: Any = None,
) -> Any:
    """Plot a 2D PDE solution as colour maps.

    Args:
        x: Coordinate array, shape ``(N, 2)``.  Columns are ``[x, y]``.
        u_pred: Predicted solution, shape ``(N,)`` or ``(N, 1)``.
        u_true: Optional reference solution, same shape as *u_pred*.
        nx: Grid resolution along x-axis for interpolation.
        ny: Grid resolution along y-axis for interpolation.
        cmap: Colour map name.
        title: Figure suptitle.
        figsize: Figure size in inches (ignored when *fig* is provided).
        fig: Existing :class:`~matplotlib.figure.Figure`.
            If ``None`` a new figure is created.

    Returns:
        :class:`matplotlib.figure.Figure`.
    """
    plt = _import_matplotlib()
    from scipy.interpolate import griddata

    xy = np.asarray(x, dtype=float)
    if isinstance(x, torch.Tensor):
        xy = x.detach().cpu().numpy()
    xy = xy.reshape(-1, 2)

    u_pred_np = _to_numpy(u_pred)
    has_ref = u_true is not None

    n_cols = 3 if has_ref else 1
    if fig is None:
        fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    else:
        axes = fig.get_axes()
        if not hasattr(axes, "__len__"):
            axes = [axes]

    if n_cols == 1:
        axes = [axes] if not hasattr(axes, "__len__") else axes

    x_lin = np.linspace(xy[:, 0].min(), xy[:, 0].max(), nx)
    y_lin = np.linspace(xy[:, 1].min(), xy[:, 1].max(), ny)
    X_grid, Y_grid = np.meshgrid(x_lin, y_lin)
    pts = np.column_stack([xy[:, 0], xy[:, 1]])

    U_pred_grid = griddata(pts, u_pred_np, (X_grid, Y_grid), method="linear")

    def _plot_panel(ax: Any, Z: Any, panel_title: str) -> None:
        im = ax.pcolormesh(X_grid, Y_grid, Z, cmap=cmap, shading="auto")
        ax.set_title(panel_title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if has_ref:
        u_true_np = _to_numpy(u_true)
        U_true_grid = griddata(pts, u_true_np, (X_grid, Y_grid), method="linear")
        U_err_grid = np.abs(U_pred_grid - U_true_grid)
        ax_list = list(axes)
        _plot_panel(ax_list[0], U_pred_grid, "Predicted")
        _plot_panel(ax_list[1], U_true_grid, "Reference")
        _plot_panel(ax_list[2], U_err_grid, "Abs. error")
    else:
        ax_list = list(axes)
        _plot_panel(ax_list[0], U_pred_grid, "Predicted")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# plot_training_history
# ---------------------------------------------------------------------------

def plot_training_history(
    losses: dict[str, list[float]],
    *,
    log_scale: bool = True,
    title: str = "Training History",
    figsize: tuple[float, float] = (7, 4),
    ax: Any = None,
) -> Any:
    """Plot training loss curves.

    Args:
        losses: Dict mapping loss component names to lists of per-epoch values.
            Example: ``{"total": [...], "pde": [...], "bc": [...]}``.
        log_scale: If ``True``, use log scale on the y-axis.
        title: Figure title.
        figsize: Figure size in inches (ignored when *ax* is provided).
        ax: Existing :class:`~matplotlib.axes.Axes`.

    Returns:
        :class:`matplotlib.figure.Figure`.
    """
    plt = _import_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    for name, values in losses.items():
        epochs = np.arange(1, len(values) + 1)
        ax.plot(epochs, values, label=name, linewidth=2)

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# plot_pareto
# ---------------------------------------------------------------------------

def _pareto_front(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Return boolean mask of Pareto-optimal points (minimise both axes)."""
    mask = np.zeros(len(xs), dtype=bool)
    for i in range(len(xs)):
        dominated = False
        for j in range(len(xs)):
            if i == j:
                continue
            if xs[j] <= xs[i] and ys[j] <= ys[i] and (xs[j] < xs[i] or ys[j] < ys[i]):
                dominated = True
                break
        mask[i] = not dominated
    return mask


def plot_pareto(
    results: list[dict[str, Any]],
    *,
    x_metric: str = "fit_time_s",
    y_metric: str = "rel_l2",
    label_key: str = "model",
    log_x: bool = False,
    log_y: bool = True,
    figsize: tuple[float, float] = (8, 5),
    ax: Any = None,
) -> Any:
    """Pareto-front scatter plot: accuracy vs. runtime.

    Args:
        results: List of dicts; each must contain *x_metric*, *y_metric*,
            and *label_key*.
        x_metric: Column name for the x-axis (default: fit time).
        y_metric: Column name for the y-axis (default: relative L² error).
        label_key: Column name used as point labels.
        log_x: Log scale on x-axis.
        log_y: Log scale on y-axis.
        figsize: Figure size (ignored when *ax* is provided).
        ax: Existing :class:`~matplotlib.axes.Axes`.

    Returns:
        :class:`matplotlib.figure.Figure`.
    """
    plt = _import_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    xs = np.array([float(r[x_metric]) for r in results])
    ys = np.array([float(r[y_metric]) for r in results])
    labels = [str(r[label_key]) for r in results]

    ax.scatter(xs, ys, zorder=3)
    for xi, yi, lbl in zip(xs, ys, labels, strict=False):
        ax.annotate(lbl, (xi, yi), textcoords="offset points",
                    xytext=(5, 3), fontsize=8)

    # Draw Pareto front
    if len(xs) > 1:
        mask = _pareto_front(xs, ys)
        px, py = xs[mask], ys[mask]
        sort_idx = np.argsort(px)
        ax.plot(px[sort_idx], py[sort_idx], "r--", linewidth=1.5,
                label="Pareto front", zorder=2)

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title("Accuracy vs. Runtime (Pareto)")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# plot_leaderboard_heatmap
# ---------------------------------------------------------------------------

def plot_leaderboard_heatmap(
    df: Any,
    *,
    metric: str = "rel_l2",
    figsize: tuple[float, float] = (12, 6),
    cmap: str = "YlOrRd_r",
    title: str = "Leaderboard Heatmap",
    fig: Any = None,
) -> Any:
    """Heatmap of model × task performance.

    Args:
        df: A 2-D structure (``numpy.ndarray``, ``list[list]``, or
            :class:`pandas.DataFrame`) with models as rows and tasks as
            columns, pre-aggregated to *metric* values.
            When a ``DataFrame`` is supplied, row/column labels are used.
        metric: Metric name used for the colour-bar label.
        figsize: Figure size (ignored when *fig* is provided).
        cmap: Colour map (default green = good: ``YlOrRd_r``).
        title: Figure title.
        fig: Existing :class:`~matplotlib.figure.Figure`.

    Returns:
        :class:`matplotlib.figure.Figure`.
    """
    plt = _import_matplotlib()

    # Extract numpy matrix + optional row/col labels
    row_labels: list[str] | None = None
    col_labels: list[str] | None = None

    try:
        import pandas as pd  # optional
        if isinstance(df, pd.DataFrame):
            row_labels = list(df.index.astype(str))
            col_labels = list(df.columns.astype(str))
            data = df.to_numpy(dtype=float)
        else:
            raise TypeError
    except (ImportError, TypeError):
        data = np.asarray(df, dtype=float)

    n_rows, n_cols = data.shape

    if fig is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = fig.axes[0] if fig.axes else fig.add_subplot(111)

    im = ax.imshow(data, cmap=cmap, aspect="auto")
    fig.colorbar(im, ax=ax, label=metric)

    if row_labels is not None:
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels, fontsize=9)
    if col_labels is not None:
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)

    # Annotate cells with values
    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2e}", ha="center", va="center",
                        fontsize=7, color="black")

    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# save_figure
# ---------------------------------------------------------------------------

def save_figure(
    fig: Any,
    path: str | Path,
    *,
    dpi: int = 300,
    bbox_inches: str = "tight",
) -> None:
    """Save a :class:`matplotlib.figure.Figure` to *path*.

    Args:
        fig: The figure to save.
        path: Output file path.  Extension determines format (``.pdf``,
            ``.png``, ``.svg``, …).
        dpi: Resolution (dots per inch).
        bbox_inches: Passed verbatim to :func:`~matplotlib.figure.Figure.savefig`.
    """
    fig.savefig(str(path), dpi=dpi, bbox_inches=bbox_inches)

