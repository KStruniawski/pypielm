"""Visualisation utilities (requires ``pip install pypielm[viz]``).

Public surface::

    from pypielm.visualization import (
        plot_solution_1d,
        plot_solution_2d,
        plot_training_history,
        plot_pareto,
        plot_leaderboard_heatmap,
        save_figure,
    )
"""

from __future__ import annotations

from .plots import (
    plot_leaderboard_heatmap,
    plot_pareto,
    plot_solution_1d,
    plot_solution_2d,
    plot_training_history,
    save_figure,
)

__all__ = [
    "plot_solution_1d",
    "plot_solution_2d",
    "plot_training_history",
    "plot_pareto",
    "plot_leaderboard_heatmap",
    "save_figure",
]
