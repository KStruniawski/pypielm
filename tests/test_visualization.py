"""Tests for pypielm.visualization.plots.

All tests run headlessly (Agg backend enforced inside the module).
No display server required.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

# Force headless backend before any matplotlib import
os.environ.setdefault("MPLBACKEND", "Agg")

from pypielm.visualization import (
    plot_leaderboard_heatmap,
    plot_pareto,
    plot_solution_1d,
    plot_solution_2d,
    plot_training_history,
    save_figure,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def x1d() -> np.ndarray:
    return np.linspace(0, 1, 50)


@pytest.fixture()
def u_pred_1d(x1d) -> np.ndarray:
    return np.sin(np.pi * x1d)


@pytest.fixture()
def u_true_1d(x1d) -> np.ndarray:
    return np.sin(np.pi * x1d) + 0.02 * np.random.default_rng(0).standard_normal(50)


@pytest.fixture()
def x2d() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.uniform(0, 1, (200, 2))


@pytest.fixture()
def u_pred_2d(x2d) -> np.ndarray:
    return np.sin(np.pi * x2d[:, 0]) * np.sin(np.pi * x2d[:, 1])


@pytest.fixture()
def u_true_2d(u_pred_2d) -> np.ndarray:
    rng = np.random.default_rng(1)
    return u_pred_2d + 0.01 * rng.standard_normal(len(u_pred_2d))


# ---------------------------------------------------------------------------
# plot_solution_1d
# ---------------------------------------------------------------------------

class TestPlotSolution1D:
    def test_returns_figure(self, x1d, u_pred_1d):
        import matplotlib.figure
        fig = plot_solution_1d(x1d, u_pred_1d)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_reference(self, x1d, u_pred_1d, u_true_1d):
        fig = plot_solution_1d(x1d, u_pred_1d, u_true_1d)
        ax = fig.axes[0]
        # predicted + reference + error band = at least 2 line artists
        assert len(ax.lines) >= 2

    def test_accepts_tensor(self, x1d, u_pred_1d):
        x_t = torch.tensor(x1d)
        u_t = torch.tensor(u_pred_1d)
        fig = plot_solution_1d(x_t, u_t)
        assert fig is not None

    def test_unsorted_x(self, x1d, u_pred_1d):
        rng = np.random.default_rng(7)
        idx = rng.permutation(len(x1d))
        fig = plot_solution_1d(x1d[idx], u_pred_1d[idx])
        ax = fig.axes[0]
        # x data on the line should be sorted
        x_plotted = ax.lines[0].get_xdata()
        assert np.all(np.diff(x_plotted) >= 0)

    def test_custom_labels_and_title(self, x1d, u_pred_1d):
        fig = plot_solution_1d(
            x1d, u_pred_1d, xlabel="t", ylabel="T(t)", title="Heat"
        )
        ax = fig.axes[0]
        assert ax.get_xlabel() == "t"
        assert ax.get_ylabel() == "T(t)"
        assert ax.get_title() == "Heat"

    def test_accepts_existing_ax(self, x1d, u_pred_1d):
        import matplotlib.pyplot as plt
        _, ax_ext = plt.subplots()
        fig = plot_solution_1d(x1d, u_pred_1d, ax=ax_ext)
        assert fig is ax_ext.get_figure()
        plt.close("all")


# ---------------------------------------------------------------------------
# plot_solution_2d
# ---------------------------------------------------------------------------

class TestPlotSolution2D:
    def test_returns_figure_no_ref(self, x2d, u_pred_2d):
        import matplotlib.figure
        fig = plot_solution_2d(x2d, u_pred_2d)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_reference_three_panels(self, x2d, u_pred_2d, u_true_2d):
        fig = plot_solution_2d(x2d, u_pred_2d, u_true_2d)
        # Should have 3 axes (pred + ref + error) each with colorbar axes
        # At least 3 main subplot axes
        assert len(fig.axes) >= 3

    def test_accepts_tensor(self, x2d, u_pred_2d):
        x_t = torch.tensor(x2d)
        u_t = torch.tensor(u_pred_2d)
        fig = plot_solution_2d(x_t, u_t)
        assert fig is not None

    def test_suptitle(self, x2d, u_pred_2d):
        fig = plot_solution_2d(x2d, u_pred_2d, title="Poisson 2D")
        assert "Poisson 2D" in fig.texts[0].get_text()


# ---------------------------------------------------------------------------
# plot_training_history
# ---------------------------------------------------------------------------

class TestPlotTrainingHistory:
    def test_returns_figure(self):
        import matplotlib.figure
        losses = {"total": [1.0, 0.5, 0.2, 0.1], "pde": [0.8, 0.4, 0.15, 0.07]}
        fig = plot_training_history(losses)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_number_of_lines(self):
        losses = {"a": [1.0, 0.5], "b": [0.8, 0.3], "c": [0.5, 0.1]}
        fig = plot_training_history(losses)
        ax = fig.axes[0]
        assert len(ax.lines) == 3

    def test_log_scale_enabled(self):
        losses = {"total": [1.0, 0.1, 0.01]}
        fig = plot_training_history(losses, log_scale=True)
        ax = fig.axes[0]
        assert ax.get_yscale() == "log"

    def test_linear_scale(self):
        losses = {"total": [1.0, 0.9, 0.8]}
        fig = plot_training_history(losses, log_scale=False)
        ax = fig.axes[0]
        assert ax.get_yscale() == "linear"

    def test_accepts_existing_ax(self):
        import matplotlib.pyplot as plt
        _, ax_ext = plt.subplots()
        losses = {"total": [1.0, 0.5]}
        fig = plot_training_history(losses, ax=ax_ext)
        assert fig is ax_ext.get_figure()
        plt.close("all")

    def test_single_curve(self):
        losses = {"loss": list(np.linspace(1, 0.001, 100))}
        fig = plot_training_history(losses)
        ax = fig.axes[0]
        assert len(ax.lines) == 1

    def test_epochs_on_x_axis(self):
        losses = {"total": [1.0, 0.5, 0.25]}
        fig = plot_training_history(losses, log_scale=False)
        ax = fig.axes[0]
        x_data = ax.lines[0].get_xdata()
        np.testing.assert_array_equal(x_data, [1, 2, 3])


# ---------------------------------------------------------------------------
# plot_pareto
# ---------------------------------------------------------------------------

class TestPlotPareto:
    @pytest.fixture()
    def results(self):
        return [
            {"model": "VanillaPIELM", "fit_time_s": 0.1, "rel_l2": 1e-2},
            {"model": "CorePIELM",    "fit_time_s": 0.5, "rel_l2": 1e-3},
            {"model": "BPIELM",       "fit_time_s": 2.0, "rel_l2": 5e-4},
            {"model": "VanillaPINN",  "fit_time_s": 10.0, "rel_l2": 2e-3},
        ]

    def test_returns_figure(self, results):
        import matplotlib.figure
        fig = plot_pareto(results)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_pareto_front_line_present(self, results):
        fig = plot_pareto(results)
        ax = fig.axes[0]
        # Pareto front is a dashed red line
        dashed_lines = [line for line in ax.lines if line.get_linestyle() in ("--", "dashed")]
        assert len(dashed_lines) >= 1

    def test_log_y_scale(self, results):
        fig = plot_pareto(results, log_y=True)
        ax = fig.axes[0]
        assert ax.get_yscale() == "log"

    def test_log_x_scale(self, results):
        fig = plot_pareto(results, log_x=True)
        ax = fig.axes[0]
        assert ax.get_xscale() == "log"

    def test_custom_metrics(self):
        results = [
            {"m": "A", "memory_mb": 10, "rmse": 0.5},
            {"m": "B", "memory_mb": 50, "rmse": 0.1},
        ]
        fig = plot_pareto(
            results, x_metric="memory_mb", y_metric="rmse",
            label_key="m", log_y=False,
        )
        ax = fig.axes[0]
        assert ax.get_xlabel() == "memory_mb"
        assert ax.get_ylabel() == "rmse"

    def test_single_point_no_crash(self):
        results = [{"model": "A", "fit_time_s": 1.0, "rel_l2": 1e-3}]
        fig = plot_pareto(results)
        assert fig is not None

    def test_accepts_existing_ax(self, results):
        import matplotlib.pyplot as plt
        _, ax_ext = plt.subplots()
        fig = plot_pareto(results, ax=ax_ext)
        assert fig is ax_ext.get_figure()
        plt.close("all")


# ---------------------------------------------------------------------------
# plot_leaderboard_heatmap
# ---------------------------------------------------------------------------

class TestPlotLeaderboardHeatmap:
    @pytest.fixture()
    def matrix(self):
        # 3 models × 4 tasks
        return np.array([
            [1e-2, 2e-2, 5e-3, 1e-3],
            [5e-3, 1e-2, 3e-3, 8e-4],
            [2e-2, 3e-2, 8e-3, 2e-3],
        ])

    def test_returns_figure(self, matrix):
        import matplotlib.figure
        fig = plot_leaderboard_heatmap(matrix)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_colorbar_label(self, matrix):
        fig = plot_leaderboard_heatmap(matrix, metric="RMSE")
        # colorbar label should contain metric name
        cbar = fig.axes[0].images[0].colorbar
        assert "RMSE" in cbar.ax.get_ylabel()

    def test_accepts_pandas_dataframe(self, matrix):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            matrix,
            index=["ModelA", "ModelB", "ModelC"],
            columns=["Poisson1D", "Poisson2D", "Heat1D", "Burgers1D"],
        )
        fig = plot_leaderboard_heatmap(df)
        ax = fig.axes[0]
        ylabels = [t.get_text() for t in ax.get_yticklabels()]
        assert "ModelA" in ylabels

    def test_cells_annotated(self, matrix):
        fig = plot_leaderboard_heatmap(matrix)
        ax = fig.axes[0]
        texts = [t.get_text() for t in ax.texts]
        # Each cell should be annotated: 3 × 4 = 12 annotations
        assert len(texts) == 12

    def test_title_set(self, matrix):
        fig = plot_leaderboard_heatmap(matrix, title="Results Table")
        ax = fig.axes[0]
        assert ax.get_title() == "Results Table"

    def test_list_of_lists_input(self):
        data = [[0.01, 0.02], [0.005, 0.01]]
        fig = plot_leaderboard_heatmap(data)
        assert fig is not None


# ---------------------------------------------------------------------------
# save_figure
# ---------------------------------------------------------------------------

class TestSaveFigure:
    def test_saves_png(self, x1d, u_pred_1d, tmp_path):
        fig = plot_solution_1d(x1d, u_pred_1d)
        out = tmp_path / "test.png"
        save_figure(fig, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_saves_pdf(self, x1d, u_pred_1d, tmp_path):
        fig = plot_solution_1d(x1d, u_pred_1d)
        out = tmp_path / "test.pdf"
        save_figure(fig, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_saves_svg(self, x1d, u_pred_1d, tmp_path):
        fig = plot_solution_1d(x1d, u_pred_1d)
        out = tmp_path / "test.svg"
        save_figure(fig, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_dpi_respected(self, x1d, u_pred_1d, tmp_path):
        fig = plot_solution_1d(x1d, u_pred_1d)
        out_lo = tmp_path / "lo.png"
        out_hi = tmp_path / "hi.png"
        save_figure(fig, out_lo, dpi=72)
        save_figure(fig, out_hi, dpi=300)
        # Higher DPI → larger file
        assert out_hi.stat().st_size >= out_lo.stat().st_size

    def test_accepts_string_path(self, x1d, u_pred_1d, tmp_path):
        fig = plot_solution_1d(x1d, u_pred_1d)
        out = str(tmp_path / "str_path.png")
        save_figure(fig, out)
        assert Path(out).exists()
