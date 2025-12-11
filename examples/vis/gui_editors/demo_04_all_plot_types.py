#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: examples/vis/gui_editors/demo_04_all_plot_types.py

"""
Demo 04: All plot types (4x4) - Comprehensive coverage

Port: 5054

Usage:
    ./demo_04_all_plot_types.py              # Flask backend (default)
    ./demo_04_all_plot_types.py --backend qt # Qt backend
"""

import numpy as np
from pathlib import Path
from typing import Literal
import scitex as stx
from scitex.plt.styles.presets import SCITEX_STYLE

Backend = Literal["auto", "flask", "dearpygui", "qt", "tkinter", "mpl"]
PORT = 5054


def create_figure(output_dir: Path) -> Path:
    """Create comprehensive figure covering all major plot types."""
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    # Generate sample data
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + 0.1 * np.random.randn(100)
    y_err = 0.2 + 0.1 * np.random.rand(100)
    scatter_x = np.random.randn(50)
    scatter_y = 0.8 * scatter_x + 0.5 * np.random.randn(50)
    groups = ["A", "B", "C", "D"]
    group_data = [np.random.normal(i, 0.5, 30) for i in range(4)]
    heatmap_data = np.random.randn(10, 10)

    STYLE = SCITEX_STYLE.copy()
    fig, axes = stx.plt.subplots(4, 4, **STYLE)

    # Row 1: Basic plots
    # (0,0) Line plot
    axes[0, 0].plot(x, y, "-", label="Line", id="line-plot")
    axes[0, 0].set_title("Line Plot")
    axes[0, 0].set_xlabel("X")
    axes[0, 0].set_ylabel("Y")

    # (0,1) Scatter plot
    axes[0, 1].scatter(
        scatter_x, scatter_y, alpha=0.6, s=30, id="scatter-plot"
    )
    axes[0, 1].set_title("Scatter Plot")
    axes[0, 1].set_xlabel("X")
    axes[0, 1].set_ylabel("Y")

    # (0,2) Bar plot
    bar_x = np.arange(len(groups))
    bar_y = [np.mean(d) for d in group_data]
    bar_err = [np.std(d) for d in group_data]
    axes[0, 2].bar(bar_x, bar_y, yerr=bar_err, capsize=3, id="bar-plot")
    axes[0, 2].set_xticks(bar_x)
    axes[0, 2].set_xticklabels(groups)
    axes[0, 2].set_title("Bar Plot")
    axes[0, 2].set_xlabel("Group")
    axes[0, 2].set_ylabel("Value")

    # (0,3) Histogram
    axes[0, 3].hist(
        np.concatenate(group_data), bins=20, alpha=0.7, id="histogram"
    )
    axes[0, 3].set_title("Histogram")
    axes[0, 3].set_xlabel("Value")
    axes[0, 3].set_ylabel("Count")

    # Row 2: Statistical plots
    # (1,0) Error bar
    axes[1, 0].errorbar(
        x[::10], y[::10], yerr=y_err[::10], fmt="o", capsize=3, id="errorbar"
    )
    axes[1, 0].set_title("Error Bar")
    axes[1, 0].set_xlabel("X")
    axes[1, 0].set_ylabel("Y")

    # (1,1) Fill between
    axes[1, 1].plot(x, y, "-", id="fill-line")
    axes[1, 1].fill_between(x, y - y_err, y + y_err, alpha=0.3, id="fill-area")
    axes[1, 1].set_title("Fill Between")
    axes[1, 1].set_xlabel("X")
    axes[1, 1].set_ylabel("Y")

    # (1,2) Box plot
    axes[1, 2].stx_box(group_data, labels=groups, id="boxplot")
    axes[1, 2].set_title("Box Plot")
    axes[1, 2].set_xlabel("Group")
    axes[1, 2].set_ylabel("Value")

    # (1,3) Violin plot
    axes[1, 3].stx_violin(group_data, labels=groups, id="violin")
    axes[1, 3].set_title("Violin Plot")
    axes[1, 3].set_xlabel("Group")
    axes[1, 3].set_ylabel("Value")

    # Row 3: Heatmaps and images
    # (2,0) Heatmap
    im = axes[2, 0].imshow(heatmap_data, cmap="viridis", id="heatmap")
    axes[2, 0].set_title("Heatmap")
    axes[2, 0].set_xlabel("Col")
    axes[2, 0].set_ylabel("Row")

    # (2,1) Contour
    xx, yy = np.meshgrid(np.linspace(-2, 2, 30), np.linspace(-2, 2, 30))
    zz = np.exp(-(xx**2 + yy**2))
    axes[2, 1].contour(xx, yy, zz, levels=8, id="contour")
    axes[2, 1].set_title("Contour")
    axes[2, 1].set_xlabel("X")
    axes[2, 1].set_ylabel("Y")

    # (2,2) Shaded line (mean±std)
    data_2d = np.random.randn(20, 50)
    mean_trace = data_2d.mean(axis=0)
    std_trace = data_2d.std(axis=0)
    trace_x = np.arange(50)
    axes[2, 2].stx_shaded_line(
        trace_x,
        mean_trace - std_trace,
        mean_trace,
        mean_trace + std_trace,
        id="shaded-line",
    )
    axes[2, 2].set_title("Shaded Line")
    axes[2, 2].set_xlabel("X")
    axes[2, 2].set_ylabel("Y")

    # (2,3) KDE
    kde_data = np.concatenate(
        [np.random.normal(0, 1, 100), np.random.normal(3, 0.5, 50)]
    )
    axes[2, 3].stx_kde(kde_data, id="kde-plot")
    axes[2, 3].set_title("KDE")
    axes[2, 3].set_xlabel("Value")
    axes[2, 3].set_ylabel("Density")

    # Row 4: Special plots
    # (3,0) Raster
    spike_trains = [
        np.random.uniform(0, 1, np.random.poisson(10)) for _ in range(15)
    ]
    axes[3, 0].stx_raster(spike_trains, id="raster")
    axes[3, 0].set_title("Raster Plot")
    axes[3, 0].set_xlabel("Time")
    axes[3, 0].set_ylabel("Trial")

    # (3,1) Fill vertical regions
    axes[3, 1].plot(x, y, "-", id="fv-line")
    axes[3, 1].stx_fillv([2, 6], [4, 8], alpha=0.2, color="red", id="fillv")
    axes[3, 1].set_title("Fill Vertical")
    axes[3, 1].set_xlabel("X")
    axes[3, 1].set_ylabel("Y")

    # (3,2) Rectangle
    axes[3, 2].plot(x, y, "-", id="rect-line")
    axes[3, 2].stx_rectangle(
        2, -0.5, 3, 2, edgecolor="red", facecolor="none", id="rect"
    )
    axes[3, 2].set_title("Rectangle")
    axes[3, 2].set_xlabel("X")
    axes[3, 2].set_ylabel("Y")

    # (3,3) ECDF
    axes[3, 3].stx_ecdf(np.concatenate(group_data), id="ecdf")
    axes[3, 3].set_title("ECDF")
    axes[3, 3].set_xlabel("Value")
    axes[3, 3].set_ylabel("Cumulative Prob")

    png_path = output_dir / "04_all_plot_types.png"
    stx.io.save(fig, png_path)
    fig.close()

    return png_path.with_suffix(".json")


@stx.session
def main(
    backend: Backend = "flask",
    CONFIG=stx.INJECTED,
    logger=stx.INJECTED,
):
    """
    Demo 04: All plot types (4x4)

    Features:
        Row 1: Line, Scatter, Bar, Histogram
        Row 2: Error Bar, Fill Between, Box, Violin
        Row 3: Heatmap, Contour, Shaded Line, KDE
        Row 4: Raster, Fill Vertical, Rectangle, ECDF

    Parameters
    ----------
    backend : str
        GUI backend: flask, dearpygui, qt, tkinter, mpl

    Port: 5054
    """
    out = Path(CONFIG.SDIR_OUT)

    logger.info("=" * 60)
    logger.info("Demo 04: All plot types (4x4)")
    logger.info(f"Port: {PORT}")
    logger.info("=" * 60)

    json_path = create_figure(out)
    logger.info(f"Created: {json_path}")

    logger.info(f"\nLaunching editor (backend={backend}, port={PORT})...")
    stx.vis.edit(str(json_path), backend=backend, port=PORT)

    return 0


if __name__ == "__main__":
    main()

# EOF
