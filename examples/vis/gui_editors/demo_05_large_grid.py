#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: examples/vis/gui_editors/demo_05_large_grid.py

"""
Demo 05: Large grid figure (6x6) - All plot type variants

Port: 5055

Usage:
    ./demo_05_large_grid.py              # Flask backend (default)
    ./demo_05_large_grid.py --backend qt # Qt backend
"""

import numpy as np
from pathlib import Path
from typing import Literal
import scitex as stx
from scitex.plt.styles.presets import SCITEX_STYLE

Backend = Literal["auto", "flask", "dearpygui", "qt", "tkinter", "mpl"]
PORT = 5055


def create_figure(output_dir: Path) -> Path:
    """Create large comprehensive figure with 6x6 grid covering all plot types."""
    from scipy import stats

    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    # Generate diverse sample data
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + 0.1 * np.random.randn(100)
    groups = ["A", "B", "C", "D"]
    group_data = [np.random.normal(i, 0.5, 30) for i in range(4)]

    STYLE = SCITEX_STYLE.copy()
    STYLE["fig_size_mm"] = (200, 170)  # Larger figure
    fig, axes = stx.plt.subplots(6, 6, **STYLE)

    # Row 0: Line variants
    axes[0, 0].plot(x, y, "-", label="Solid", id="line-solid")
    axes[0, 0].set_title("Solid Line")

    axes[0, 1].plot(x, y, "--", label="Dashed", id="line-dashed")
    axes[0, 1].set_title("Dashed Line")

    axes[0, 2].plot(x, y, "-.", label="Dash-dot", id="line-dashdot")
    axes[0, 2].set_title("Dash-dot Line")

    axes[0, 3].plot(x, y, ":", label="Dotted", id="line-dotted")
    axes[0, 3].set_title("Dotted Line")

    axes[0, 4].step(x, y, where="mid", id="step-plot")
    axes[0, 4].set_title("Step Plot")

    axes[0, 5].plot(x, y, "-o", markersize=3, markevery=10, id="line-marker")
    axes[0, 5].set_title("Line + Markers")

    # Row 1: Scatter variants
    scatter_x = np.random.randn(50)
    scatter_y = 0.8 * scatter_x + 0.5 * np.random.randn(50)
    colors = np.random.rand(50)
    sizes = 50 * np.random.rand(50) + 10

    axes[1, 0].scatter(scatter_x, scatter_y, s=30, id="scatter-basic")
    axes[1, 0].set_title("Basic Scatter")

    axes[1, 1].scatter(
        scatter_x,
        scatter_y,
        c=colors,
        cmap="viridis",
        s=30,
        id="scatter-color",
    )
    axes[1, 1].set_title("Colored Scatter")

    axes[1, 2].scatter(
        scatter_x, scatter_y, s=sizes, alpha=0.6, id="scatter-size"
    )
    axes[1, 2].set_title("Sized Scatter")

    axes[1, 3].scatter(
        scatter_x, scatter_y, c=colors, s=sizes, alpha=0.6, id="scatter-full"
    )
    axes[1, 3].set_title("Full Scatter")

    axes[1, 4].plot(
        scatter_x, scatter_y, "^", markersize=5, id="triangle-marker"
    )
    axes[1, 4].set_title("Triangle Markers")

    axes[1, 5].plot(
        scatter_x, scatter_y, "D", markersize=4, id="diamond-marker"
    )
    axes[1, 5].set_title("Diamond Markers")

    # Row 2: Bar and histogram variants
    bar_x = np.arange(len(groups))
    bar_y = [np.mean(d) for d in group_data]
    bar_err = [np.std(d) for d in group_data]

    axes[2, 0].bar(bar_x, bar_y, id="bar-basic")
    axes[2, 0].set_title("Basic Bar")

    axes[2, 1].bar(bar_x, bar_y, yerr=bar_err, capsize=3, id="bar-error")
    axes[2, 1].set_title("Bar + Error")

    axes[2, 2].barh(bar_x, bar_y, id="barh")
    axes[2, 2].set_title("Horizontal Bar")

    hist_data = np.concatenate(group_data)
    axes[2, 3].hist(hist_data, bins=20, alpha=0.7, id="hist-basic")
    axes[2, 3].set_title("Histogram")

    axes[2, 4].hist(
        hist_data, bins=20, density=True, alpha=0.7, id="hist-density"
    )
    axes[2, 4].set_title("Density Hist")

    axes[2, 5].hist(
        hist_data, bins=20, cumulative=True, alpha=0.7, id="hist-cumul"
    )
    axes[2, 5].set_title("Cumulative Hist")

    # Row 3: Statistical plots
    axes[3, 0].stx_box(group_data, labels=groups, id="box")
    axes[3, 0].set_title("Box Plot")

    axes[3, 1].stx_violin(group_data, labels=groups, id="violin")
    axes[3, 1].set_title("Violin Plot")

    # Strip plot
    for i, (data, label) in enumerate(zip(group_data, groups)):
        jitter = np.random.normal(0, 0.05, len(data))
        axes[3, 2].scatter(
            np.full_like(data, i) + jitter,
            data,
            alpha=0.5,
            s=15,
            id=f"strip-{label}",
        )
    axes[3, 2].set_title("Strip Plot")

    # Swarm-like
    for i, (data, label) in enumerate(zip(group_data, groups)):
        jitter = np.random.normal(0, 0.1, len(data))
        axes[3, 3].scatter(
            np.full_like(data, i) + jitter,
            data,
            alpha=0.6,
            s=20,
            id=f"swarm-{label}",
        )
    axes[3, 3].set_title("Swarm-like")

    axes[3, 4].stx_ecdf(hist_data, id="ecdf")
    axes[3, 4].set_title("ECDF")

    axes[3, 5].stx_kde(hist_data, id="kde")
    axes[3, 5].set_title("KDE")

    # Row 4: Area and fill
    y_err = 0.2 + 0.1 * np.random.rand(100)

    axes[4, 0].fill_between(x, 0, y, alpha=0.5, id="area")
    axes[4, 0].set_title("Area Plot")

    axes[4, 1].fill_between(
        x, y - y_err, y + y_err, alpha=0.3, id="fill-between"
    )
    axes[4, 1].plot(x, y, "-", id="fill-line")
    axes[4, 1].set_title("Fill Between")

    # Stacked area
    y1 = np.abs(np.sin(x))
    y2 = np.abs(np.cos(x))
    axes[4, 2].fill_between(x, 0, y1, alpha=0.5, label="A", id="stack-a")
    axes[4, 2].fill_between(x, y1, y1 + y2, alpha=0.5, label="B", id="stack-b")
    axes[4, 2].set_title("Stacked Area")

    axes[4, 3].stx_fillv([2, 6], [4, 8], alpha=0.3, color="red", id="fillv")
    axes[4, 3].plot(x, y, "-", id="fillv-line")
    axes[4, 3].set_title("Fill Vertical")

    axes[4, 4].stx_rectangle(
        2, -0.5, 4, 2, edgecolor="blue", facecolor="none", id="rect"
    )
    axes[4, 4].plot(x, y, "-", id="rect-line")
    axes[4, 4].set_title("Rectangle")

    data_2d = np.random.randn(20, 50)
    mean_trace = data_2d.mean(axis=0)
    std_trace = data_2d.std(axis=0)
    trace_x = np.arange(50)
    axes[4, 5].stx_shaded_line(
        trace_x,
        mean_trace - std_trace,
        mean_trace,
        mean_trace + std_trace,
        id="shaded",
    )
    axes[4, 5].set_title("Shaded Line")

    # Row 5: Heatmaps and special
    heatmap = np.random.randn(10, 10)
    axes[5, 0].imshow(heatmap, cmap="viridis", id="heatmap")
    axes[5, 0].set_title("Heatmap")

    axes[5, 1].imshow(heatmap, cmap="hot", id="hot-heatmap")
    axes[5, 1].set_title("Hot Colormap")

    xx, yy = np.meshgrid(np.linspace(-2, 2, 30), np.linspace(-2, 2, 30))
    zz = np.exp(-(xx**2 + yy**2))
    axes[5, 2].contour(xx, yy, zz, levels=8, id="contour")
    axes[5, 2].set_title("Contour")

    axes[5, 3].contourf(xx, yy, zz, levels=8, id="contourf")
    axes[5, 3].set_title("Filled Contour")

    spike_trains = [
        np.random.uniform(0, 1, np.random.poisson(10)) for _ in range(15)
    ]
    axes[5, 4].stx_raster(spike_trains, id="raster")
    axes[5, 4].set_title("Raster")

    # Error bar plot
    ex = np.arange(10)
    ey = np.random.randn(10)
    err = 0.2 + 0.1 * np.random.rand(10)
    axes[5, 5].errorbar(ex, ey, yerr=err, fmt="o", capsize=3, id="errorbar")
    axes[5, 5].set_title("Error Bar")

    png_path = output_dir / "05_large_grid.png"
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
    Demo 05: Large grid figure (6x6)

    Features:
        Row 0: Line variants (solid, dashed, dash-dot, dotted, step, marker)
        Row 1: Scatter variants (basic, colored, sized, full, triangle, diamond)
        Row 2: Bar & histogram variants
        Row 3: Statistical plots (box, violin, strip, swarm, ECDF, KDE)
        Row 4: Area & fill plots
        Row 5: Heatmaps & special plots

    Parameters
    ----------
    backend : str
        GUI backend: flask, dearpygui, qt, tkinter, mpl

    Port: 5055
    """
    out = Path(CONFIG.SDIR_OUT)

    logger.info("=" * 60)
    logger.info("Demo 05: Large grid figure (6x6)")
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
