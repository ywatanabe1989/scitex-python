#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-11 02:03:56 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/vis/demo_gui_editors_complex_figures.py

"""
Demo: stx.vis.edit() - Interactive Figure Editors for Complex Multi-Panel Figures

Tests schema v0.3 geometry extraction for shape-based hit testing.

Usage:
    ./demo_gui_editors_complex_figures.py                      # Auto-detect backend
    ./demo_gui_editors_complex_figures.py --backend flask      # Browser-based
    ./demo_gui_editors_complex_figures.py --backend dearpygui  # GPU-accelerated
    ./demo_gui_editors_complex_figures.py --backend qt         # Qt desktop
    ./demo_gui_editors_complex_figures.py --backend tkinter    # Built-in Python
    ./demo_gui_editors_complex_figures.py --backend mpl        # Minimal matplotlib
    ./demo_gui_editors_complex_figures.py --figure 1           # Edit specific figure (1-3)
    ./demo_gui_editors_complex_figures.py --test-geometry      # Test geometry extraction only

Install GUI backends: pip install scitex[gui]
"""

import numpy as np
from pathlib import Path
from typing import Literal
import scitex as stx
from scitex.plt.styles.presets import SCITEX_STYLE

Backend = Literal["auto", "flask", "dearpygui", "qt", "tkinter", "mpl"]


def test_geometry_extraction(fig, axes, fig_name: str = "figure"):
    """Test schema v0.3 geometry extraction on a figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure or stx.plt.FigureWrapper
        The figure to test
    axes : array-like
        The axes array
    fig_name : str
        Name for logging
    """
    from scitex.vis.editor.flask_editor._bbox import (
        extract_bboxes_multi,
        GEOMETRY_V03_AVAILABLE,
    )

    if not GEOMETRY_V03_AVAILABLE:
        print(f"WARNING: Schema v0.3 geometry extraction not available")
        return

    # Get matplotlib objects
    mpl_fig = fig.figure if hasattr(fig, "figure") else fig
    mpl_fig.canvas.draw()
    renderer = mpl_fig.canvas.get_renderer()

    # Build axes map
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    axes_map = {}
    for idx, ax in enumerate(axes_flat):
        mpl_ax = ax._axis_mpl if hasattr(ax, "_axis_mpl") else ax
        row, col = divmod(idx, axes.shape[1] if hasattr(axes, "shape") else 1)
        axes_map[f"ax_{row}{col}"] = mpl_ax

    # Extract bboxes with geometry
    bboxes = extract_bboxes_multi(mpl_fig, axes_map, renderer, 800, 600)

    print(f"\n{'='*60}")
    print(f"Schema v0.3 Geometry Test: {fig_name}")
    print(f"{'='*60}")

    # Check metadata
    if "_meta" in bboxes:
        meta = bboxes["_meta"]
        print(f"Schema version: {meta.get('schema_version')}")
        print(f"Axes count: {len(meta.get('axes', {}))}")
    else:
        print("WARNING: No _meta found")

    # Count elements by type
    stats = {"line": 0, "scatter": 0, "fill": 0, "bar": 0, "other": 0}
    geom_stats = {"with_geometry": 0, "without_geometry": 0}

    print(f"\n--- Elements with geometry_px ---")
    for name, bbox in sorted(bboxes.items()):
        if name == "_meta":
            continue

        elem_type = bbox.get("element_type", "other")
        if elem_type in stats:
            stats[elem_type] += 1
        else:
            stats["other"] += 1

        if "geometry_px" in bbox:
            geom_stats["with_geometry"] += 1
            geom = bbox["geometry_px"]
            if "path_simplified" in geom:
                pts = len(geom["path_simplified"])
                print(f"  {name}: line ({pts} pts)")
            elif "points" in geom:
                pts = len(geom["points"])
                hr = geom.get("hit_radius_px", "N/A")
                print(f"  {name}: scatter ({pts} pts, hit_r={hr})")
            elif "polygon" in geom:
                pts = len(geom["polygon"])
                print(f"  {name}: fill ({pts} vertices)")
            elif "rectangles" in geom:
                rects = len(geom["rectangles"])
                print(f"  {name}: bar ({rects} bars)")
            else:
                print(f"  {name}: {elem_type} (geometry_px present)")
        else:
            geom_stats["without_geometry"] += 1

    print(f"\n--- Summary ---")
    print(f"Element types: {stats}")
    print(
        f"Geometry coverage: {geom_stats['with_geometry']}/{geom_stats['with_geometry']+geom_stats['without_geometry']} elements"
    )
    print(f"{'='*60}\n")


def create_figure_01_multi_type(output_dir: Path, test_geometry: bool = False):
    """Create multi-type per axis figure: Line + Scatter + Fill.

    Parameters
    ----------
    output_dir : Path
        Output directory for saved files
    test_geometry : bool
        If True, test geometry extraction before saving

    Returns
    -------
    Path
        Path to the JSON metadata file
    """
    from scipy import stats

    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    n_samples = 100
    x = np.linspace(0, 10, n_samples)
    y1 = np.sin(x) + np.random.normal(0, 0.1, n_samples)
    groups = ["Control", "Treatment A", "Treatment B"]
    group_data = [np.random.normal(loc, 0.5, 50) for loc in [0, 1, 2]]

    STYLE = SCITEX_STYLE.copy()
    fig, axes = stx.plt.subplots(2, 2, **STYLE)

    # Panel A: Line + Scatter + Vertical fill regions
    axes[0, 0].plot(x, y1, "-", linewidth=1, label="Trend", id="trend-line")
    axes[0, 0].plot(
        x,
        y1,
        "o",
        markersize=2,
        alpha=0.5,
        label="Data points",
        id="data-points",
    )
    axes[0, 0].stx_fillv([2, 6], [4, 8], alpha=0.2, color="gray", id="regions")
    axes[0, 0].set_title("A) Line + Scatter + Fill Regions")
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("Signal [a.u.]")
    axes[0, 0].legend()

    # Panel B: Multiple lines + error bars
    y_mean = np.sin(x)
    y_err = 0.2 + 0.1 * np.random.rand(n_samples)
    axes[0, 1].plot(x, y_mean, "-", color="blue", label="Mean", id="mean-line")
    axes[0, 1].fill_between(
        x, y_mean - y_err, y_mean + y_err, alpha=0.3, id="uncertainty"
    )
    axes[0, 1].errorbar(
        x[::10],
        y_mean[::10],
        yerr=y_err[::10],
        fmt="o",
        markersize=4,
        capsize=2,
        label="Error bars",
        id="errorbars",
    )
    axes[0, 1].set_title("B) Line + Fill + Error Bars")
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("Value [a.u.]")
    axes[0, 1].legend()

    # Panel C: Box + Strip + Scatter overlay
    for i, (data, label) in enumerate(zip(group_data, groups)):
        jitter = np.random.normal(0, 0.05, len(data))
        axes[1, 0].scatter(
            np.full_like(data, i) + jitter,
            data,
            alpha=0.3,
            s=10,
            id=f"scatter-{label}",
        )
    axes[1, 0].stx_box(group_data, labels=groups, id="boxplot")
    axes[1, 0].set_title("C) Box + Scatter Overlay")
    axes[1, 0].set_xlabel("Group")
    axes[1, 0].set_ylabel("Value [a.u.]")

    # Panel D: Histogram + KDE overlay
    hist_data = np.concatenate(group_data)
    axes[1, 1].hist(
        hist_data,
        bins=25,
        density=True,
        alpha=0.6,
        label="Histogram",
        id="hist",
    )
    kde_x = np.linspace(hist_data.min() - 1, hist_data.max() + 1, 200)
    kde = stats.gaussian_kde(hist_data)
    axes[1, 1].plot(
        kde_x, kde(kde_x), "-", linewidth=2, label="KDE", id="kde-overlay"
    )
    for i, (data, label) in enumerate(zip(group_data, groups)):
        axes[1, 1].axvline(
            np.mean(data),
            linestyle="--",
            alpha=0.7,
            label=f"{label} mean",
            id=f"mean-{label}",
        )
    axes[1, 1].set_title("D) Histogram + KDE + Mean Lines")
    axes[1, 1].set_xlabel("Value [a.u.]")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].legend(fontsize=5)

    # Test geometry extraction if requested
    if test_geometry:
        test_geometry_extraction(fig, axes, "Figure 01: Multi-type per axis")

    png_path = output_dir / "01_multi_type_per_axis.png"
    stx.io.save(fig, png_path)
    fig.close()

    return png_path.with_suffix(".json")


def create_figure_02_scientific(
    output_dir: Path, test_geometry: bool = False
) -> Path:
    """Create scientific figure: Time series + Statistics + Correlation.

    Parameters
    ----------
    output_dir : Path
        Output directory for saved files
    test_geometry : bool
        If True, test geometry extraction before saving

    Returns
    -------
    Path
        Path to the JSON metadata file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    groups = ["Control", "Treatment A", "Treatment B"]
    group_data = [np.random.normal(loc, 0.5, 50) for loc in [0, 1, 2]]

    STYLE = SCITEX_STYLE.copy()
    fig, axes = stx.plt.subplots(2, 3, **STYLE)

    # Panel A: Multi-channel time series with annotations
    time = np.linspace(0, 5, 500)
    ch1 = np.sin(2 * np.pi * time) + 0.3 * np.random.randn(500)
    ch2 = np.sin(2 * np.pi * time + np.pi / 4) + 0.3 * np.random.randn(500) + 3
    ch3 = np.sin(2 * np.pi * time + np.pi / 2) + 0.3 * np.random.randn(500) + 6

    axes[0, 0].plot(time, ch1, label="Ch1", id="ch1")
    axes[0, 0].plot(time, ch2, label="Ch2", id="ch2")
    axes[0, 0].plot(time, ch3, label="Ch3", id="ch3")
    axes[0, 0].stx_fillv(
        [1, 3], [2, 4], alpha=0.2, color="red", id="stim-period"
    )
    axes[0, 0].stx_rectangle(
        1,
        ch1.min() - 0.5,
        1,
        ch3.max() - ch1.min() + 1,
        edgecolor="red",
        facecolor="none",
        linestyle="--",
        id="stim-box",
    )
    axes[0, 0].set_title("A) Multi-channel Recording")
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("Amplitude [mV]")
    axes[0, 0].legend(loc="upper right")

    # Panel B: Shaded line with individual traces
    data_2d = np.random.randn(20, 100)
    mean_trace = data_2d.mean(axis=0)
    std_trace = data_2d.std(axis=0)
    trace_x = np.arange(100)

    for i in range(min(5, len(data_2d))):
        axes[0, 1].plot(
            trace_x, data_2d[i], alpha=0.2, color="gray", id=f"trace-{i}"
        )
    axes[0, 1].stx_shaded_line(
        trace_x,
        mean_trace - std_trace,
        mean_trace,
        mean_trace + std_trace,
        id="mean-std",
    )
    axes[0, 1].set_title("B) Individual Traces + Mean +/- Std")
    axes[0, 1].set_xlabel("Sample")
    axes[0, 1].set_ylabel("Value [a.u.]")

    # Panel C: Heatmap with contour overlay
    xx, yy = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
    zz = np.exp(-(xx**2 + yy**2) / 2) + 0.5 * np.exp(
        -((xx - 1) ** 2 + (yy - 1) ** 2) / 0.5
    )
    axes[0, 2].imshow(
        zz, extent=[-3, 3, -3, 3], origin="lower", cmap="viridis", id="heatmap"
    )
    axes[0, 2].contour(
        xx, yy, zz, levels=5, colors="white", linewidths=0.5, id="contours"
    )
    axes[0, 2].set_title("C) Heatmap + Contour Overlay")
    axes[0, 2].set_xlabel("X [a.u.]")
    axes[0, 2].set_ylabel("Y [a.u.]")

    # Panel D: Violin + Strip + Mean markers
    for i, (data, label) in enumerate(zip(group_data, groups)):
        jitter = np.random.normal(0, 0.08, len(data))
        axes[1, 0].scatter(
            np.full_like(data, i) + jitter,
            data,
            alpha=0.4,
            s=8,
            id=f"strip-{label}",
        )
    axes[1, 0].stx_violin(group_data, labels=groups, id="violin")
    for i, data in enumerate(group_data):
        axes[1, 0].plot(
            i,
            np.mean(data),
            "D",
            color="red",
            markersize=6,
            id=f"mean-marker-{i}",
        )
    axes[1, 0].set_title("D) Violin + Strip + Mean")
    axes[1, 0].set_xlabel("Group")
    axes[1, 0].set_ylabel("Value [a.u.]")

    # Panel E: Scatter + Regression + CI
    scatter_x = np.random.randn(50)
    scatter_y = 0.8 * scatter_x + 0.5 * np.random.randn(50)
    axes[1, 1].scatter(scatter_x, scatter_y, alpha=0.6, s=20, id="scatter")
    slope, intercept = np.polyfit(scatter_x, scatter_y, 1)
    reg_x = np.linspace(scatter_x.min(), scatter_x.max(), 100)
    reg_y = slope * reg_x + intercept
    axes[1, 1].plot(
        reg_x,
        reg_y,
        "-",
        color="red",
        linewidth=2,
        label=f"y={slope:.2f}x+{intercept:.2f}",
        id="regression",
    )
    ci = 0.3
    axes[1, 1].fill_between(
        reg_x, reg_y - ci, reg_y + ci, alpha=0.2, color="red", id="ci-band"
    )
    axes[1, 1].set_title("E) Scatter + Regression + CI")
    axes[1, 1].set_xlabel("X [a.u.]")
    axes[1, 1].set_ylabel("Y [a.u.]")
    axes[1, 1].legend()

    # Panel F: Bar + Error bars + Significance markers
    means = [np.mean(d) for d in group_data]
    stds = [np.std(d) for d in group_data]
    bar_x = np.arange(len(groups))
    axes[1, 2].bar(bar_x, means, yerr=stds, capsize=3, alpha=0.7, id="bars")
    y_max = max(means) + max(stds) + 0.3
    axes[1, 2].plot(
        [0, 0, 2, 2],
        [y_max, y_max + 0.1, y_max + 0.1, y_max],
        "k-",
        linewidth=1,
        id="sig-bracket",
    )
    axes[1, 2].text(
        1, y_max + 0.15, "***", ha="center", va="bottom", fontsize=10
    )
    axes[1, 2].set_xticks(bar_x)
    axes[1, 2].set_xticklabels(groups)
    axes[1, 2].set_title("F) Bar + Error + Significance")
    axes[1, 2].set_xlabel("Group")
    axes[1, 2].set_ylabel("Mean Value [a.u.]")

    # Test geometry extraction if requested
    if test_geometry:
        test_geometry_extraction(fig, axes, "Figure 02: Scientific figure")

    png_path = output_dir / "02_scientific_figure.png"
    stx.io.save(fig, png_path)
    fig.close()

    return png_path.with_suffix(".json")


def create_figure_04_all_plot_types(
    output_dir: Path, test_geometry: bool = False
) -> Path:
    """Create comprehensive figure covering all major plot types.

    Parameters
    ----------
    output_dir : Path
        Output directory for saved files
    test_geometry : bool
        If True, test geometry extraction before saving

    Returns
    -------
    Path
        Path to the JSON metadata file
    """
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

    # Test geometry extraction if requested
    if test_geometry:
        test_geometry_extraction(fig, axes, "Figure 04: All plot types")

    png_path = output_dir / "04_all_plot_types.png"
    stx.io.save(fig, png_path)
    fig.close()

    return png_path.with_suffix(".json")


def create_figure_05_large_grid(
    output_dir: Path, test_geometry: bool = False
) -> Path:
    """Create large comprehensive figure with 6x6 grid covering all plot types.

    Parameters
    ----------
    output_dir : Path
        Output directory for saved files
    test_geometry : bool
        If True, test geometry extraction before saving

    Returns
    -------
    Path
        Path to the JSON metadata file
    """
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

    # Test geometry extraction if requested
    if test_geometry:
        test_geometry_extraction(fig, axes, "Figure 05: Large grid")

    png_path = output_dir / "05_large_grid.png"
    stx.io.save(fig, png_path)
    fig.close()

    return png_path.with_suffix(".json")


def create_figure_06_different_sizes(
    output_dir: Path, test_geometry: bool = False
) -> Path:
    """Create figure with different aspect ratio (wide).

    Parameters
    ----------
    output_dir : Path
        Output directory for saved files
    test_geometry : bool
        If True, test geometry extraction before saving

    Returns
    -------
    Path
        Path to the JSON metadata file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    STYLE = SCITEX_STYLE.copy()
    STYLE["fig_size_mm"] = (180, 60)  # Wide aspect ratio
    fig, axes = stx.plt.subplots(1, 4, **STYLE)

    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    axes[0].plot(x, y, "-", id="line-1")
    axes[0].set_title("Panel A")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")

    axes[1].scatter(np.random.randn(30), np.random.randn(30), id="scatter-1")
    axes[1].set_title("Panel B")
    axes[1].set_xlabel("X")

    axes[2].bar([0, 1, 2], [1, 2, 1.5], id="bar-1")
    axes[2].set_title("Panel C")
    axes[2].set_xlabel("Group")

    axes[3].hist(np.random.randn(100), bins=15, id="hist-1")
    axes[3].set_title("Panel D")
    axes[3].set_xlabel("Value")

    if test_geometry:
        test_geometry_extraction(fig, axes, "Figure 06: Wide format")

    png_path = output_dir / "06_wide_format.png"
    stx.io.save(fig, png_path)
    fig.close()

    return png_path.with_suffix(".json")


def create_figure_07_tall(
    output_dir: Path, test_geometry: bool = False
) -> Path:
    """Create figure with tall aspect ratio.

    Parameters
    ----------
    output_dir : Path
        Output directory for saved files
    test_geometry : bool
        If True, test geometry extraction before saving

    Returns
    -------
    Path
        Path to the JSON metadata file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    STYLE = SCITEX_STYLE.copy()
    STYLE["fig_size_mm"] = (60, 180)  # Tall aspect ratio
    fig, axes = stx.plt.subplots(4, 1, **STYLE)

    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    axes[0].plot(x, y, "-", id="line-tall-1")
    axes[0].set_title("A")
    axes[0].set_ylabel("Y")

    axes[1].scatter(
        np.random.randn(30), np.random.randn(30), id="scatter-tall-1"
    )
    axes[1].set_title("B")
    axes[1].set_ylabel("Y")

    groups = ["A", "B", "C"]
    group_data = [np.random.randn(20) + i for i in range(3)]
    axes[2].stx_box(group_data, labels=groups, id="box-tall")
    axes[2].set_title("C")
    axes[2].set_ylabel("Value")

    axes[3].hist(np.random.randn(100), bins=15, id="hist-tall")
    axes[3].set_title("D")
    axes[3].set_xlabel("Value")
    axes[3].set_ylabel("Count")

    if test_geometry:
        test_geometry_extraction(fig, axes, "Figure 07: Tall format")

    png_path = output_dir / "07_tall_format.png"
    stx.io.save(fig, png_path)
    fig.close()

    return png_path.with_suffix(".json")


def create_figure_08_single_panel(
    output_dir: Path, test_geometry: bool = False
) -> Path:
    """Create single panel figure with multiple elements.

    Parameters
    ----------
    output_dir : Path
        Output directory for saved files
    test_geometry : bool
        If True, test geometry extraction before saving

    Returns
    -------
    Path
        Path to the JSON metadata file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    STYLE = SCITEX_STYLE.copy()
    STYLE["fig_size_mm"] = (100, 80)
    fig, ax = stx.plt.subplots(1, 1, **STYLE)

    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.sin(x) * np.exp(-x / 5)

    ax.plot(x, y1, "-", label="sin(x)", id="line-sin")
    ax.plot(x, y2, "--", label="cos(x)", id="line-cos")
    ax.plot(x, y3, "-.", label="damped", id="line-damped")
    ax.fill_between(x, y1 - 0.2, y1 + 0.2, alpha=0.2, id="fill-sin")
    ax.stx_fillv([2], [4], alpha=0.1, color="yellow", id="region")
    ax.scatter(x[::10], y1[::10], s=30, zorder=5, id="markers")

    ax.set_title("Single Panel - Multiple Elements")
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.legend()

    if test_geometry:
        # For single panel, axes is just ax - wrap in array for test function
        axes_arr = np.array([[ax]])
        test_geometry_extraction(fig, axes_arr, "Figure 08: Single panel")

    png_path = output_dir / "08_single_panel.png"
    stx.io.save(fig, png_path)
    fig.close()

    return png_path.with_suffix(".json")


def create_figure_03_neuroscience(
    output_dir: Path, test_geometry: bool = False
) -> Path:
    """Create neuroscience figure: Raster + PSTH + Waveforms.

    Parameters
    ----------
    output_dir : Path
        Output directory for saved files
    test_geometry : bool
        If True, test geometry extraction before saving

    Returns
    -------
    Path
        Path to the JSON metadata file
    """
    from scipy.ndimage import gaussian_filter1d

    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    STYLE = SCITEX_STYLE.copy()
    fig, axes = stx.plt.subplots(3, 2, **STYLE)

    # Generate spike data
    n_trials = 30
    spike_trains = []
    for trial in range(n_trials):
        pre_stim = np.random.uniform(0, 0.5, np.random.poisson(5))
        stim = np.random.uniform(0.5, 1.5, np.random.poisson(15))
        post_stim = np.random.uniform(1.5, 2.0, np.random.poisson(5))
        spike_trains.append(
            np.sort(np.concatenate([pre_stim, stim, post_stim]))
        )

    # Panel A: Raster + stimulus period
    axes[0, 0].stx_raster(spike_trains, id="raster")
    axes[0, 0].stx_fillv([0.5], [1.5], alpha=0.2, color="yellow", id="stim")
    axes[0, 0].axvline(
        0.5, color="green", linestyle="--", linewidth=1, id="stim-onset"
    )
    axes[0, 0].axvline(
        1.5, color="red", linestyle="--", linewidth=1, id="stim-offset"
    )
    axes[0, 0].set_title("A) Raster Plot")
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("Trial")

    # Panel B: PSTH + smoothed firing rate
    all_spikes = np.concatenate(spike_trains)
    hist_counts, bin_edges = np.histogram(all_spikes, bins=40, range=(0, 2))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    firing_rate = hist_counts / n_trials / (bin_edges[1] - bin_edges[0])

    axes[0, 1].bar(
        bin_centers,
        firing_rate,
        width=bin_edges[1] - bin_edges[0],
        alpha=0.6,
        id="psth",
    )
    smooth_rate = gaussian_filter1d(firing_rate.astype(float), sigma=2)
    axes[0, 1].plot(
        bin_centers,
        smooth_rate,
        "-",
        color="red",
        linewidth=2,
        label="Smoothed",
        id="smooth-rate",
    )
    axes[0, 1].stx_fillv(
        [0.5], [1.5], alpha=0.2, color="yellow", id="stim-period"
    )
    axes[0, 1].set_title("B) PSTH + Smoothed Rate")
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("Firing Rate [Hz]")
    axes[0, 1].legend()

    # Panel C: Spike waveforms overlay
    n_waveforms = 50
    waveform_time = np.linspace(0, 1.5, 30)
    for i in range(n_waveforms):
        waveform = -np.exp(
            -((waveform_time - 0.3) ** 2) / 0.02
        ) + 0.3 * np.exp(-((waveform_time - 0.8) ** 2) / 0.05)
        waveform += 0.05 * np.random.randn(len(waveform_time))
        axes[1, 0].plot(
            waveform_time,
            waveform,
            alpha=0.2,
            color="blue",
            id=f"waveform-{i}",
        )
    mean_waveform = -np.exp(
        -((waveform_time - 0.3) ** 2) / 0.02
    ) + 0.3 * np.exp(-((waveform_time - 0.8) ** 2) / 0.05)
    axes[1, 0].plot(
        waveform_time,
        mean_waveform,
        "-",
        color="red",
        linewidth=2,
        label="Mean",
        id="mean-waveform",
    )
    axes[1, 0].set_title("C) Spike Waveforms")
    axes[1, 0].set_xlabel("Time [ms]")
    axes[1, 0].set_ylabel("Amplitude [a.u.]")
    axes[1, 0].legend()

    # Panel D: ISI histogram + Exponential fit
    isis = []
    for train in spike_trains:
        if len(train) > 1:
            isis.extend(np.diff(train) * 1000)
    isis = np.array(isis)

    axes[1, 1].hist(isis, bins=30, density=True, alpha=0.6, id="isi-hist")
    isi_x = np.linspace(0, isis.max(), 100)
    mean_isi = np.mean(isis)
    exp_fit = (1 / mean_isi) * np.exp(-isi_x / mean_isi)
    axes[1, 1].plot(
        isi_x,
        exp_fit,
        "-",
        color="red",
        linewidth=2,
        label=f"Exp fit (tau={mean_isi:.1f}ms)",
        id="exp-fit",
    )
    axes[1, 1].set_title("D) ISI Distribution")
    axes[1, 1].set_xlabel("ISI [ms]")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].legend()

    # Panel E: Tuning curve with error bars and fit
    orientations = np.arange(0, 180, 20)
    responses = (
        10
        + 8 * np.exp(-((orientations - 90) ** 2) / (2 * 30**2))
        + np.random.randn(len(orientations))
    )
    response_sem = 1 + 0.5 * np.random.rand(len(orientations))

    axes[2, 0].errorbar(
        orientations,
        responses,
        yerr=response_sem,
        fmt="o",
        capsize=3,
        id="tuning-data",
    )
    fit_x = np.linspace(0, 180, 100)
    fit_y = 10 + 8 * np.exp(-((fit_x - 90) ** 2) / (2 * 30**2))
    axes[2, 0].plot(
        fit_x,
        fit_y,
        "-",
        color="red",
        linewidth=2,
        label="Gaussian fit",
        id="tuning-fit",
    )
    axes[2, 0].axvline(
        90, color="gray", linestyle="--", alpha=0.5, id="preferred"
    )
    axes[2, 0].set_title("E) Orientation Tuning Curve")
    axes[2, 0].set_xlabel("Orientation [deg]")
    axes[2, 0].set_ylabel("Response [spikes/s]")
    axes[2, 0].legend()

    # Panel F: Population activity heatmap
    n_neurons = 20
    n_time = 100
    pop_activity = np.zeros((n_neurons, n_time))
    for i in range(n_neurons):
        peak_time = 30 + np.random.randint(-10, 10)
        pop_activity[i] = np.exp(
            -((np.arange(n_time) - peak_time) ** 2) / (2 * 10**2)
        )
        pop_activity[i] += 0.1 * np.random.randn(n_time)

    axes[2, 1].imshow(
        pop_activity, aspect="auto", cmap="hot", id="pop-heatmap"
    )
    axes[2, 1].set_title("F) Population Activity")
    axes[2, 1].set_xlabel("Time [bins]")
    axes[2, 1].set_ylabel("Neuron #")

    # Test geometry extraction if requested
    if test_geometry:
        test_geometry_extraction(fig, axes, "Figure 03: Neuroscience figure")

    png_path = output_dir / "03_neuroscience_figure.png"
    stx.io.save(fig, png_path)
    fig.close()

    return png_path.with_suffix(".json")


# Figure creators registry - maps figure number to (name, creator_function)
FIGURE_CREATORS = {
    1: ("Multi-type per axis (2x2)", create_figure_01_multi_type),
    2: ("Scientific figure (2x3)", create_figure_02_scientific),
    3: ("Neuroscience figure (3x2)", create_figure_03_neuroscience),
    4: ("All plot types (4x4)", create_figure_04_all_plot_types),
    5: ("Large grid (6x6)", create_figure_05_large_grid),
    6: ("Wide format (1x4)", create_figure_06_different_sizes),
    7: ("Tall format (4x1)", create_figure_07_tall),
    8: ("Single panel (1x1)", create_figure_08_single_panel),
}


def _generate_figure_choices_doc():
    """Generate documentation string for available figure choices."""
    lines = ["Available figures:"]
    for num, (name, _) in sorted(FIGURE_CREATORS.items()):
        lines.append(f"        {num}: {name}")
    return "\n".join(lines)


@stx.session
def main(
    backend: Backend = "auto",
    figure: int = 2,
    test_geometry: bool = False,
    CONFIG=stx.INJECTED,
    plt=stx.INJECTED,
    COLORS=stx.INJECTED,
    rng_manager=stx.INJECTED,
    logger=stx.INJECTED,
):
    """Launch GUI editor for complex multi-panel figures."""
    out = Path(CONFIG.SDIR_OUT)

    logger.info("=" * 60)
    logger.info("Demo: Complex Figure Editor")
    logger.info("=" * 60)

    creators = FIGURE_CREATORS

    if figure not in creators:
        logger.error(
            f"Invalid figure number: {figure}. Choose 1-{len(creators)}."
        )
        return 1

    name, creator = creators[figure]
    logger.info(f"Creating figure {figure}: {name}")
    json_path = creator(out, test_geometry=test_geometry)
    logger.info(f"Created: {json_path}")

    # If test_geometry mode, skip editor launch
    if test_geometry:
        logger.info("Geometry test mode - skipping editor launch")
        return 0

    logger.info(f"\nLaunching editor (backend={backend})...")
    stx.vis.edit(str(json_path), backend=backend)

    return 0


# Dynamically generate docstring with figure choices
_main_doc = f"""Launch GUI editor for complex multi-panel figures.

Parameters
----------
backend : str
    GUI backend: auto, flask, dearpygui, qt, tkinter, mpl
figure : int
    Figure to edit (1-{len(FIGURE_CREATORS)})
    {_generate_figure_choices_doc()}
test_geometry : bool
    If True, run geometry extraction test only (no editor launch)
"""
main.__doc__ = _main_doc
# Also set on wrapped function if decorator exposes it
if hasattr(main, "__wrapped__"):
    main.__wrapped__.__doc__ = _main_doc


if __name__ == "__main__":
    main()

# EOF
