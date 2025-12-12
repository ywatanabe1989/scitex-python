#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_gallery.py - Comprehensive gallery of all plot types

"""
Comprehensive gallery covering almost all matplotlib/scitex plot types.

This plotter creates a large multi-panel figure demonstrating:
- All basic plot types (line, scatter, bar, histogram, etc.)
- Statistical plots (box, violin, errorbar, etc.)
- 2D plots with colorbars (heatmap, contour, etc.)
- Special plots (pie, polar, quiver, etc.)
- scitex-specific plots (raster, kde, ecdf, etc.)

Useful for:
- Testing plot rendering
- Testing colorbar handling
- Visual regression tests
- Documentation screenshots
"""

import numpy as np
import scitex as stx


def plot_gallery(plt, rng, figsize_mm=(300, 400)):
    """Comprehensive gallery of all plot types.

    Creates a large 8x6 grid (48 panels) covering:
    - Row 0: Line variants
    - Row 1: Scatter variants
    - Row 2: Bar variants
    - Row 3: Statistical plots
    - Row 4: Area/fill plots
    - Row 5: Heatmaps with colorbars
    - Row 6: Special plots
    - Row 7: scitex-specific plots

    Parameters
    ----------
    plt : module
        Plotting module (matplotlib.pyplot or scitex.plt)
    rng : numpy.random.Generator
        Random number generator
    figsize_mm : tuple
        Figure size in mm (width, height)

    Returns
    -------
    fig : Figure
        The figure object
    axes : ndarray
        2D array of axes (8, 6)
    """
    # Create large figure
    fig, axes = plt.subplots(8, 6, figsize_mm=figsize_mm)

    # Generate common data
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + 0.1 * rng.standard_normal(100)
    y2 = np.cos(x) + 0.1 * rng.standard_normal(100)
    y_err = 0.2 + 0.1 * rng.random(100)

    scatter_x = rng.standard_normal(50)
    scatter_y = 0.8 * scatter_x + 0.5 * rng.standard_normal(50)
    colors = rng.random(50)
    sizes = 50 * rng.random(50) + 10

    groups = ["A", "B", "C", "D"]
    group_data = [rng.normal(i, 0.5, 30) for i in range(4)]
    hist_data = np.concatenate(group_data)

    # ==========================================================================
    # Row 0: Line variants
    # ==========================================================================
    axes[0, 0].plot(x, y, "-", linewidth=1.5, id="line-solid")
    axes[0, 0].set_title("Solid Line")

    axes[0, 1].plot(x, y, "--", linewidth=1.5, id="line-dashed")
    axes[0, 1].set_title("Dashed Line")

    axes[0, 2].plot(x, y, "-.", linewidth=1.5, id="line-dashdot")
    axes[0, 2].set_title("Dash-dot Line")

    axes[0, 3].plot(x, y, ":", linewidth=1.5, id="line-dotted")
    axes[0, 3].set_title("Dotted Line")

    axes[0, 4].plot(x, y, "-o", markersize=3, markevery=10, id="line-marker")
    axes[0, 4].set_title("Line + Markers")

    # Multiple lines
    axes[0, 5].plot(x, y, "-", label="sin", id="multi-sin")
    axes[0, 5].plot(x, y2, "-", label="cos", id="multi-cos")
    axes[0, 5].legend(fontsize=6)
    axes[0, 5].set_title("Multiple Lines")

    # ==========================================================================
    # Row 1: Scatter variants
    # ==========================================================================
    axes[1, 0].scatter(scatter_x, scatter_y, s=20, id="scatter-basic")
    axes[1, 0].set_title("Basic Scatter")

    im = axes[1, 1].scatter(scatter_x, scatter_y, c=colors, cmap="viridis", s=20, id="scatter-color")
    fig.colorbar(im, ax=axes[1, 1], shrink=0.8)
    axes[1, 1].set_title("Colored Scatter")

    axes[1, 2].scatter(scatter_x, scatter_y, s=sizes, alpha=0.6, id="scatter-size")
    axes[1, 2].set_title("Sized Scatter")

    im = axes[1, 3].scatter(scatter_x, scatter_y, c=colors, s=sizes, alpha=0.6, cmap="plasma", id="scatter-full")
    fig.colorbar(im, ax=axes[1, 3], shrink=0.8)
    axes[1, 3].set_title("Color + Size Scatter")

    axes[1, 4].plot(scatter_x, scatter_y, "^", markersize=5, id="triangle")
    axes[1, 4].set_title("Triangle Markers")

    axes[1, 5].plot(scatter_x, scatter_y, "D", markersize=4, id="diamond")
    axes[1, 5].set_title("Diamond Markers")

    # ==========================================================================
    # Row 2: Bar variants
    # ==========================================================================
    bar_x = np.arange(len(groups))
    bar_y = [np.mean(d) for d in group_data]
    bar_err = [np.std(d) for d in group_data]

    axes[2, 0].bar(bar_x, bar_y, id="bar-basic")
    axes[2, 0].set_xticks(bar_x)
    axes[2, 0].set_xticklabels(groups)
    axes[2, 0].set_title("Basic Bar")

    axes[2, 1].bar(bar_x, bar_y, yerr=bar_err, capsize=3, id="bar-error")
    axes[2, 1].set_xticks(bar_x)
    axes[2, 1].set_xticklabels(groups)
    axes[2, 1].set_title("Bar + Error")

    axes[2, 2].barh(bar_x, bar_y, id="barh")
    axes[2, 2].set_yticks(bar_x)
    axes[2, 2].set_yticklabels(groups)
    axes[2, 2].set_title("Horizontal Bar")

    # Grouped bar
    width = 0.35
    axes[2, 3].bar(bar_x - width/2, bar_y, width, label="Group 1", id="bar-g1")
    axes[2, 3].bar(bar_x + width/2, [v * 0.8 for v in bar_y], width, label="Group 2", id="bar-g2")
    axes[2, 3].set_xticks(bar_x)
    axes[2, 3].set_xticklabels(groups)
    axes[2, 3].legend(fontsize=6)
    axes[2, 3].set_title("Grouped Bar")

    # Stacked bar
    bottom = np.zeros(len(groups))
    for i, label in enumerate(["Layer 1", "Layer 2", "Layer 3"]):
        vals = rng.uniform(1, 3, len(groups))
        axes[2, 4].bar(groups, vals, bottom=bottom, label=label, id=f"stack-{i}")
        bottom += vals
    axes[2, 4].legend(fontsize=5)
    axes[2, 4].set_title("Stacked Bar")

    # Histogram
    axes[2, 5].hist(hist_data, bins=20, alpha=0.7, edgecolor="white", id="hist")
    axes[2, 5].set_title("Histogram")

    # ==========================================================================
    # Row 3: Statistical plots
    # ==========================================================================
    axes[3, 0].stx_box(group_data, labels=groups, id="box")
    axes[3, 0].set_title("Box Plot")

    axes[3, 1].stx_violin(group_data, labels=groups, id="violin")
    axes[3, 1].set_title("Violin Plot")

    # Strip plot
    for i, (data, label) in enumerate(zip(group_data, groups)):
        jitter = rng.normal(0, 0.05, len(data))
        axes[3, 2].scatter(np.full_like(data, i) + jitter, data, alpha=0.5, s=10, id=f"strip-{label}")
    axes[3, 2].set_xticks(range(len(groups)))
    axes[3, 2].set_xticklabels(groups)
    axes[3, 2].set_title("Strip Plot")

    # Error bar
    axes[3, 3].errorbar(x[::10], y[::10], yerr=y_err[::10], fmt="o", capsize=3, id="errorbar")
    axes[3, 3].set_title("Error Bar")

    # ECDF
    axes[3, 4].stx_ecdf(hist_data, id="ecdf")
    axes[3, 4].set_title("ECDF")

    # KDE
    axes[3, 5].stx_kde(hist_data, id="kde")
    axes[3, 5].set_title("KDE")

    # ==========================================================================
    # Row 4: Area/fill plots
    # ==========================================================================
    axes[4, 0].fill_between(x, 0, y, alpha=0.5, id="area")
    axes[4, 0].set_title("Area Plot")

    axes[4, 1].fill_between(x, y - y_err, y + y_err, alpha=0.3, id="fill-between")
    axes[4, 1].plot(x, y, "-", linewidth=1, id="fill-line")
    axes[4, 1].set_title("Fill Between")

    # Stacked area
    y1 = np.abs(np.sin(x))
    y2_area = np.abs(np.cos(x))
    axes[4, 2].fill_between(x, 0, y1, alpha=0.5, label="A", id="stack-a")
    axes[4, 2].fill_between(x, y1, y1 + y2_area, alpha=0.5, label="B", id="stack-b")
    axes[4, 2].legend(fontsize=6)
    axes[4, 2].set_title("Stacked Area")

    # Fill vertical regions
    axes[4, 3].plot(x, y, "-", id="fillv-line")
    axes[4, 3].stx_fillv([2, 6], [4, 8], alpha=0.2, color="red", id="fillv")
    axes[4, 3].set_title("Fill Vertical")

    # Step plot
    axes[4, 4].step(x, y, where="mid", linewidth=1.5, id="step")
    axes[4, 4].set_title("Step Plot")

    # Stem plot
    x_stem = np.arange(0, 10, 0.5)
    y_stem = np.sin(x_stem)
    markerline, stemlines, baseline = axes[4, 5].stem(x_stem, y_stem)
    markerline.set_markersize(4)
    axes[4, 5].set_title("Stem Plot")

    # ==========================================================================
    # Row 5: Heatmaps with colorbars
    # ==========================================================================
    # Basic heatmap
    heatmap = rng.standard_normal((10, 10))
    im = axes[5, 0].imshow(heatmap, cmap="viridis", aspect="auto", id="heatmap-viridis")
    fig.colorbar(im, ax=axes[5, 0], shrink=0.8)
    axes[5, 0].set_title("Heatmap (viridis)")

    # Hot colormap
    im = axes[5, 1].imshow(heatmap, cmap="hot", aspect="auto", id="heatmap-hot")
    fig.colorbar(im, ax=axes[5, 1], shrink=0.8)
    axes[5, 1].set_title("Heatmap (hot)")

    # Coolwarm colormap (diverging)
    im = axes[5, 2].imshow(heatmap, cmap="coolwarm", aspect="auto", id="heatmap-coolwarm")
    fig.colorbar(im, ax=axes[5, 2], shrink=0.8)
    axes[5, 2].set_title("Heatmap (coolwarm)")

    # Contour
    xx, yy = np.meshgrid(np.linspace(-2, 2, 30), np.linspace(-2, 2, 30))
    zz = np.exp(-(xx**2 + yy**2))
    cs = axes[5, 3].contour(xx, yy, zz, levels=8, id="contour")
    axes[5, 3].clabel(cs, inline=True, fontsize=6)
    axes[5, 3].set_title("Contour")

    # Filled contour with colorbar
    im = axes[5, 4].contourf(xx, yy, zz, levels=8, cmap="RdYlBu_r", id="contourf")
    fig.colorbar(im, ax=axes[5, 4], shrink=0.8)
    axes[5, 4].set_title("Filled Contour")

    # pcolormesh with colorbar
    im = axes[5, 5].pcolormesh(xx, yy, zz, cmap="magma", id="pcolormesh")
    fig.colorbar(im, ax=axes[5, 5], shrink=0.8)
    axes[5, 5].set_title("Pcolormesh")

    # ==========================================================================
    # Row 6: Special plots
    # ==========================================================================
    # Pie chart
    pie_sizes = rng.uniform(10, 30, 5)
    axes[6, 0].pie(pie_sizes, labels=["A", "B", "C", "D", "E"], autopct="%1.0f%%")
    axes[6, 0].set_title("Pie Chart")

    # Quiver (vector field)
    qx = np.linspace(-2, 2, 8)
    qy = np.linspace(-2, 2, 8)
    QX, QY = np.meshgrid(qx, qy)
    U = -QY
    V = QX
    axes[6, 1].quiver(QX, QY, U, V, id="quiver")
    axes[6, 1].set_title("Quiver (Vectors)")

    # Streamplot
    axes[6, 2].streamplot(QX, QY, U, V, density=0.8, id="streamplot")
    axes[6, 2].set_title("Streamplot")

    # 2D histogram with colorbar
    x2d = rng.standard_normal(1000)
    y2d = rng.standard_normal(1000)
    h = axes[6, 3].hist2d(x2d, y2d, bins=20, cmap="Blues", id="hist2d")
    fig.colorbar(h[3], ax=axes[6, 3], shrink=0.8)
    axes[6, 3].set_title("2D Histogram")

    # Hexbin with colorbar
    hb = axes[6, 4].hexbin(x2d, y2d, gridsize=15, cmap="Greens", id="hexbin")
    fig.colorbar(hb, ax=axes[6, 4], shrink=0.8)
    axes[6, 4].set_title("Hexbin")

    # Multiple histograms
    for i, (data, label) in enumerate(zip(group_data[:3], groups[:3])):
        axes[6, 5].hist(data, bins=15, alpha=0.5, label=label, id=f"hist-{label}")
    axes[6, 5].legend(fontsize=6)
    axes[6, 5].set_title("Multiple Histograms")

    # ==========================================================================
    # Row 7: scitex-specific and misc plots
    # ==========================================================================
    # Raster plot
    spike_trains = [rng.uniform(0, 1, rng.poisson(10)) for _ in range(15)]
    axes[7, 0].stx_raster(spike_trains, id="raster")
    axes[7, 0].set_title("Raster Plot")

    # Shaded line (mean ± std)
    data_2d = rng.standard_normal((20, 50))
    mean_trace = data_2d.mean(axis=0)
    std_trace = data_2d.std(axis=0)
    trace_x = np.arange(50)
    axes[7, 1].stx_shaded_line(trace_x, mean_trace - std_trace, mean_trace, mean_trace + std_trace, id="shaded")
    axes[7, 1].set_title("Shaded Line")

    # Rectangle annotation
    axes[7, 2].plot(x, y, "-", id="rect-line")
    axes[7, 2].stx_rectangle(2, -0.5, 3, 1.5, edgecolor="red", facecolor="none", linewidth=2, id="rect")
    axes[7, 2].set_title("Rectangle")

    # Axvline/Axhline
    axes[7, 3].plot(x, y, "-", id="axline-base")
    axes[7, 3].axvline(5, color="red", linestyle="--", linewidth=1, id="axvline")
    axes[7, 3].axhline(0, color="blue", linestyle=":", linewidth=1, id="axhline")
    axes[7, 3].set_title("Axvline/Axhline")

    # Span (axvspan/axhspan)
    axes[7, 4].plot(x, y, "-", id="span-base")
    axes[7, 4].axvspan(3, 5, alpha=0.2, color="yellow", id="axvspan")
    axes[7, 4].axhspan(-0.5, 0.5, alpha=0.2, color="cyan", id="axhspan")
    axes[7, 4].set_title("Axvspan/Axhspan")

    # Text annotations
    axes[7, 5].plot(x, y, "-", id="text-base")
    axes[7, 5].annotate("Peak", xy=(1.5, 1), xytext=(3, 1.2),
                        arrowprops=dict(arrowstyle="->", color="red"),
                        fontsize=8, id="annotation")
    axes[7, 5].text(7, -0.5, "Text", fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat"))
    axes[7, 5].set_title("Annotations")

    # Tight layout
    fig.tight_layout()

    return fig, axes


# Convenience function for quick demo
def plot_gallery_quick(plt, rng, ax=None):
    """Quick gallery demo - smaller 4x4 version.

    This is a reduced version for quick testing.
    Use plot_gallery() for the full comprehensive gallery.
    """
    if ax is not None:
        raise ValueError("plot_gallery_quick creates its own multi-axis figure")

    fig, axes = plt.subplots(4, 4, figsize_mm=(200, 200))

    x = np.linspace(0, 10, 100)
    y = np.sin(x) + 0.1 * rng.standard_normal(100)
    groups = ["A", "B", "C", "D"]
    group_data = [rng.normal(i, 0.5, 30) for i in range(4)]

    # Row 0
    axes[0, 0].plot(x, y, "-", id="line")
    axes[0, 0].set_title("Line")

    axes[0, 1].scatter(rng.standard_normal(30), rng.standard_normal(30), id="scatter")
    axes[0, 1].set_title("Scatter")

    axes[0, 2].bar(groups, [np.mean(d) for d in group_data], id="bar")
    axes[0, 2].set_title("Bar")

    axes[0, 3].hist(np.concatenate(group_data), bins=15, id="hist")
    axes[0, 3].set_title("Histogram")

    # Row 1
    axes[1, 0].stx_box(group_data, labels=groups, id="box")
    axes[1, 0].set_title("Box")

    axes[1, 1].stx_violin(group_data, labels=groups, id="violin")
    axes[1, 1].set_title("Violin")

    axes[1, 2].errorbar(x[::10], y[::10], yerr=0.2, fmt="o", capsize=3, id="errorbar")
    axes[1, 2].set_title("Errorbar")

    axes[1, 3].fill_between(x, y - 0.2, y + 0.2, alpha=0.3, id="fill")
    axes[1, 3].plot(x, y, "-", id="fill-line")
    axes[1, 3].set_title("Fill Between")

    # Row 2
    hm = rng.standard_normal((8, 8))
    im = axes[2, 0].imshow(hm, cmap="viridis", id="heatmap")
    fig.colorbar(im, ax=axes[2, 0], shrink=0.8)
    axes[2, 0].set_title("Heatmap")

    xx, yy = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
    zz = np.exp(-(xx**2 + yy**2))
    im = axes[2, 1].contourf(xx, yy, zz, levels=8, cmap="RdYlBu_r", id="contourf")
    fig.colorbar(im, ax=axes[2, 1], shrink=0.8)
    axes[2, 1].set_title("Contourf")

    axes[2, 2].stx_kde(np.concatenate(group_data), id="kde")
    axes[2, 2].set_title("KDE")

    axes[2, 3].stx_ecdf(np.concatenate(group_data), id="ecdf")
    axes[2, 3].set_title("ECDF")

    # Row 3
    spike_trains = [rng.uniform(0, 1, rng.poisson(8)) for _ in range(10)]
    axes[3, 0].stx_raster(spike_trains, id="raster")
    axes[3, 0].set_title("Raster")

    axes[3, 1].step(x, y, where="mid", id="step")
    axes[3, 1].set_title("Step")

    pie_sizes = rng.uniform(10, 30, 4)
    axes[3, 2].pie(pie_sizes, labels=groups)
    axes[3, 2].set_title("Pie")

    data_2d = rng.standard_normal((10, 30))
    mean_trace = data_2d.mean(axis=0)
    std_trace = data_2d.std(axis=0)
    axes[3, 3].stx_shaded_line(np.arange(30), mean_trace - std_trace, mean_trace, mean_trace + std_trace, id="shaded")
    axes[3, 3].set_title("Shaded Line")

    fig.tight_layout()
    return fig, axes


