#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_jointplot.py - Joint plot (scatter with marginal distributions)

"""
Joint plot - bivariate plot with marginal distributions.
"""

import numpy as np
import scitex as stx


def plot_jointplot(plt, rng, ax=None):
    """Joint plot with scatter and marginal histograms.

    Note: This creates its own figure layout.
    The ax parameter is ignored.

    Parameters
    ----------
    plt : module
        Plotting module
    rng : numpy.random.Generator
        Random number generator
    ax : Axes, optional
        Ignored - jointplot creates its own layout.

    Returns
    -------
    fig : Figure
        The figure object
    ax_dict : dict
        Dictionary of axes: {"main", "top", "right"}
    """
    from scipy.stats import gaussian_kde

    # Generate bivariate data
    n = 300
    # Correlated data
    x = rng.normal(0, 1, n)
    y = 0.7 * x + rng.normal(0, 0.5, n)

    # Create figure with gridspec
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[4, 1],
        height_ratios=[1, 4],
        wspace=0.05,
        hspace=0.05,
    )

    # Main scatter plot
    ax_main = fig.add_subplot(gs[1, 0])
    ax_main.scatter(x, y, s=20, alpha=0.5, color="#1f77b4")

    # Add regression line
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax_main.plot(x_line, slope * x_line + intercept, "r--", linewidth=2, label=f"y = {slope:.2f}x + {intercept:.2f}")
    ax_main.legend(loc="upper left")

    # Add 2D KDE contours
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)
    x_grid = np.linspace(x.min(), x.max(), 50)
    y_grid = np.linspace(y.min(), y.max(), 50)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    ax_main.contour(X, Y, Z, levels=5, colors="darkblue", alpha=0.5, linewidths=0.5)

    ax_main.set_xlabel("X")
    ax_main.set_ylabel("Y")

    # Top histogram (X marginal)
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_top.hist(x, bins=30, density=True, alpha=0.7, color="#1f77b4", edgecolor="white")
    kde_x = gaussian_kde(x)
    x_kde = np.linspace(x.min(), x.max(), 100)
    ax_top.plot(x_kde, kde_x(x_kde), color="darkblue", linewidth=1.5)
    ax_top.set_ylabel("Density")
    ax_top.tick_params(labelbottom=False)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    # Right histogram (Y marginal)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_right.hist(y, bins=30, density=True, alpha=0.7, color="#1f77b4", orientation="horizontal", edgecolor="white")
    kde_y = gaussian_kde(y)
    y_kde = np.linspace(y.min(), y.max(), 100)
    ax_right.plot(kde_y(y_kde), y_kde, color="darkblue", linewidth=1.5)
    ax_right.set_xlabel("Density")
    ax_right.tick_params(labelleft=False)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)

    # Empty corner
    ax_empty = fig.add_subplot(gs[0, 1])
    ax_empty.axis("off")

    # Add correlation text
    corr = np.corrcoef(x, y)[0, 1]
    ax_empty.text(0.5, 0.5, f"r = {corr:.3f}", transform=ax_empty.transAxes,
                  fontsize=14, ha="center", va="center",
                  bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle("Joint Plot with Marginal Distributions", fontsize=12, y=0.98)

    ax_dict = {
        "main": ax_main,
        "top": ax_top,
        "right": ax_right,
    }

    return fig, ax_dict


