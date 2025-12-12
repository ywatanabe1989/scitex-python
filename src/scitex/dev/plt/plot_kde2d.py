#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_kde2d.py - 2D KDE (kernel density estimation)

"""
2D KDE plot - bivariate kernel density estimation visualization.
"""

import numpy as np
import scitex as stx


def plot_kde2d(plt, rng, ax=None):
    """2D KDE with contours and scatter overlay.

    Parameters
    ----------
    plt : module
        Plotting module
    rng : numpy.random.Generator
        Random number generator
    ax : Axes, optional
        Axes to plot on. If None, creates new figure.

    Returns
    -------
    fig : Figure
        The figure object
    ax : Axes
        The axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.get_figure() if hasattr(ax, 'get_figure') else ax._fig_scitex

    from scipy.stats import gaussian_kde

    # Generate bivariate data with two clusters
    n1, n2 = 300, 200
    # Cluster 1
    x1 = rng.normal(0, 1, n1)
    y1 = rng.normal(0, 1, n1)
    # Cluster 2
    x2 = rng.normal(3, 0.8, n2)
    y2 = rng.normal(2, 0.6, n2)

    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])

    # Compute 2D KDE
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)

    # Create grid for evaluation
    x_grid = np.linspace(x.min() - 1, x.max() + 1, 100)
    y_grid = np.linspace(y.min() - 1, y.max() + 1, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(X.shape)

    # Plot filled contours
    contourf = ax.contourf(X, Y, Z, levels=15, cmap="Blues", alpha=0.8)
    fig.colorbar(contourf, ax=ax, shrink=0.8, label="Density")

    # Plot contour lines
    ax.contour(X, Y, Z, levels=8, colors="darkblue", linewidths=0.5, alpha=0.6)

    # Overlay scatter points
    ax.scatter(x, y, s=5, alpha=0.3, color="black", label="Data points")

    ax.set_xyt("X", "Y", "2D KDE with Scatter Overlay")
    ax.legend()

    return fig, ax


