#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_pairplot.py - Pair plot (scatter matrix)

"""
Pair plot - grid of scatter plots and histograms for multivariate data.
"""

import numpy as np
import scitex as stx


def plot_pairplot(plt, rng, ax=None):
    """Pair plot (scatter matrix) for multivariate data.

    Note: This creates its own figure layout.
    The ax parameter is ignored.

    Parameters
    ----------
    plt : module
        Plotting module
    rng : numpy.random.Generator
        Random number generator
    ax : Axes, optional
        Ignored - pairplot creates its own layout.

    Returns
    -------
    fig : Figure
        The figure object
    axes : ndarray
        2D array of axes (n_vars x n_vars)
    """
    from scipy.stats import gaussian_kde

    # Generate correlated multivariate data
    n_samples = 200
    n_vars = 4
    var_names = ["X", "Y", "Z", "W"]

    # Correlation structure
    cov = np.array([
        [1.0, 0.7, 0.3, -0.2],
        [0.7, 1.0, 0.5, 0.1],
        [0.3, 0.5, 1.0, 0.4],
        [-0.2, 0.1, 0.4, 1.0],
    ])
    mean = [0, 1, 2, 3]
    data = rng.multivariate_normal(mean, cov, n_samples)

    # Create grid
    fig, axes = plt.subplots(n_vars, n_vars, figsize=(10, 10))

    for i in range(n_vars):
        for j in range(n_vars):
            ax = axes[i, j]

            if i == j:
                # Diagonal: histogram/KDE
                ax.hist(data[:, i], bins=20, density=True, alpha=0.7, color="#1f77b4")
                # Add KDE line
                kde = gaussian_kde(data[:, i])
                x_kde = np.linspace(data[:, i].min(), data[:, i].max(), 100)
                ax.plot(x_kde, kde(x_kde), color="darkblue", linewidth=1.5)
            else:
                # Off-diagonal: scatter
                ax.scatter(data[:, j], data[:, i], s=10, alpha=0.5, color="#1f77b4")

            # Labels
            if i == n_vars - 1:
                ax.set_xlabel(var_names[j])
            else:
                ax.set_xticklabels([])

            if j == 0:
                ax.set_ylabel(var_names[i])
            else:
                ax.set_yticklabels([])

    fig.suptitle("Pair Plot (Scatter Matrix)", fontsize=12, y=1.02)
    fig.tight_layout()

    return fig, axes


