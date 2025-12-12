#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_parallel.py - Parallel coordinates plot

"""
Parallel coordinates plot for multivariate data visualization.
"""

import numpy as np
import scitex as stx


def plot_parallel(plt, rng, ax=None):
    """Parallel coordinates plot for multivariate data.

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
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure() if hasattr(ax, 'get_figure') else ax._fig_scitex

    # Generate multivariate data with clusters
    n_samples = 100
    n_vars = 6
    var_names = ["Var A", "Var B", "Var C", "Var D", "Var E", "Var F"]

    # Create 3 clusters
    cluster_means = [
        [0.2, 0.8, 0.3, 0.7, 0.5, 0.4],
        [0.7, 0.3, 0.8, 0.2, 0.6, 0.8],
        [0.5, 0.5, 0.5, 0.5, 0.2, 0.3],
    ]
    cluster_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    cluster_labels = ["Cluster A", "Cluster B", "Cluster C"]

    # Generate data
    for cluster_idx, (means, color, label) in enumerate(zip(cluster_means, cluster_colors, cluster_labels)):
        n_cluster = n_samples // 3
        data = np.array([rng.normal(m, 0.1, n_cluster) for m in means]).T

        # Normalize to 0-1 for each variable
        data = np.clip(data, 0, 1)

        # Plot each sample
        x = np.arange(n_vars)
        for i, sample in enumerate(data):
            alpha = 0.3 if i > 0 else 0.8
            label_used = label if i == 0 else None
            ax.plot(x, sample, color=color, alpha=alpha, linewidth=1, label=label_used)

    # Set axis properties
    ax.set_xticks(range(n_vars))
    ax.set_xticklabels(var_names, rotation=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Normalized Value")
    ax.set_title("Parallel Coordinates Plot")
    ax.legend(loc="upper right")

    # Add vertical lines at each variable
    for i in range(n_vars):
        ax.axvline(i, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    ax.grid(axis="y", alpha=0.3)

    return fig, ax


