#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_joyplot.py - Joy plot (ridgeline plot)

"""
Joy plot (ridgeline plot) - overlapping density plots for distributions.

Also known as:
- Ridgeline plot
- Joyplot (after Joy Division album cover)
"""

import numpy as np
import scitex as stx


def plot_joyplot(plt, rng, ax=None):
    """Joy plot (ridgeline) - overlapping KDE distributions.

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
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure() if hasattr(ax, 'get_figure') else ax._fig_scitex

    from scipy.stats import gaussian_kde

    # Generate multiple distributions with shifting means
    n_distributions = 8
    labels = [f"Group {i+1}" for i in range(n_distributions)]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_distributions)) if hasattr(plt, 'cm') else None

    # Common x-axis for all KDEs
    x_range = np.linspace(-4, 10, 200)
    overlap = 0.6  # Overlap factor

    for i in range(n_distributions):
        # Generate data with shifting mean
        mean = i * 0.8
        std = 0.8 + 0.1 * rng.random()
        data = rng.normal(mean, std, 200)

        # Compute KDE
        kde = gaussian_kde(data)
        density = kde(x_range)

        # Normalize density for consistent height
        density = density / density.max() * 0.8

        # Vertical offset
        y_offset = i * (1 - overlap)

        # Fill under curve
        color = colors[i] if colors is not None else None
        ax.fill_between(x_range, y_offset, y_offset + density, alpha=0.7, color=color, edgecolor="white", linewidth=0.5)

        # Add line on top
        ax.plot(x_range, y_offset + density, color="black", linewidth=0.5)

    # Set y-ticks at the base of each distribution
    y_positions = [i * (1 - overlap) for i in range(n_distributions)]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Value")
    ax.set_title("Joy Plot (Ridgeline)")
    ax.set_xlim(x_range[0], x_range[-1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    return fig, ax


