#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_heatmap_annotated.py - Annotated heatmap with values

"""
Annotated heatmap - heatmap with cell values displayed.
"""

import numpy as np
import scitex as stx


def plot_heatmap_annotated(plt, rng, ax=None):
    """Annotated heatmap with cell values.

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

    # Create correlation-like matrix
    n = 8
    data = rng.uniform(-1, 1, (n, n))
    # Make it symmetric
    data = (data + data.T) / 2
    # Set diagonal to 1
    np.fill_diagonal(data, 1.0)

    labels = [f"Var{i+1}" for i in range(n)]

    # Plot heatmap
    im = ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Correlation")

    # Add text annotations
    for i in range(n):
        for j in range(n):
            value = data[i, j]
            # Choose text color based on background
            text_color = "white" if abs(value) > 0.5 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8)

    # Set ticks and labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_title("Annotated Correlation Heatmap")

    return fig, ax


