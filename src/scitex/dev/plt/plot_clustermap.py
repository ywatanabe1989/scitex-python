#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_clustermap.py - Cluster map (heatmap with dendrograms)

"""
Cluster map - hierarchical clustering visualization with heatmap.
"""

import numpy as np
import scitex as stx


def plot_clustermap(plt, rng, ax=None):
    """Cluster map with row/column dendrograms.

    Note: This creates its own figure layout due to dendrogram requirements.
    The ax parameter is ignored.

    Parameters
    ----------
    plt : module
        Plotting module
    rng : numpy.random.Generator
        Random number generator
    ax : Axes, optional
        Ignored - clustermap creates its own layout.

    Returns
    -------
    fig : Figure
        The figure object
    ax_dict : dict
        Dictionary of axes: {"heatmap", "row_dendrogram", "col_dendrogram", "colorbar"}
    """
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist

    # Generate data with some structure
    n_rows, n_cols = 20, 15
    # Create data with block structure
    data = rng.standard_normal((n_rows, n_cols))
    # Add block patterns
    data[:5, :5] += 2
    data[10:15, 8:12] += 2
    data[15:, :3] -= 1.5

    # Row and column labels
    row_labels = [f"Gene_{i+1}" for i in range(n_rows)]
    col_labels = [f"Sample_{j+1}" for j in range(n_cols)]

    # Compute linkages
    row_linkage = linkage(pdist(data), method="ward")
    col_linkage = linkage(pdist(data.T), method="ward")

    # Create figure with gridspec
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[0.15, 1, 0.05],
        height_ratios=[0.15, 1],
        wspace=0.02,
        hspace=0.02,
    )

    # Column dendrogram (top)
    ax_col_dend = fig.add_subplot(gs[0, 1])
    col_dend = dendrogram(col_linkage, ax=ax_col_dend, orientation="top", no_labels=True, color_threshold=0)
    ax_col_dend.set_xticks([])
    ax_col_dend.set_yticks([])
    ax_col_dend.spines["top"].set_visible(False)
    ax_col_dend.spines["right"].set_visible(False)
    ax_col_dend.spines["bottom"].set_visible(False)
    ax_col_dend.spines["left"].set_visible(False)

    # Row dendrogram (left)
    ax_row_dend = fig.add_subplot(gs[1, 0])
    row_dend = dendrogram(row_linkage, ax=ax_row_dend, orientation="left", no_labels=True, color_threshold=0)
    ax_row_dend.set_xticks([])
    ax_row_dend.set_yticks([])
    ax_row_dend.spines["top"].set_visible(False)
    ax_row_dend.spines["right"].set_visible(False)
    ax_row_dend.spines["bottom"].set_visible(False)
    ax_row_dend.spines["left"].set_visible(False)

    # Reorder data according to dendrogram
    row_order = row_dend["leaves"]
    col_order = col_dend["leaves"]
    data_ordered = data[row_order, :][:, col_order]

    # Heatmap
    ax_heatmap = fig.add_subplot(gs[1, 1])
    im = ax_heatmap.imshow(data_ordered, aspect="auto", cmap="RdBu_r")
    ax_heatmap.set_xticks(range(n_cols))
    ax_heatmap.set_yticks(range(n_rows))
    ax_heatmap.set_xticklabels([col_labels[i] for i in col_order], rotation=45, ha="right", fontsize=7)
    ax_heatmap.set_yticklabels([row_labels[i] for i in row_order], fontsize=7)

    # Colorbar
    ax_cbar = fig.add_subplot(gs[1, 2])
    fig.colorbar(im, cax=ax_cbar, label="Value")

    # Empty top-left corner
    ax_empty = fig.add_subplot(gs[0, 0])
    ax_empty.axis("off")

    # Title
    fig.suptitle("Cluster Map (Hierarchical Clustering)", fontsize=12, y=0.98)

    ax_dict = {
        "heatmap": ax_heatmap,
        "row_dendrogram": ax_row_dend,
        "col_dendrogram": ax_col_dend,
        "colorbar": ax_cbar,
    }

    return fig, ax_dict


