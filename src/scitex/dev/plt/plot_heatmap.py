#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_heatmap.py - Heatmap / imshow

import numpy as np
import scitex as stx


def plot_heatmap(plt, rng, ax=None):
    """Heatmap / imshow."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure() if hasattr(ax, 'get_figure') else ax._fig_scitex

    data = rng.uniform(0, 100, (8, 10))
    im = ax.imshow(data, cmap="hot", aspect="auto")
    fig.colorbar(im, ax=ax, label="Value")

    ax.set_xticks(range(10))
    ax.set_yticks(range(8))
    ax.set_xticklabels([f"C{i}" for i in range(10)])
    ax.set_yticklabels([f"R{i}" for i in range(8)])
    ax.set_xyt("Column", "Row", "Heatmap")
    return fig, ax


