#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_scatter_sizes.py - Scatter plot with varying sizes and colors

import numpy as np
import scitex as stx


def plot_scatter_sizes(plt, rng):
    """Scatter plot with varying sizes and colors."""
    fig, ax = plt.subplots(figsize=(8, 6))

    n_points = 50
    x = rng.uniform(0, 10, n_points)
    y = rng.uniform(0, 10, n_points)
    sizes = rng.uniform(20, 200, n_points)
    colors = rng.uniform(0, 1, n_points)

    scatter = ax.scatter(
        x,
        y,
        s=sizes,
        c=colors,
        cmap="viridis",
        alpha=0.7,
        edgecolors="white",
        linewidths=1,
    )
    fig.colorbar(scatter, ax=ax, label="Value")

    ax.set_xyt("X", "Y", "Scatter with Size and Color Encoding")
    return fig, ax


