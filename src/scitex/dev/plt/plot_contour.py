#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_contour.py - Contour plot

import numpy as np
import scitex as stx


def plot_contour(plt, rng, ax=None):
    """Contour plot - levels should be grouped."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure() if hasattr(ax, 'get_figure') else ax._fig_scitex

    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-(X**2 + Y**2)) + 0.5 * np.exp(-((X - 1.5) ** 2 + (Y - 1) ** 2))

    cs = ax.contourf(X, Y, Z, levels=15, cmap="viridis")
    ax.contour(X, Y, Z, levels=15, colors="k", linewidths=0.5, alpha=0.5)
    fig.colorbar(cs, ax=ax, label="Value")

    ax.set_xyt("X", "Y", "Contour Plot (Levels Grouped)")
    return fig, ax


