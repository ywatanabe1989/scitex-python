#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_fill_between.py - Fill between areas

import numpy as np
import scitex as stx


def plot_fill_between(plt, rng, ax=None):
    """Fill between areas."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure() if hasattr(ax, 'get_figure') else ax._fig_scitex
    x = np.linspace(0, 10, 100)

    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = 0.5 * np.sin(2 * x)

    ax.fill_between(
        x, y1, y2, alpha=0.3, label="sin-cos region", color="#1f77b4"
    )
    ax.fill_between(
        x, y2, y3, alpha=0.3, label="cos-sin2x region", color="#ff7f0e"
    )
    ax.fill_between(
        x, y3, -1, alpha=0.3, label="sin2x-bottom region", color="#2ca02c"
    )

    ax.plot(x, y1, "-", linewidth=2, color="#1f77b4")
    ax.plot(x, y2, "-", linewidth=2, color="#ff7f0e")
    ax.plot(x, y3, "-", linewidth=2, color="#2ca02c")

    ax.set_xyt("x", "y", "Fill Between Plot")
    ax.legend()
    return fig, ax


