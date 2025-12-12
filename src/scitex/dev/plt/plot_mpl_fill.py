#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_mpl_fill.py - mpl_fill demo

"""mpl_fill: filled polygon."""

import numpy as np


def plot_mpl_fill(plt, rng, ax=None):
    """mpl_fill - filled polygon.

    Demonstrates: ax.mpl_fill() - identical to ax.fill()
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex

    theta = np.linspace(0, 2*np.pi, 100)
    r = 1 + 0.3 * np.sin(5 * theta)
    x, y = r * np.cos(theta), r * np.sin(theta)
    ax.mpl_fill(x, y, alpha=0.5)
    ax.set_xyt("X", "Y", "mpl_fill")
    if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
        ax.legend()
    return fig, ax


# EOF
