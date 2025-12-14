#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_mpl_plot.py - mpl_plot demo

"""mpl_plot: line plot."""

import numpy as np


def plot_mpl_plot(plt, rng, ax=None):
    """mpl_plot - line plot.

    Demonstrates: ax.mpl_plot() - identical to ax.plot()
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex

    x = np.linspace(0, 2*np.pi, 100)
    ax.mpl_plot(x, np.sin(x), '-', label='sin(x)')
    ax.mpl_plot(x, np.cos(x), '--', label='cos(x)')
    ax.set_xyt("X", "Y", "mpl_plot")
    if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
        ax.legend()
    return fig, ax


# EOF
