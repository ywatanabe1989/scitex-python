#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_mpl_stackplot.py - mpl_stackplot demo

"""mpl_stackplot: stacked area."""

import numpy as np


def plot_mpl_stackplot(plt, rng, ax=None):
    """mpl_stackplot - stacked area.

    Demonstrates: ax.mpl_stackplot() - identical to ax.stackplot()
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex

    x = np.arange(10)
    y1 = rng.uniform(1, 3, 10)
    y2 = rng.uniform(1, 3, 10)
    y3 = rng.uniform(1, 3, 10)
    ax.mpl_stackplot(x, y1, y2, y3, labels=['A', 'B', 'C'])
    ax.set_xyt("X", "Y", "mpl_stackplot")
    if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
        ax.legend()
    return fig, ax


# EOF
