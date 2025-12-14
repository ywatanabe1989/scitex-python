#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_mpl_hist2d.py - mpl_hist2d demo

"""mpl_hist2d: 2D histogram."""

import numpy as np


def plot_mpl_hist2d(plt, rng, ax=None):
    """mpl_hist2d - 2D histogram.

    Demonstrates: ax.mpl_hist2d() - identical to ax.hist2d()
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex

    x, y = rng.normal(0, 1, 1000), rng.normal(0, 1, 1000)
    ax.mpl_hist2d(x, y, bins=20)
    ax.set_xyt("X", "Y", "mpl_hist2d")
    if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
        ax.legend()
    return fig, ax


# EOF
