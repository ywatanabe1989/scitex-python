#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_mpl_hexbin.py - mpl_hexbin demo

"""mpl_hexbin: hexbin plot."""

import numpy as np


def plot_mpl_hexbin(plt, rng, ax=None):
    """mpl_hexbin - hexbin plot.

    Demonstrates: ax.mpl_hexbin() - identical to ax.hexbin()
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex

    x, y = rng.normal(0, 1, 1000), rng.normal(0, 1, 1000)
    ax.mpl_hexbin(x, y, gridsize=15)
    ax.set_xyt("X", "Y", "mpl_hexbin")
    if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
        ax.legend()
    return fig, ax


# EOF
