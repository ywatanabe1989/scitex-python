#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_mpl_bar.py - mpl_bar demo

"""mpl_bar: bar plot."""

import numpy as np


def plot_mpl_bar(plt, rng, ax=None):
    """mpl_bar - bar plot.

    Demonstrates: ax.mpl_bar() - identical to ax.bar()
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex

    x = [1, 2, 3, 4, 5]
    height = rng.uniform(2, 8, 5)
    ax.mpl_bar(x, height)
    ax.set_xyt("X", "Y", "mpl_bar")
    if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
        ax.legend()
    return fig, ax


# EOF
