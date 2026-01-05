#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_mpl_errorbar.py - mpl_errorbar demo

"""mpl_errorbar: error bar."""

import numpy as np


def plot_mpl_errorbar(plt, rng, ax=None):
    """mpl_errorbar - error bar.

    Demonstrates: ax.mpl_errorbar() - identical to ax.errorbar()
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex

    x = np.arange(1, 6)
    y = rng.uniform(2, 8, 5)
    yerr = rng.uniform(0.5, 1.5, 5)
    ax.mpl_errorbar(x, y, yerr=yerr, fmt='o-', capsize=3)
    ax.set_xyt("X", "Y", "mpl_errorbar")
    if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
        ax.legend()
    return fig, ax


# EOF
