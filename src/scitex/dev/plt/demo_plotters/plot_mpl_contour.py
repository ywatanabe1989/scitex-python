#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_mpl_contour.py - mpl_contour demo

"""mpl_contour: contour plot."""

import numpy as np


def plot_mpl_contour(plt, rng, ax=None):
    """mpl_contour - contour plot.

    Demonstrates: ax.mpl_contour() - identical to ax.contour()
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex

    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-(X**2 + Y**2))
    ax.mpl_contour(X, Y, Z, levels=10)
    ax.set_xyt("X", "Y", "mpl_contour")
    if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
        ax.legend()
    return fig, ax


# EOF
