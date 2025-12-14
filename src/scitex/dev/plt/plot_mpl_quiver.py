#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_mpl_quiver.py - mpl_quiver demo

"""mpl_quiver: vector field."""

import numpy as np


def plot_mpl_quiver(plt, rng, ax=None):
    """mpl_quiver - vector field.

    Demonstrates: ax.mpl_quiver() - identical to ax.quiver()
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex

    x = np.arange(0, 5, 0.5)
    y = np.arange(0, 5, 0.5)
    X, Y = np.meshgrid(x, y)
    U, V = np.cos(X), np.sin(Y)
    ax.mpl_quiver(X, Y, U, V)
    ax.set_xyt("X", "Y", "mpl_quiver")
    if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
        ax.legend()
    return fig, ax


# EOF
