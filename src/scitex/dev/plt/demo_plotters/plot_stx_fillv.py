#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_stx_fillv.py - stx_fillv demo

"""stx_fillv: vertical fill region."""

import numpy as np


def plot_stx_fillv(plt, rng, ax=None):
    """stx_fillv - vertical fill region.

    Demonstrates: ax.stx_fillv()
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex

    ax.plot(np.sin(np.linspace(0, 4*np.pi, 100)))
    ax.stx_fillv([20, 60], [40, 80], alpha=0.3)
    ax.set_xyt("X", "Y", "stx_fillv")
    if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
        ax.legend()
    return fig, ax


# EOF
