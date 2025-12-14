#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_stx_fill_between.py - stx_fill_between demo

"""stx_fill_between: x, y1, y2 arrays."""

import numpy as np


def plot_stx_fill_between(plt, rng, ax=None):
    """stx_fill_between - x, y1, y2 arrays.

    Demonstrates: ax.stx_fill_between()
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex

    x = np.linspace(0, 2*np.pi, 100)
    y1, y2 = np.sin(x), np.sin(x) + 0.5
    ax.stx_fill_between(x, y1, y2, alpha=0.3, label='Region')
    ax.plot(x, y1)
    ax.plot(x, y2)
    ax.set_xyt("X", "Y", "stx_fill_between")
    if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
        ax.legend()
    return fig, ax


# EOF
