#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_stx_imshow.py - stx_imshow demo

"""stx_imshow: 2D array imshow."""

import numpy as np


def plot_stx_imshow(plt, rng, ax=None):
    """stx_imshow - 2D array imshow.

    Demonstrates: ax.stx_imshow()
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex

    data = rng.uniform(0, 1, (10, 10))
    ax.stx_imshow(data, cmap='viridis')
    ax.set_xyt("X", "Y", "stx_imshow")
    if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
        ax.legend()
    return fig, ax


# EOF
