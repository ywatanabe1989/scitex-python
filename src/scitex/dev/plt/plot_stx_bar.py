#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_stx_bar.py - stx_bar demo

"""stx_bar: x, height arrays."""

import numpy as np


def plot_stx_bar(plt, rng, ax=None):
    """stx_bar - x, height arrays.

    Demonstrates: ax.stx_bar()
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex

    x = [1, 2, 3, 4, 5]
    height = rng.uniform(2, 8, 5)
    ax.stx_bar(x, height, label='Values')
    ax.set_xyt("X", "Y", "stx_bar")
    if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
        ax.legend()
    return fig, ax


# EOF
