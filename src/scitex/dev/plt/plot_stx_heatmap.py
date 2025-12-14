#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_stx_heatmap.py - stx_heatmap demo

"""stx_heatmap: 2D array heatmap."""

import numpy as np


def plot_stx_heatmap(plt, rng, ax=None):
    """stx_heatmap - 2D array heatmap.

    Demonstrates: ax.stx_heatmap()
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex

    data = rng.uniform(0, 1, (5, 5))
    ax.stx_heatmap(data, annot=True, fmt='.2f')
    ax.set_xyt("X", "Y", "stx_heatmap")
    if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
        ax.legend()
    return fig, ax


# EOF
