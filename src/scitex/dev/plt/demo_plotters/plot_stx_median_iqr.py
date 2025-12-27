#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_stx_median_iqr.py - stx_median_iqr demo

"""stx_median_iqr: 2D array with IQR."""

import numpy as np


def plot_stx_median_iqr(plt, rng, ax=None):
    """stx_median_iqr - 2D array with IQR.

    Demonstrates: ax.stx_median_iqr()
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex

    data = rng.normal(0, 1, (100, 50)) + np.linspace(0, 2, 50)
    ax.stx_median_iqr(data, label='Median +/- IQR')
    ax.set_xyt("X", "Y", "stx_median_iqr")
    if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
        ax.legend()
    return fig, ax


# EOF
