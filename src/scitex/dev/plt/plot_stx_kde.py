#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_stx_kde.py - stx_kde demo

"""stx_kde: 1D array density."""

import numpy as np


def plot_stx_kde(plt, rng, ax=None):
    """stx_kde - 1D array density.

    Demonstrates: ax.stx_kde()
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex

    data = np.concatenate([rng.normal(-2, 0.5, 200), rng.normal(2, 1, 300)])
    ax.stx_kde(data, label='Bimodal')
    ax.set_xyt("X", "Y", "stx_kde")
    if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
        ax.legend()
    return fig, ax


# EOF
