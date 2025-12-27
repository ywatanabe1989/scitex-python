#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_mpl_axhspan.py - mpl_axhspan demo

"""mpl_axhspan: horizontal span."""

import numpy as np


def plot_mpl_axhspan(plt, rng, ax=None):
    """mpl_axhspan - horizontal span.

    Demonstrates: ax.mpl_axhspan() - identical to ax.axhspan()
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex

    ax.plot(rng.uniform(0, 10, 20))
    ax.mpl_axhspan(3, 7, alpha=0.3, color='yellow', label='range')
    ax.set_xyt("X", "Y", "mpl_axhspan")
    if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
        ax.legend()
    return fig, ax


# EOF
