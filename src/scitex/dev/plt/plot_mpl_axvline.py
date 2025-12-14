#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_mpl_axvline.py - mpl_axvline demo

"""mpl_axvline: vertical line."""

import numpy as np


def plot_mpl_axvline(plt, rng, ax=None):
    """mpl_axvline - vertical line.

    Demonstrates: ax.mpl_axvline() - identical to ax.axvline()
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex

    ax.plot(rng.uniform(0, 10, 20))
    ax.mpl_axvline(x=10, color='r', linestyle='--', label='event')
    ax.set_xyt("X", "Y", "mpl_axvline")
    if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
        ax.legend()
    return fig, ax


# EOF
