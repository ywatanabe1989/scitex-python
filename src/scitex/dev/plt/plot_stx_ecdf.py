#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_stx_ecdf.py - stx_ecdf demo

"""stx_ecdf: 1D array ECDF."""

import numpy as np


def plot_stx_ecdf(plt, rng, ax=None):
    """stx_ecdf - 1D array ECDF.

    Demonstrates: ax.stx_ecdf()
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex

    data = rng.normal(0, 1, 200)
    ax.stx_ecdf(data, label='ECDF')
    ax.set_xyt("X", "Y", "stx_ecdf")
    if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
        ax.legend()
    return fig, ax


# EOF
