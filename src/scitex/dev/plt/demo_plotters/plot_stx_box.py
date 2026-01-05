#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_stx_box.py - stx_box demo

"""stx_box: list of arrays."""

import numpy as np


def plot_stx_box(plt, rng, ax=None):
    """stx_box - list of arrays.

    Demonstrates: ax.stx_box()
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex

    data = [rng.normal(i, 1, 100) for i in range(4)]
    ax.stx_box(data, labels=['A', 'B', 'C', 'D'])
    ax.set_xyt("X", "Y", "stx_box")
    if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
        ax.legend()
    return fig, ax


# EOF
