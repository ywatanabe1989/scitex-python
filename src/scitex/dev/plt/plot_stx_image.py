#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_stx_image.py - stx_image demo

"""stx_image: 2D array image."""

import numpy as np


def plot_stx_image(plt, rng, ax=None):
    """stx_image - 2D array image.

    Demonstrates: ax.stx_image()
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex

    data = rng.uniform(0, 1, (10, 10))
    ax.stx_image(data, cmap='viridis')
    ax.set_xyt("X", "Y", "stx_image")
    if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
        ax.legend()
    return fig, ax


# EOF
