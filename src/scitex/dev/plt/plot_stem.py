#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_stem.py - Stem plot

import numpy as np
import scitex as stx


def plot_stem(plt, rng, ax=None):
    """Stem plot."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure() if hasattr(ax, 'get_figure') else ax._fig_scitex

    x = np.arange(0, 10, 0.5)
    y = np.sin(x)

    markerline, stemlines, baseline = ax.stem(x, y)
    markerline.set_color("#ff7f0e")
    stemlines.set_color("#ff7f0e")
    ax.set_xyt("x", "y", "Stem Plot")
    return fig, ax


