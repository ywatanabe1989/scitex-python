#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_histogram.py - Single histogram

import numpy as np
import scitex as stx


def plot_histogram(plt, rng, ax=None):
    """Histogram - all bins should be grouped as one element."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure() if hasattr(ax, 'get_figure') else ax._fig_scitex

    data = rng.standard_normal(2000)
    ax.hist(data, bins=40, color="#1f77b4", edgecolor="white", alpha=0.8)

    ax.set_xyt("Value", "Frequency", "Histogram (All Bins Grouped)")
    return fig, ax


