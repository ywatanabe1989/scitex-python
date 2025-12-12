#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_histogram_multiple.py - Multiple overlapping histograms

import numpy as np
import scitex as stx


def plot_histogram_multiple(plt, rng, ax=None):
    """Multiple overlapping histograms."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure() if hasattr(ax, 'get_figure') else ax._fig_scitex

    data1 = rng.normal(0, 1, 1000)
    data2 = rng.normal(2, 1.5, 1000)
    data3 = rng.normal(-1, 0.8, 1000)

    ax.hist(data1, bins=30, alpha=0.6, label="Distribution A", color="#1f77b4")
    ax.hist(data2, bins=30, alpha=0.6, label="Distribution B", color="#ff7f0e")
    ax.hist(data3, bins=30, alpha=0.6, label="Distribution C", color="#2ca02c")

    ax.set_xyt("Value", "Frequency", "Multiple Histograms")
    ax.legend()
    return fig, ax


