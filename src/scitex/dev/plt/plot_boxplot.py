#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_boxplot.py - Box plot

import numpy as np
import scitex as stx


def plot_boxplot(plt, rng, ax=None):
    """Box plot."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure() if hasattr(ax, 'get_figure') else ax._fig_scitex

    data = [rng.normal(0, std, 100) for std in [1, 1.5, 2, 0.8, 1.2]]
    bp = ax.boxplot(data, patch_artist=True, labels=["A", "B", "C", "D", "E"])

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xyt("Group", "Value", "Box Plot")
    return fig, ax


