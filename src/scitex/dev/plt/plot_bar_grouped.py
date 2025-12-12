#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_bar_grouped.py - Grouped bar chart

import numpy as np
import scitex as stx


def plot_bar_grouped(plt, rng, ax=None):
    """Grouped bar chart - bars should be grouped by series."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure() if hasattr(ax, 'get_figure') else ax._fig_scitex

    categories = ["A", "B", "C", "D", "E"]
    x = np.arange(len(categories))
    width = 0.25

    values1 = rng.uniform(10, 30, len(categories))
    values2 = rng.uniform(15, 35, len(categories))
    values3 = rng.uniform(5, 25, len(categories))

    ax.bar(x - width, values1, width, label="Series 1", color="#1f77b4")
    ax.bar(x, values2, width, label="Series 2", color="#ff7f0e")
    ax.bar(x + width, values3, width, label="Series 3", color="#2ca02c")

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_xyt("Category", "Value", "Grouped Bar Chart (Series Grouped)")
    ax.legend()
    return fig, ax


