#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_bar_stacked.py - Stacked bar chart

import numpy as np
import scitex as stx


def plot_bar_stacked(plt, rng, ax=None):
    """Stacked bar chart."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure() if hasattr(ax, 'get_figure') else ax._fig_scitex

    categories = ["Q1", "Q2", "Q3", "Q4"]
    values1 = rng.uniform(10, 20, len(categories))
    values2 = rng.uniform(10, 20, len(categories))
    values3 = rng.uniform(10, 20, len(categories))

    ax.bar(categories, values1, label="Product A", color="#1f77b4")
    ax.bar(categories, values2, bottom=values1, label="Product B", color="#ff7f0e")
    ax.bar(categories, values3, bottom=values1 + values2, label="Product C", color="#2ca02c")

    ax.set_xyt("Quarter", "Revenue", "Stacked Bar Chart")
    ax.legend()
    return fig, ax


