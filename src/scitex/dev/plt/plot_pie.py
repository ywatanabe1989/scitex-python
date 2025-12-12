#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_pie.py - Pie chart

import numpy as np
import scitex as stx


def plot_pie(plt, rng, ax=None):
    """Pie chart - wedges should be grouped."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.get_figure() if hasattr(ax, 'get_figure') else ax._fig_scitex

    sizes = rng.uniform(10, 30, 6)
    labels = [
        "Category A",
        "Category B",
        "Category C",
        "Category D",
        "Category E",
        "Category F",
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    explode = (0.05, 0, 0, 0.1, 0, 0)

    ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        shadow=True,
        startangle=90,
    )
    ax.set_title("Pie Chart (Wedges Grouped)")
    return fig, ax


