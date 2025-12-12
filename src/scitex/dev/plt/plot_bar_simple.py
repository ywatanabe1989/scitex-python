#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_bar_simple.py - Simple categorical bar chart

import numpy as np
import scitex as stx


def plot_bar_simple(plt, rng, ax=None):
    """Simple categorical bar chart - each bar should be standalone."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure() if hasattr(ax, 'get_figure') else ax._fig_scitex
    categories = ["Category A", "Category B", "Category C", "Category D", "Category E"]
    values = rng.uniform(10, 30, len(categories))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    ax.bar(categories, values, color=colors, edgecolor="white", linewidth=2)
    ax.set_xyt("Category", "Value", "Simple Bar Chart (Each Bar Standalone)")
    return fig, ax


