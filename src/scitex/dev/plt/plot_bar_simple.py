#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_bar_simple.py - Simple categorical bar chart

import numpy as np
import scitex as stx


def plot_bar_simple(plt, rng):
    """Simple categorical bar chart - each bar should be standalone."""
    fig, ax = plt.subplots(figsize=(8, 6))
    categories = ["Category A", "Category B", "Category C", "Category D", "Category E"]
    values = rng.uniform(10, 30, len(categories))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    ax.bar(categories, values, color=colors, edgecolor="white", linewidth=2)
    ax.set_xyt("Category", "Value", "Simple Bar Chart (Each Bar Standalone)")
    return fig, ax


