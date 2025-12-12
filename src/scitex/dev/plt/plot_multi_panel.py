#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_multi_panel.py - Multi-panel figure with mixed types

import numpy as np
import scitex as stx


def plot_multi_panel(plt, rng):
    """Multi-panel figure with mixed types."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    # Panel 1: Line plot
    x = np.linspace(0, 2 * np.pi, 100)
    axes[0].plot(x, np.sin(x), label="sin")
    axes[0].plot(x, np.cos(x), label="cos")
    axes[0].set_xyt("x", "y", "Trigonometric Functions")
    axes[0].legend()

    # Panel 2: Simple bar chart (each bar standalone)
    axes[1].bar(
        ["A", "B", "C", "D"],
        rng.uniform(5, 20, 4),
        color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
    )
    axes[1].set_xyt("Category", "Value", "Simple Bar")

    # Panel 3: Scatter
    axes[2].scatter(rng.uniform(0, 10, 30), rng.uniform(0, 10, 30), s=50)
    axes[2].set_xyt("X", "Y", "Scatter")

    # Panel 4: Histogram (bins grouped)
    data = rng.standard_normal(1000)
    axes[3].hist(data, bins=30, edgecolor="white")
    axes[3].set_xyt("Value", "Count", "Histogram")

    fig.tight_layout()
    return fig, axes[0]


