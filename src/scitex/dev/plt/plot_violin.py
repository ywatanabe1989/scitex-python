#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_violin.py - Violin plot

import numpy as np
import scitex as stx


def plot_violin(plt, rng):
    """Violin plot."""
    fig, ax = plt.subplots(figsize=(8, 6))

    data = [rng.normal(loc, 1, 200) for loc in [0, 1, -0.5, 0.5, -1]]
    vp = ax.violinplot(
        data, positions=range(1, 6), showmeans=True, showmedians=True
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(colors[i])
        body.set_alpha(0.7)

    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(["A", "B", "C", "D", "E"])
    ax.set_xyt("Group", "Value", "Violin Plot")
    return fig, ax


