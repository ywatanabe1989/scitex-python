#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_swarm.py - Swarm plot (beeswarm)

"""
Swarm plot (beeswarm) - non-overlapping scatter points for categorical data.
"""

import numpy as np
import scitex as stx


def plot_swarm(plt, rng, ax=None):
    """Swarm plot (beeswarm) - scatter with jittered non-overlapping points.

    Parameters
    ----------
    plt : module
        Plotting module
    rng : numpy.random.Generator
        Random number generator
    ax : Axes, optional
        Axes to plot on. If None, creates new figure.

    Returns
    -------
    fig : Figure
        The figure object
    ax : Axes
        The axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure() if hasattr(ax, 'get_figure') else ax._fig_scitex

    groups = ["Control", "Treatment A", "Treatment B", "Treatment C"]
    group_data = [
        rng.normal(0, 1, 40),
        rng.normal(1.5, 1.2, 35),
        rng.normal(0.5, 0.8, 45),
        rng.normal(2, 1.5, 30),
    ]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, (data, label, color) in enumerate(zip(group_data, groups, colors)):
        # Simple beeswarm: jitter based on density
        # Sort data and add position-dependent jitter
        sorted_idx = np.argsort(data)
        sorted_data = data[sorted_idx]

        # Calculate jitter based on local density
        jitter = np.zeros_like(data)
        for j in range(len(data)):
            # Count nearby points
            nearby = np.sum(np.abs(sorted_data - sorted_data[j]) < 0.3)
            # Alternate left/right based on index
            sign = 1 if j % 2 == 0 else -1
            jitter[sorted_idx[j]] = sign * 0.02 * (j % nearby)

        # Clip jitter to reasonable range
        jitter = np.clip(jitter, -0.3, 0.3)

        x_pos = np.full_like(data, i, dtype=float) + jitter
        ax.scatter(x_pos, data, s=25, alpha=0.7, color=color, edgecolors="white", linewidth=0.5)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=15, ha="right")
    ax.set_xyt("Group", "Value", "Swarm Plot")
    ax.grid(axis="y", alpha=0.3)

    return fig, ax


