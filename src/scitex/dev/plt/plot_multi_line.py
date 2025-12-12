#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_multi_line.py - Multiple overlapping lines with scatter overlay

import numpy as np
import scitex as stx


def plot_multi_line(plt, rng, ax=None):
    """Create plot with multiple overlapping lines."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure() if hasattr(ax, 'get_figure') else ax._fig_scitex
    x = np.linspace(0, 4 * np.pi, 200)

    ax.plot(x, np.sin(x), "-", linewidth=2, label="sin(x)", color="#1f77b4")
    ax.plot(x, np.cos(x), "--", linewidth=2, label="cos(x)", color="#ff7f0e")
    ax.plot(x, np.sin(x) * np.cos(x), ":", linewidth=3, label="sin*cos", color="#2ca02c")
    ax.plot(x, 0.5 * np.sin(2 * x), "-.", linewidth=2, label="0.5*sin(2x)", color="#d62728")

    sample_idx = np.arange(0, len(x), 20)
    ax.scatter(x[sample_idx], np.sin(x[sample_idx]), s=50, c="#1f77b4", marker="o", zorder=5)
    ax.scatter(x[sample_idx], np.cos(x[sample_idx]), s=50, c="#ff7f0e", marker="s", zorder=5)

    ax.set_xyt("x (radians)", "y", "Multi-Line Plot with Scatter Overlay")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    return fig, ax


