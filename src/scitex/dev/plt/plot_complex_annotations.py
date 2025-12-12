#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_complex_annotations.py - Plot with annotations and markers

import numpy as np
import scitex as stx


def plot_complex_annotations(plt, rng, ax=None):
    """Plot with annotations and markers."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure() if hasattr(ax, 'get_figure') else ax._fig_scitex
    x = np.linspace(0, 10, 100)
    y = np.sin(x) * np.exp(-x / 10)

    ax.plot(x, y, "-", linewidth=2, label="Damped sine", color="#1f77b4")

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    ax.axvline(x=np.pi, color="red", linestyle=":", linewidth=2, label="x=pi")

    peaks_x = [np.pi / 2, 5 * np.pi / 2]
    peaks_y = [np.sin(px) * np.exp(-px / 10) for px in peaks_x]
    ax.scatter(
        peaks_x, peaks_y, s=100, c="red", marker="*", zorder=10, label="Peaks"
    )

    from matplotlib.patches import Rectangle

    rect = Rectangle(
        (6, -0.2),
        2,
        0.4,
        fill=True,
        facecolor="yellow",
        edgecolor="orange",
        alpha=0.5,
        linewidth=2,
    )
    ax.add_patch(rect)

    ax.set_xyt("x", "y", "Damped Sine with Annotations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig, ax


