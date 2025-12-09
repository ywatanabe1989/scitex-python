#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-02 09:03:23 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex_repo/src/scitex/plt/ax/_plot/_plot_joyplot.py
# ----------------------------------------
import os

__FILE__ = "./src/scitex/plt/ax/_plot/_plot_joyplot.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import numpy as np
from scipy import stats

from ....plt.utils import assert_valid_axis


def stx_joyplot(
    ax, arrays, overlap=0.5, fill_alpha=0.7, line_alpha=1.0, colors=None, **kwargs
):
    """
    Create a joyplot (ridgeline plot) on the provided axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    arrays : list of array-like
        List of 1D arrays for each ridge
    overlap : float, default 0.5
        Amount of overlap between ridges (0 = no overlap, 1 = full overlap)
    fill_alpha : float, default 0.7
        Alpha for the filled KDE area
    line_alpha : float, default 1.0
        Alpha for the KDE line
    colors : list, optional
        Colors for each ridge. If None, uses scitex palette.
    **kwargs
        Additional keyword arguments

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the joyplot
    """
    assert_valid_axis(
        ax, "First argument must be a matplotlib axis or scitex axis wrapper"
    )

    # Convert dict to list of arrays (values only)
    if isinstance(arrays, dict):
        arrays = list(arrays.values())

    # Add sample size per distribution to label if provided (show range if variable)
    if kwargs.get("label"):
        n_per_dist = [len(arr) for arr in arrays]
        n_min, n_max = min(n_per_dist), max(n_per_dist)
        n_str = str(n_min) if n_min == n_max else f"{n_min}-{n_max}"
        kwargs["label"] = f"{kwargs['label']} ($n$={n_str})"

    # Import scitex colors
    from scitex.plt.color._PARAMS import HEX

    # Default colors from scitex palette
    if colors is None:
        colors = [
            HEX["blue"],
            HEX["red"],
            HEX["green"],
            HEX["yellow"],
            HEX["purple"],
            HEX["orange"],
            HEX["lightblue"],
            HEX["pink"],
        ]

    n_ridges = len(arrays)

    # Calculate global x range
    all_data = np.concatenate([np.asarray(arr) for arr in arrays])
    x_min, x_max = np.min(all_data), np.max(all_data)
    x_range = x_max - x_min
    x_padding = x_range * 0.1
    x = np.linspace(x_min - x_padding, x_max + x_padding, 200)

    # Calculate KDEs and find max density for scaling
    kdes = []
    max_density = 0
    for arr in arrays:
        arr = np.asarray(arr)
        if len(arr) > 1:
            kde = stats.gaussian_kde(arr)
            density = kde(x)
            kdes.append(density)
            max_density = max(max_density, np.max(density))
        else:
            kdes.append(np.zeros_like(x))

    # Scale factor for ridge height
    ridge_height = 1.0 / (1.0 - overlap * 0.5) if overlap < 1 else 2.0

    # Plot each ridge from back to front
    for i in range(n_ridges - 1, -1, -1):
        color = colors[i % len(colors)]
        baseline = i * (1.0 - overlap)

        # Scale density to fit nicely
        scaled_density = (
            kdes[i] / max_density * ridge_height if max_density > 0 else kdes[i]
        )

        # Fill
        ax.fill_between(
            x,
            baseline,
            baseline + scaled_density,
            facecolor=color,
            edgecolor="none",
            alpha=fill_alpha,
        )
        # Line on top
        ax.plot(
            x, baseline + scaled_density, color=color, alpha=line_alpha, linewidth=1.0
        )

    # Set y limits
    ax.set_ylim(-0.1, n_ridges * (1.0 - overlap) + ridge_height)

    # Hide y-axis ticks for cleaner look (joyplots typically don't show y values)
    ax.set_yticks([])

    return ax


# EOF
