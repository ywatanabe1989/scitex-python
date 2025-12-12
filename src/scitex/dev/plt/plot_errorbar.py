#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_errorbar.py - Error bar plot

import numpy as np
import scitex as stx


def plot_errorbar(plt, rng):
    """Error bar plot."""
    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(1, 8)
    y1 = rng.uniform(10, 20, len(x))
    y2 = rng.uniform(15, 25, len(x))
    yerr1 = rng.uniform(1, 3, len(x))
    yerr2 = rng.uniform(1, 3, len(x))

    ax.errorbar(
        x - 0.15,
        y1,
        yerr=yerr1,
        fmt="o-",
        capsize=5,
        label="Method A",
        color="#1f77b4",
        markersize=8,
    )
    ax.errorbar(
        x + 0.15,
        y2,
        yerr=yerr2,
        fmt="s-",
        capsize=5,
        label="Method B",
        color="#ff7f0e",
        markersize=8,
    )

    ax.set_xyt("Condition", "Measurement", "Error Bar Plot")
    ax.legend()
    return fig, ax


