#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_step.py - Step plot

import numpy as np
import scitex as stx


def plot_step(plt, rng, ax=None):
    """Step plot."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure() if hasattr(ax, 'get_figure') else ax._fig_scitex

    x = np.arange(0, 10, 0.5)
    y = np.sin(x)

    ax.step(x, y, where="mid", linewidth=2, color="#1f77b4", label="step")
    ax.set_xyt("x", "y", "Step Plot")
    ax.legend()
    return fig, ax


