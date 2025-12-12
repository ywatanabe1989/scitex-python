#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_step_stem.py - Step and stem plots

import numpy as np
import scitex as stx


def plot_step_stem(plt, rng):
    """Step and stem plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(0, 10, 0.5)
    y = np.sin(x)

    axes[0].step(x, y, where="mid", linewidth=2, color="#1f77b4", label="step")
    axes[0].set_xyt("x", "y", "Step Plot")
    axes[0].legend()

    markerline, stemlines, baseline = axes[1].stem(x, y)
    markerline.set_color("#ff7f0e")
    stemlines.set_color("#ff7f0e")
    axes[1].set_xyt("x", "y", "Stem Plot")

    fig.tight_layout()
    return fig, axes[0]


