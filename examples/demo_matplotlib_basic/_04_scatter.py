#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:45:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_matplotlib_basic/_04_scatter.py

"""ax.scatter() - Scatter plot."""

import numpy as np


def demo_scatter(fig, ax, stx):
    """ax.scatter() - Scatter plot."""
    np.random.seed(42)
    x = np.random.normal(0, 1, 100)
    y = 2 * x + np.random.normal(0, 0.5, 100)

    ax.scatter(x, y, label="Data", id="scatter")

    # Add fitted line
    stx.plt.ax.add_fitted_line(ax, x, y, show_stats=True)

    ax.set_xyt(x="Predictor [a.u.]", y="Response [a.u.]", t="ax.scatter(x, y)")
    ax.legend(frameon=False, fontsize=6)

    return fig, ax


# EOF
