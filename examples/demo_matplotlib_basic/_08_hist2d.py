#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:45:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_matplotlib_basic/_08_hist2d.py

"""ax.hist2d() - 2D histogram."""

import numpy as np


def demo_hist2d(fig, ax, stx):
    """ax.hist2d() - 2D histogram."""
    np.random.seed(42)
    x = np.random.randn(1000)
    y = x + np.random.randn(1000) * 0.5

    h = ax.hist2d(x, y, bins=30, cmap="viridis", id="hist2d")
    stx.plt.utils.colorbar(h[3], ax=ax._axes_mpl, label="Count")

    ax.set_xyt(x="X [a.u.]", y="Y [a.u.]", t="ax.hist2d(x, y)")

    return fig, ax


# EOF
