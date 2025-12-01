#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:45:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_matplotlib_basic/_09_hexbin.py

"""ax.hexbin() - Hexagonal binning plot."""

import numpy as np


def demo_hexbin(fig, ax, stx):
    """ax.hexbin() - Hexagonal binning plot."""
    np.random.seed(42)
    x = np.random.randn(1000)
    y = x + np.random.randn(1000) * 0.5

    hb = ax.hexbin(x, y, gridsize=20, cmap="viridis", id="hexbin")
    stx.plt.utils.colorbar(hb, ax=ax._axes_mpl, label="Count")

    ax.set_xyt(x="X [a.u.]", y="Y [a.u.]", t="ax.hexbin(x, y)")

    return fig, ax


# EOF
