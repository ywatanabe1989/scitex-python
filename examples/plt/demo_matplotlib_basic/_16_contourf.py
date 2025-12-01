#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:45:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_matplotlib_basic/_16_contourf.py

"""ax.contourf() - Filled contour plot."""

import numpy as np


def demo_contourf(fig, ax, stx):
    """ax.contourf() - Filled contour plot."""
    delta = 0.1
    x = np.arange(-2.0, 2.0, delta)
    y = np.arange(-2.0, 2.0, delta)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y)

    cf = ax.contourf(X, Y, Z, levels=8, cmap="viridis", id="contourf")
    stx.plt.utils.colorbar(cf, ax=ax._axes_mpl, label="Value")

    ax.set_xyt(x="X", y="Y", t="ax.contourf(X, Y, Z)")

    return fig, ax


# EOF
