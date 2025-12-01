#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:45:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_matplotlib_basic/_15_contour.py

"""ax.contour() - Contour plot."""

import numpy as np


def demo_contour(fig, ax, stx):
    """ax.contour() - Contour plot."""
    delta = 0.1
    x = np.arange(-2.0, 2.0, delta)
    y = np.arange(-2.0, 2.0, delta)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y)

    contour = ax.contour(X, Y, Z, levels=8, id="contour")
    ax.clabel(contour, inline=True, fontsize=6)

    ax.set_xyt(x="X", y="Y", t="ax.contour(X, Y, Z)")

    return fig, ax


# EOF
