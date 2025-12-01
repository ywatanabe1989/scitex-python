#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers/_16_plot_contour.py

"""ax.plot_contour(X, Y, Z) - Contour plot wrapper."""

import numpy as np


def demo_plot_contour(fig, ax, stx):
    """ax.plot_contour(X, Y, Z) - Contour plot wrapper."""
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y)

    ax.plot_contour(X, Y, Z, levels=8, id="contour")

    ax.set_xyt(x="X", y="Y", t="ax.plot_contour(X, Y, Z)")

    return fig, ax


# EOF
