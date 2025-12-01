#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers/_16_stx_contour.py

"""ax.stx_contour(X, Y, Z) - Contour plot wrapper."""

import numpy as np


def demo_stx_contour(fig, ax, stx):
    """ax.stx_contour(X, Y, Z) - Contour plot wrapper."""
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y)

    ax.stx_contour(X, Y, Z, id="contour")

    ax.set_xyt(x="X", y="Y", t="ax.stx_contour(X, Y, Z)")

    return fig, ax


# EOF
