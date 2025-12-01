#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:45:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_matplotlib_basic/_20_quiver.py

"""ax.quiver() - Vector field."""

import numpy as np


def demo_quiver(fig, ax, stx):
    """ax.quiver() - Vector field."""
    x = np.arange(-2, 2.5, 0.5)
    y = np.arange(-2, 2.5, 0.5)
    X, Y = np.meshgrid(x, y)
    U = -Y
    V = X

    ax.quiver(X, Y, U, V, id="quiver")

    ax.set_xyt(x="X", y="Y", t="ax.quiver(X, Y, U, V)")

    return fig, ax


# EOF
