#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:45:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_matplotlib_basic/_21_streamplot.py

"""ax.streamplot() - Stream lines."""

import numpy as np


def demo_streamplot(fig, ax, stx):
    """ax.streamplot() - Stream lines."""
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    U = -Y
    V = X

    ax.streamplot(X, Y, U, V, id="stream")

    ax.set_xyt(x="X", y="Y", t="ax.streamplot(X, Y, U, V)")

    return fig, ax


# EOF
