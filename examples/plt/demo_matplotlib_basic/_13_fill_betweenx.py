#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:45:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_matplotlib_basic/_13_fill_betweenx.py

"""ax.fill_betweenx() - Horizontal fill between."""

import numpy as np


def demo_fill_betweenx(fig, ax, stx):
    """ax.fill_betweenx() - Horizontal fill between."""
    y = np.linspace(0, 10, 100)
    x_mean = np.sin(y)
    x_std = 0.2

    ax.fill_betweenx(y, x_mean - x_std, x_mean + x_std, id="fillx")
    ax.plot(x_mean, y, id="line")

    ax.set_xyt(x="Value [a.u.]", y="Position [a.u.]", t="ax.fill_betweenx(y, x1, x2)")

    return fig, ax


# EOF
