#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers/_12_stx_scatter.py

"""ax.stx_scatter(x, y) - Scatter plot wrapper."""

import numpy as np


def demo_stx_scatter(fig, ax, stx):
    """ax.stx_scatter(x, y) - Scatter plot wrapper."""
    np.random.seed(42)
    x = np.random.normal(0, 1, 50)
    y = 2 * x + np.random.normal(0, 0.5, 50)

    ax.stx_scatter(x, y, id="scatter", label="Data")

    ax.set_xyt(x="X [a.u.]", y="Y [a.u.]", t="ax.stx_scatter(x, y)")
    ax.legend()

    return fig, ax


# EOF
