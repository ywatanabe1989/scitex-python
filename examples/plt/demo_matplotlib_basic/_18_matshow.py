#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:45:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_matplotlib_basic/_18_matshow.py

"""ax.imshow() - Image/matrix display."""

import numpy as np


def demo_matshow(fig, ax, stx):
    """ax.imshow() - Image/matrix display."""
    np.random.seed(42)
    data = np.random.rand(10, 10)

    # Use imshow with origin='lower' to have (0,0) at bottom-left
    ms = ax.imshow(data, cmap="viridis", origin="lower", aspect="equal")
    stx.plt.utils.colorbar(ms, ax=ax._axes_mpl, label="Value")

    ax.set_xyt(x="Column", y="Row", t="ax.imshow(data)")

    return fig, ax


# EOF
