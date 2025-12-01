#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:45:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_matplotlib_basic/_17_imshow.py

"""ax.imshow() - Image display."""

import numpy as np


def demo_imshow(fig, ax, stx):
    """ax.imshow() - Image display."""
    np.random.seed(42)
    data = np.random.rand(20, 30)

    im = ax.imshow(data, cmap="viridis", aspect="auto", id="imshow")
    ax.spines[:].set_visible(True)
    stx.plt.utils.colorbar(im, ax=ax._axes_mpl, label="Intensity")

    ax.set_xyt(t="ax.imshow(data)")
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax


# EOF
