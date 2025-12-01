#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers/_18_stx_image.py

"""ax.stx_image(data) - Image file display."""

import numpy as np


def demo_stx_image(fig, ax, stx):
    """ax.stx_image(data) - Image file display."""
    # Create a sample 2D grayscale image array (stx_image requires 2D)
    np.random.seed(42)
    img = np.random.rand(50, 50)

    ax.stx_image(img, id="image")

    ax.set_xyt(t="ax.stx_image(data)")

    return fig, ax


# EOF
