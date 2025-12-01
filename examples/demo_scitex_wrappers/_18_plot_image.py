#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers/_18_plot_image.py

"""ax.plot_image(data) - Image file display."""

import numpy as np


def demo_plot_image(fig, ax, stx):
    """ax.plot_image(data) - Image file display."""
    # Create a sample 2D grayscale image array (plot_image requires 2D)
    np.random.seed(42)
    img = np.random.rand(50, 50)

    ax.plot_image(img, id="image")

    ax.set_xyt(t="ax.plot_image(data)")

    return fig, ax


# EOF
