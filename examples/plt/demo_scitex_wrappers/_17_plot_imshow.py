#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers/_17_plot_imshow.py

"""ax.plot_imshow(data) - Image display wrapper."""

import numpy as np


def demo_plot_imshow(fig, ax, stx):
    """ax.plot_imshow(data) - Image display wrapper."""
    np.random.seed(42)
    data = np.random.rand(20, 30)

    ax.plot_imshow(data, id="imshow")

    ax.set_xyt(t="ax.plot_imshow(data)")

    return fig, ax


# EOF
