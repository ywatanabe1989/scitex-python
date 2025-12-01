#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers/_17_stx_imshow.py

"""ax.stx_imshow(data) - Image display wrapper."""

import numpy as np


def demo_stx_imshow(fig, ax, stx):
    """ax.stx_imshow(data) - Image display wrapper."""
    np.random.seed(42)
    data = np.random.rand(20, 30)

    ax.stx_imshow(data, id="imshow")

    ax.set_xyt(t="ax.stx_imshow(data)")

    return fig, ax


# EOF
