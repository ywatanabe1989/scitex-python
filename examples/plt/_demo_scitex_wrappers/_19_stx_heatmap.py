#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers/_19_stx_heatmap.py

"""ax.stx_heatmap(values_2d) - Heatmap with annotations."""

import numpy as np


def demo_stx_heatmap(fig, ax, stx):
    """ax.stx_heatmap(values_2d) - Heatmap with annotations."""
    np.random.seed(42)
    values_2d = np.random.rand(5, 5)  # Square matrix for cleaner display
    labels = ["A", "B", "C", "D", "E"]

    ax.stx_heatmap(
        values_2d,
        x_labels=labels,
        y_labels=labels,
        id="heatmap",
    )

    ax.set_xyt(t="ax.stx_heatmap(values_2d)")

    return fig, ax


# EOF
