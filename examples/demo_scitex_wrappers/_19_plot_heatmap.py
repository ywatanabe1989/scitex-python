#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers/_19_plot_heatmap.py

"""ax.plot_heatmap(data, x_labels, y_labels) - Heatmap with annotations."""

import numpy as np


def demo_plot_heatmap(fig, ax, stx):
    """ax.plot_heatmap(data, x_labels, y_labels) - Heatmap with annotations."""
    np.random.seed(42)
    data = np.random.rand(5, 5)  # Square matrix for cleaner display
    labels = ["A", "B", "C", "D", "E"]

    ax.plot_heatmap(
        data,
        x_labels=labels,
        y_labels=labels,
        cbar_label="Values",
        show_annot=True,
        annot_format="{x:.2f}",
        cmap="viridis",
        id="heatmap",
    )

    ax.set_xyt(t="ax.plot_heatmap(data, x_labels, y_labels)")

    return fig, ax


# EOF
