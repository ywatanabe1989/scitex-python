#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers/_20_plot_conf_mat.py

"""ax.plot_conf_mat(data, x_labels, y_labels) - Confusion matrix."""

import numpy as np


def demo_plot_conf_mat(fig, ax, stx):
    """ax.plot_conf_mat(data, x_labels, y_labels) - Confusion matrix."""
    # Create confusion matrix data
    data = np.array([
        [50, 5, 3],
        [8, 45, 7],
        [2, 6, 42],
    ])
    labels = ["A", "B", "C"]

    ax.plot_conf_mat(
        data,
        x_labels=labels,
        y_labels=labels,
        id="conf_mat",
    )

    ax.set_xyt(t="ax.plot_conf_mat(data, x_labels, y_labels)")

    return fig, ax


# EOF
