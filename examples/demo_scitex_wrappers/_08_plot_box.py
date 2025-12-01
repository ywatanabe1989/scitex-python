#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers/_08_plot_box.py

"""ax.plot_box(data, labels) - Enhanced box plot."""

import numpy as np


def demo_plot_box(fig, ax, stx):
    """ax.plot_box(data, labels) - Enhanced box plot."""
    np.random.seed(42)
    data = [np.random.normal(i, 1, 100) for i in range(4)]
    labels = ["A", "B", "C", "D"]

    ax.plot_box(data, labels=labels, id="box")

    ax.set_xyt(x="Group", y="Value [a.u.]", t="ax.plot_box(data, labels)")

    return fig, ax


# EOF
