#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers/_09_plot_violin.py

"""ax.plot_violin(values_list) - Enhanced violin plot."""

import numpy as np


def demo_plot_violin(fig, ax, stx):
    """ax.plot_violin(values_list) - Enhanced violin plot."""
    np.random.seed(42)
    values_list = [np.random.normal(i, 1, 100) for i in range(3)]
    labels = ["A", "B", "C"]

    ax.plot_violin(values_list, labels=labels, id="violin")

    ax.set_xyt(x="Group", y="Value [a.u.]", t="ax.plot_violin(values_list)")

    return fig, ax


# EOF
