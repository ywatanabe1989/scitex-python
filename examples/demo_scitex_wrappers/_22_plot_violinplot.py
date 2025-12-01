#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers/_22_plot_violinplot.py

"""ax.plot_violinplot(data) - Violinplot wrapper."""

import numpy as np


def demo_plot_violinplot(fig, ax, stx):
    """ax.plot_violinplot(data) - Violinplot wrapper."""
    np.random.seed(42)
    data = [np.random.normal(i, 1, 100) for i in range(4)]

    ax.plot_violinplot(data, positions=[1, 2, 3, 4], showextrema=False, id="violinplot")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["A", "B", "C", "D"])

    ax.set_xyt(x="Group", y="Value [a.u.]", t="ax.plot_violinplot(data)")

    return fig, ax


# EOF
