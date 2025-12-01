#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers/_21_plot_boxplot.py

"""ax.plot_boxplot(data, labels) - Boxplot wrapper."""

import numpy as np


def demo_plot_boxplot(fig, ax, stx):
    """ax.plot_boxplot(data, labels) - Boxplot wrapper."""
    np.random.seed(42)
    data = [np.random.normal(i, 1, 100) for i in range(4)]

    ax.plot_boxplot(data, labels=["A", "B", "C", "D"], id="boxplot", label="Data")

    ax.set_xyt(x="Group", y="Value [a.u.]", t="ax.boxplot(data)")
    ax.legend()

    return fig, ax


# EOF
