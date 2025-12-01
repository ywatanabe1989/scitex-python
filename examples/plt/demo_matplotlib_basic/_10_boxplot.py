#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:45:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_matplotlib_basic/_10_boxplot.py

"""ax.boxplot() - Box plot."""

import numpy as np


def demo_boxplot(fig, ax, stx):
    """ax.boxplot() - Box plot."""
    np.random.seed(42)
    data = [np.random.normal(i, 1, 100) for i in range(4)]

    ax.boxplot(data, labels=["A", "B", "C", "D"], id="boxplot")

    ax.set_xyt(x="Group", y="Value [a.u.]", t="ax.boxplot(data)")

    return fig, ax


# EOF
