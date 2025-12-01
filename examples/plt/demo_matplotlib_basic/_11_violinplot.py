#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:45:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_matplotlib_basic/_11_violinplot.py

"""ax.violinplot() - Violin plot with boxplot overlay."""

import numpy as np


def demo_violinplot(fig, ax, stx):
    """ax.violinplot() - Violin plot with boxplot overlay (default)."""
    np.random.seed(42)
    data = [np.random.normal(i, 1, 100) for i in range(4)]
    positions = [1, 2, 3, 4]

    # Violin plot with automatic boxplot overlay (boxplot=True by default)
    ax.violinplot(data, positions=positions, id="violinplot")

    ax.set_xticks(positions)
    ax.set_xticklabels(["A", "B", "C", "D"])
    ax.set_xyt(x="Group", y="Value [a.u.]", t="ax.violinplot(data)")

    return fig, ax


# EOF
