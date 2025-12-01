#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:45:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_matplotlib_basic/_05_bar.py

"""ax.bar() - Vertical bar plot."""


def demo_bar(fig, ax, stx):
    """ax.bar() - Vertical bar plot."""
    categories = ["A", "B", "C", "D"]
    values = [23, 45, 31, 52]
    errors = [3, 5, 4, 6]

    ax.bar(categories, values, yerr=errors, id="bars")

    ax.set_xyt(x="Category", y="Value [a.u.]", t="ax.bar(x, height)")

    return fig, ax


# EOF
