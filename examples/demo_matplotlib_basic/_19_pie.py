#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:45:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_matplotlib_basic/_19_pie.py

"""ax.pie() - Pie chart."""


def demo_pie(fig, ax, stx):
    """ax.pie() - Pie chart."""
    sizes = [30, 25, 20, 15, 10]
    labels = ["A", "B", "C", "D", "E"]

    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, id="pie")

    ax.set_xyt(t="ax.pie(sizes)")

    return fig, ax


# EOF
