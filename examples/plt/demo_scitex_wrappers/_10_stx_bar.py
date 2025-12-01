#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers/_10_stx_bar.py

"""ax.stx_bar(categories, values) - Bar plot wrapper."""


def demo_stx_bar(fig, ax, stx):
    """ax.stx_bar(categories, values) - Bar plot wrapper."""
    categories = ["A", "B", "C", "D"]
    values = [23, 45, 31, 52]

    ax.stx_bar(categories, values, id="bar")

    ax.set_xyt(x="Category", y="Value [a.u.]", t="ax.stx_bar(x, y)")

    return fig, ax


# EOF
