#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers/_11_plot_barh.py

"""ax.plot_barh(categories, values) - Horizontal bar plot wrapper."""


def demo_plot_barh(fig, ax, stx):
    """ax.plot_barh(categories, values) - Horizontal bar plot wrapper."""
    methods = ["A", "B", "C"]
    scores = [0.85, 0.92, 0.78]

    ax.plot_barh(methods, scores, id="barh")

    ax.set_xyt(x="Score", y="Method", t="ax.plot_barh(x, y)")

    return fig, ax


# EOF
