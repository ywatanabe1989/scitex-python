#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:45:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_matplotlib_basic/_06_barh.py

"""ax.barh() - Horizontal bar plot."""


def demo_barh(fig, ax, stx):
    """ax.barh() - Horizontal bar plot."""
    methods = ["Method A", "Method B", "Method C", "Method D"]
    scores = [0.85, 0.92, 0.78, 0.88]
    errors = [0.03, 0.04, 0.05, 0.02]

    ax.barh(methods, scores, xerr=errors, id="barh")

    ax.set_xyt(x="Score", y="Method", t="ax.barh(y, width)")

    return fig, ax


# EOF
