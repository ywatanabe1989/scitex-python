#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers/_25_stx_rectangle.py

"""ax.stx_rectangle(x, y, w, h) - Rectangle annotation."""

import numpy as np


def demo_stx_rectangle(fig, ax, stx):
    """ax.stx_rectangle(x, y, w, h) - Rectangle annotation."""
    # Add rectangles (no edge by default for publication figures)
    ax.stx_rectangle(1, 0, 3, 2, id="rect1")
    ax.stx_rectangle(5, 1, 2, 3, id="rect2")
    ax.stx_rectangle(3, 2.5, 4, 1.5, id="rect3")

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.set_xyt(x="X", y="Y", t="ax.stx_rectangle(x, y, w, h)")

    return fig, ax


# EOF
