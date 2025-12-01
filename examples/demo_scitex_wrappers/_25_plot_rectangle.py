#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers/_25_plot_rectangle.py

"""ax.plot_rectangle(x, y, w, h) - Rectangle annotation."""

import numpy as np


def demo_plot_rectangle(fig, ax, stx):
    """ax.plot_rectangle(x, y, w, h) - Rectangle annotation."""
    # Base plot
    x = np.linspace(0, 10, 100)
    ax.plot(x, np.sin(x), id="signal")

    # Add rectangles to highlight regions
    ax.plot_rectangle(2, -0.5, 2, 1, id="rect1")
    ax.plot_rectangle(6, -0.5, 2, 1, id="rect2")

    ax.set_xyt(x="X", y="Y", t="ax.plot_rectangle(x, y, w, h)")

    return fig, ax


# EOF
