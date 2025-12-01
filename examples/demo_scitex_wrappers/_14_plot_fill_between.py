#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers/_14_plot_fill_between.py

"""ax.plot_fill_between(x, y1, y2) - Fill between wrapper."""

import numpy as np


def demo_plot_fill_between(fig, ax, stx):
    """ax.plot_fill_between(x, y1, y2) - Fill between wrapper."""
    x = np.linspace(0, 2 * np.pi, 100)
    y1 = np.sin(x) - 0.2
    y2 = np.sin(x) + 0.2

    ax.plot_fill_between(x, y1, y2, id="fill")
    ax.plot(x, (y1 + y2) / 2, id="center")

    ax.set_xyt(x="Time [s]", y="Signal [a.u.]", t="ax.plot_fill_between(x, y1, y2)")

    return fig, ax


# EOF
