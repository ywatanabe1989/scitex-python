#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 20:06:03 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/plt/demo_matplotlib_basic/_01_plot.py


"""ax.plot() - Basic line plot."""

import numpy as np


def demo_plot(fig, ax, stx):
    """ax.plot() - Basic line plot."""
    x = np.linspace(0, 2 * np.pi, 100)
    ax.plot(x, np.sin(x), "b-", label="sin(x)", id="sine")
    ax.plot(x, np.cos(x), "r--", label="cos(x)", id="cosine")

    ax.set_xyt(x="Time [s]", y="Amplitude [a.u.]", t="ax.plot(x, y)")
    ax.legend(frameon=False, fontsize=6)

    return fig, ax

# EOF
