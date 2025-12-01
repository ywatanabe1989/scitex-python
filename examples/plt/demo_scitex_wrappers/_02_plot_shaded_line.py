#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers/_02_plot_shaded_line.py

"""ax.stx_shaded_line(x, y_lo, y_mid, y_hi) - Line with shaded uncertainty."""

import numpy as np


def demo_plot_shaded_line(fig, ax, stx):
    """ax.stx_shaded_line(x, y_lo, y_mid, y_hi) - Line with shaded uncertainty."""
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y_middle = np.sin(x)
    y_lower = y_middle - 0.3
    y_upper = y_middle + 0.3

    ax.stx_shaded_line(x, y_lower, y_middle, y_upper, id="shaded", label="Signal")

    ax.set_xyt(x="Time [s]", y="Signal [a.u.]", t="ax.stx_shaded_line(x, lo, mid, hi)")
    ax.legend()

    return fig, ax


# EOF
