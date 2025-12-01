#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:45:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_matplotlib_basic/_12_fill_between.py

"""ax.fill_between() - Fill between curves."""

import numpy as np


def demo_fill_between(fig, ax, stx):
    """ax.fill_between() - Fill between curves."""
    x = np.linspace(0, 2 * np.pi, 100)
    y_mean = np.sin(x)
    y_std = 0.2

    ax.fill_between(x, y_mean - y_std, y_mean + y_std, label="±1 SD", id="fill")
    ax.plot(x, y_mean, label="Mean", id="mean")

    ax.set_xyt(x="Time [s]", y="Signal [a.u.]", t="ax.fill_between(x, y1, y2)")
    ax.legend(frameon=False, fontsize=6)

    return fig, ax


# EOF
