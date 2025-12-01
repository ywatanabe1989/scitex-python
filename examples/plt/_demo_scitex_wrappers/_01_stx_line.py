#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers/_01_stx_line.py

"""ax.stx_line(values_1d) - Simple line plot wrapper."""

import numpy as np


def demo_stx_line(fig, ax, stx):
    """ax.stx_line(values_1d) - Simple line plot wrapper."""
    np.random.seed(42)
    values_1d = np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.normal(0, 0.1, 100)

    ax.stx_line(values_1d, id="line", label="Signal")

    ax.set_xyt(x="Sample", y="Amplitude [a.u.]", t="ax.stx_line(values_1d)")
    ax.legend()

    return fig, ax


# EOF
