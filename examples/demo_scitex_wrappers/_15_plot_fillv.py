#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers/_15_plot_fillv.py

"""ax.plot_fillv(starts, ends) - Vertical fill regions."""

import numpy as np


def demo_plot_fillv(fig, ax, stx):
    """ax.plot_fillv(starts, ends) - Vertical fill regions."""
    # Highlight regions
    ax.plot_fillv([1, 4, 7], [2, 5, 8], id="regions")

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.set_xyt(x="Time [s]", y="Signal [a.u.]", t="ax.plot_fillv(starts, ends)")

    return fig, ax


# EOF
