#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers/_24_plot_joyplot.py

"""ax.plot_joyplot(arrays) - Joy/ridge plot."""

import numpy as np


def demo_plot_joyplot(fig, ax, stx):
    """ax.plot_joyplot(arrays) - Joy/ridge plot."""
    np.random.seed(42)
    arrays = [np.random.normal(i * 0.5, 1, 200) for i in range(5)]

    ax.plot_joyplot(arrays, id="joyplot")

    ax.set_xyt(x="Value [a.u.]", y="Density", t="ax.plot_joyplot(arrays)")

    return fig, ax


# EOF
