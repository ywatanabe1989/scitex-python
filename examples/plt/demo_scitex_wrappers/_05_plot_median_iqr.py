#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers/_05_plot_median_iqr.py

"""ax.stx_median_iqr(values_2d) - Median with interquartile range."""

import numpy as np


def demo_plot_median_iqr(fig, ax, stx):
    """ax.stx_median_iqr(values_2d) - Median with interquartile range."""
    np.random.seed(42)
    xx = np.linspace(0, 10, 20)
    values_2d = np.sin(xx)[np.newaxis, :] + np.random.normal(0, 0.5, (20, len(xx)))

    ax.stx_median_iqr(values_2d, xx=xx, id="median_iqr", label="Signal")

    ax.set_xyt(x="Time [s]", y="Signal [a.u.]", t="ax.stx_median_iqr(values_2d)")
    ax.legend()

    return fig, ax


# EOF
