#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers/_05_plot_median_iqr.py

"""ax.plot_median_iqr(data, xx) - Median with interquartile range."""

import numpy as np


def demo_plot_median_iqr(fig, ax, stx):
    """ax.plot_median_iqr(data, xx) - Median with interquartile range."""
    np.random.seed(42)
    x = np.linspace(0, 10, 20)
    y_data = np.sin(x)[np.newaxis, :] + np.random.normal(0, 0.5, (20, len(x)))

    ax.plot_median_iqr(y_data, xx=x, label="Median ± IQR", id="median_iqr")

    ax.set_xyt(x="Time [s]", y="Signal [a.u.]", t="ax.plot_median_iqr(data, xx)")
    ax.legend(frameon=False, fontsize=6)

    return fig, ax


# EOF
