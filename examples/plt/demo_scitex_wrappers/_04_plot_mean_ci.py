#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers/_04_plot_mean_ci.py

"""ax.stx_mean_ci(values_2d) - Mean with confidence interval."""

import numpy as np


def demo_plot_mean_ci(fig, ax, stx):
    """ax.stx_mean_ci(values_2d) - Mean with confidence interval."""
    np.random.seed(42)
    xx = np.linspace(0, 10, 20)
    values_2d = np.sin(xx)[np.newaxis, :] + np.random.normal(0, 0.3, (30, len(xx)))

    ax.stx_mean_ci(values_2d, xx=xx, id="mean_ci", label="Signal")

    ax.set_xyt(x="Time [s]", y="Signal [a.u.]", t="ax.stx_mean_ci(values_2d)")
    ax.legend()

    return fig, ax


# EOF
