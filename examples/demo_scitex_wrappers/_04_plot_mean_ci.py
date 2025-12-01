#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers/_04_plot_mean_ci.py

"""ax.plot_mean_ci(data, xx, perc) - Mean with confidence interval."""

import numpy as np


def demo_plot_mean_ci(fig, ax, stx):
    """ax.plot_mean_ci(data, xx, perc) - Mean with confidence interval."""
    np.random.seed(42)
    x = np.linspace(0, 10, 20)
    y_data = np.sin(x)[np.newaxis, :] + np.random.normal(0, 0.3, (30, len(x)))

    ax.plot_mean_ci(y_data, xx=x, id="mean_ci")

    ax.set_xyt(x="Time [s]", y="Signal [a.u.]", t="ax.plot_mean_ci(data, xx)")

    return fig, ax


# EOF
