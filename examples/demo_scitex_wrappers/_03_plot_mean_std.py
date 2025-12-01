#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers/_03_plot_mean_std.py

"""ax.plot_mean_std(values_2d) - Mean with standard deviation shading."""

import numpy as np


def demo_plot_mean_std(fig, ax, stx):
    """ax.plot_mean_std(values_2d) - Mean with standard deviation shading."""
    np.random.seed(42)
    xx = np.linspace(0, 10, 20)
    # Create 2D data: (n_samples, n_timepoints)
    values_2d = np.sin(xx)[np.newaxis, :] + np.random.normal(0, 0.3, (10, len(xx)))

    ax.plot_mean_std(values_2d, xx=xx, id="mean_std", label="Signal")

    ax.set_xyt(x="Time [s]", y="Signal [a.u.]", t="ax.plot_mean_std(values_2d)")
    ax.legend()

    return fig, ax


# EOF
