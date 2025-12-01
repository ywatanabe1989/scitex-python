#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers/_06_plot_kde.py

"""ax.plot_kde(values_1d) - Kernel density estimate."""

import numpy as np


def demo_plot_kde(fig, ax, stx):
    """ax.plot_kde(values_1d) - Kernel density estimate."""
    np.random.seed(42)
    values_1d = np.concatenate([
        np.random.normal(0, 1, 500),
        np.random.normal(5, 1, 300),
    ])

    ax.plot_kde(values_1d, id="kde", label="Mixture")

    ax.set_xyt(x="Value [a.u.]", y="Density", t="ax.plot_kde(values_1d)")
    ax.legend()

    return fig, ax


# EOF
