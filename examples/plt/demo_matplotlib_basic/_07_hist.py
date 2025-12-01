#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:45:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_matplotlib_basic/_07_hist.py

"""ax.hist() - Histogram."""

import numpy as np
from scipy import stats


def demo_hist(fig, ax, stx):
    """ax.hist() - Histogram."""
    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(0, 1, 500),
        np.random.normal(4, 1, 300),
    ])

    ax.hist(data, bins=40, density=True, label="Data", id="histogram")

    # Add KDE overlay
    kde = stats.gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 200)
    ax.plot(x_range, kde(x_range), label="KDE", id="kde")

    ax.set_xyt(x="Value [a.u.]", y="Density", t="ax.hist(data)")
    ax.legend(frameon=False, fontsize=6)

    return fig, ax


# EOF
