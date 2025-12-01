#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers/_07_plot_ecdf.py

"""ax.plot_ecdf(data) - Empirical cumulative distribution function."""

import numpy as np


def demo_plot_ecdf(fig, ax, stx):
    """ax.plot_ecdf(data) - Empirical cumulative distribution function."""
    np.random.seed(42)
    data = np.random.normal(0, 1, 1000)

    ax.plot_ecdf(data, label="ECDF", id="ecdf")

    ax.set_xyt(x="Value [a.u.]", y="Cumulative Prob.", t="ax.plot_ecdf(data)")
    ax.legend(frameon=False, fontsize=6)

    return fig, ax


# EOF
