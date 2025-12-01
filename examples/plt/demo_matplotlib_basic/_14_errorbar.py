#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:45:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_matplotlib_basic/_14_errorbar.py

"""ax.errorbar() - Error bar plot."""

import numpy as np


def demo_errorbar(fig, ax, stx):
    """ax.errorbar() - Error bar plot."""
    x = np.arange(1, 6)
    y = np.array([2.3, 3.5, 4.1, 3.8, 2.9])
    yerr = np.array([0.3, 0.4, 0.3, 0.5, 0.4])

    ax.errorbar(x, y, yerr=yerr, label="Data ± SE", id="errorbar")

    ax.set_xyt(x="Time [min]", y="Concentration [µM]", t="ax.errorbar(x, y, yerr)")
    ax.legend(frameon=False, fontsize=6)

    return fig, ax


# EOF
