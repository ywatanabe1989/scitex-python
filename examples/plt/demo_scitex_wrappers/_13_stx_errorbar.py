#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers/_13_stx_errorbar.py

"""ax.stx_errorbar(x, y, yerr) - Error bar wrapper."""

import numpy as np


def demo_stx_errorbar(fig, ax, stx):
    """ax.stx_errorbar(x, y, yerr) - Error bar wrapper."""
    x = np.arange(1, 6)
    y = np.array([2.3, 3.5, 4.1, 3.8, 2.9])
    yerr = np.array([0.3, 0.4, 0.3, 0.5, 0.4])

    ax.stx_errorbar(x, y, yerr=yerr, id="errorbar", label="Measurement")

    ax.set_xyt(x="Time [min]", y="Conc. [µM]", t="ax.stx_errorbar(x, y, yerr)")
    ax.legend()

    return fig, ax


# EOF
