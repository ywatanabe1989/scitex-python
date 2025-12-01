#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:45:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_matplotlib_basic/_03_stem.py

"""ax.stem() - Stem plot."""

import numpy as np


def demo_stem(fig, ax, stx):
    """ax.stem() - Stem plot."""
    x = np.linspace(0, 2 * np.pi, 20)
    y = np.sin(x)
    ax.stem(x, y, id="stem")
    ax.set_xyt(x="Phase [rad]", y="Amplitude [a.u.]", t="ax.stem(x, y)")

    return fig, ax


# EOF
