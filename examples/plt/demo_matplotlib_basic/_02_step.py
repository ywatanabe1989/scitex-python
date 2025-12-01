#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:45:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_matplotlib_basic/_02_step.py

"""ax.step() - Step plot."""

import numpy as np


def demo_step(fig, ax, stx):
    """ax.step() - Step plot."""
    x = np.arange(10)
    y = np.random.rand(10)
    ax.step(x, y, where="mid", label="Step", id="step")

    ax.set_xyt(x="Index", y="Value [a.u.]", t="ax.step(x, y)")
    ax.legend(frameon=False, fontsize=6)

    return fig, ax


# EOF
