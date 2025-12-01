#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers/_15_stx_fillv.py

"""ax.stx_fillv(starts_1d, ends_1d) - Vertical fill regions."""

import numpy as np


def demo_stx_fillv(fig, ax, stx):
    """ax.stx_fillv(starts_1d, ends_1d) - Vertical fill regions."""
    # Highlight regions
    starts_1d = [1, 4, 7]
    ends_1d = [2, 5, 8]
    ax.stx_fillv(starts_1d, ends_1d, id="regions")

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.set_xyt(x="Time [s]", y="Signal [a.u.]", t="ax.stx_fillv(starts_1d, ends_1d)")

    return fig, ax


# EOF
