#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers/_23_stx_raster.py

"""ax.stx_raster(spike_times_list) - Raster/spike plot."""

import numpy as np


def demo_stx_raster(fig, ax, stx):
    """ax.stx_raster(spike_times_list) - Raster/spike plot."""
    np.random.seed(42)
    # Create spike train data (list of spike times for each trial)
    spike_times_list = [
        np.sort(np.random.uniform(0, 1, np.random.randint(5, 20)))
        for _ in range(10)
    ]

    ax.stx_raster(spike_times_list, id="raster")

    ax.set_xyt(x="Time [s]", y="Trial", t="ax.stx_raster(spike_times_list)")

    return fig, ax


# EOF
