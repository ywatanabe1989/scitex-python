#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_scitex_wrappers/_23_plot_raster.py

"""ax.plot_raster(spike_trains) - Raster/spike plot."""

import numpy as np


def demo_plot_raster(fig, ax, stx):
    """ax.plot_raster(spike_trains) - Raster/spike plot."""
    np.random.seed(42)
    # Create spike train data (list of spike times for each trial)
    spike_trains = [
        np.sort(np.random.uniform(0, 1, np.random.randint(5, 20)))
        for _ in range(10)
    ]

    ax.plot_raster(spike_trains, id="raster")

    ax.set_xyt(x="Time [s]", y="Trial", t="ax.plot_raster(spike_trains)")

    return fig, ax


# EOF
