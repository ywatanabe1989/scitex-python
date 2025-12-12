#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_raster.py - Raster plot (spike trains)

"""
Raster plot for visualizing spike trains or event data.
"""

import numpy as np
import scitex as stx


def plot_raster(plt, rng, ax=None):
    """Raster plot with multiple spike trains and stimulus region.

    Parameters
    ----------
    plt : module
        Plotting module
    rng : numpy.random.Generator
        Random number generator
    ax : Axes, optional
        Axes to plot on. If None, creates new figure.

    Returns
    -------
    fig : Figure
        The figure object
    ax : Axes
        The axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure() if hasattr(ax, 'get_figure') else ax._fig_scitex

    n_trials = 30

    # Generate spike trains with different firing patterns
    # Pre-stimulus: baseline firing
    # During stimulus: elevated firing
    # Post-stimulus: return to baseline
    spike_trains = []
    for trial in range(n_trials):
        # Pre-stimulus (0-0.5s): baseline
        pre = rng.uniform(0, 0.5, rng.poisson(5))
        # Stimulus (0.5-1.5s): elevated firing
        stim = rng.uniform(0.5, 1.5, rng.poisson(15))
        # Post-stimulus (1.5-2.0s): return to baseline
        post = rng.uniform(1.5, 2.0, rng.poisson(5))
        spike_trains.append(np.sort(np.concatenate([pre, stim, post])))

    # Plot raster
    ax.stx_raster(spike_trains, id="spikes")

    # Add stimulus period indicator
    ax.stx_fillv([0.5], [1.5], alpha=0.2, color="yellow", id="stim-period")

    # Add stimulus onset/offset lines
    ax.axvline(0.5, color="green", linestyle="--", linewidth=1.5, label="Stim ON")
    ax.axvline(1.5, color="red", linestyle="--", linewidth=1.5, label="Stim OFF")

    ax.set_xyt("Time [s]", "Trial", "Raster Plot with Stimulus Period")
    ax.legend(loc="upper right")

    return fig, ax


