#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-01 15:23:01 (ywatanabe)"
# File: /home/ywatanabe/proj/_scitex_repo/src/scitex/plt/ax/_plot/_plot_raster.py
# ----------------------------------------
import os

__FILE__ = "./src/scitex/plt/ax/_plot/_plot_raster.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import matplotlib
from bisect import bisect_left

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ....plt.utils import assert_valid_axis


def stx_raster(
    ax,
    spike_times_list,
    time=None,
    labels=None,
    colors=None,
    orientation="horizontal",
    y_offset=None,
    lineoffsets=None,
    linelengths=None,
    apply_set_n_ticks=True,
    n_xticks=4,
    n_yticks=None,
    **kwargs,
):
    """
    Create a raster plot using eventplot with custom labels and colors.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or scitex.plt._subplots.AxisWrapper
        The axes on which to draw the raster plot.
    spike_times_list : list of array-like, shape (n_trials,) where each element is (n_spikes,)
        List of spike/event time arrays, one per trial/channel
    time : array-like, optional
        The time indices for the events (default: np.linspace(0, max(event_times))).
    labels : list, optional
        Labels for each channel/trial.
    colors : list, optional
        Colors for each channel/trial.
    orientation: str, optional
        Orientation of raster plot (default: horizontal).
    y_offset : float, optional
        Vertical spacing between trials/channels (default: 1.0).
    lineoffsets : array-like, optional
        Y-positions for each trial/channel (overrides automatic positioning).
    linelengths : float, optional
        Height of each spike mark (default: 0.8, slightly less than y_offset to prevent overlap).
    apply_set_n_ticks : bool, optional
        Whether to apply set_n_ticks for cleaner axis (default: True).
    n_xticks : int, optional
        Number of x-axis ticks (default: 4).
    n_yticks : int or None, optional
        Number of y-axis ticks (default: None, auto-determined).
    **kwargs : dict
        Additional keyword arguments for eventplot.

    Returns
    -------
    ax : matplotlib.axes.Axes or scitex.plt._subplots.AxisWrapper
        The axes with the raster plot.
    df : pandas.DataFrame
        DataFrame with time indices and channel events.
    """
    assert_valid_axis(
        ax, "First argument must be a matplotlib axis or scitex axis wrapper"
    )

    # Format spike_times_list data
    spike_times_list = _ensure_list(spike_times_list)

    # Add sample size (number of trials) to label if provided
    if kwargs.get("label"):
        n_trials = len(spike_times_list)
        kwargs["label"] = f"{kwargs['label']} ($n$={n_trials})"

    # Handle colors and labels
    colors = _handle_colors(colors, spike_times_list)

    # Handle lineoffsets for positioning between trials/channels
    if y_offset is None:
        y_offset = 1.0  # Default spacing
    if lineoffsets is None:
        lineoffsets = np.arange(len(spike_times_list)) * y_offset

    # Set linelengths to prevent overlap (80% of y_offset by default)
    if linelengths is None:
        linelengths = y_offset * 0.8

    # Ensure lineoffsets is iterable and matches spike_times_list length
    if np.isscalar(lineoffsets):
        lineoffsets = [lineoffsets]
    if len(lineoffsets) < len(spike_times_list):
        lineoffsets = list(lineoffsets) + list(
            range(len(lineoffsets), len(spike_times_list))
        )

    # Plotting as eventplot using spike_times_list with proper positioning
    for ii, (pos, color, offset) in enumerate(
        zip(spike_times_list, colors, lineoffsets)
    ):
        label = _define_label(labels, ii)
        ax.eventplot(
            pos,
            lineoffsets=offset,
            linelengths=linelengths,
            orientation=orientation,
            colors=color,
            label=label,
            **kwargs,
        )

    # Apply set_n_ticks for cleaner axes if requested
    if apply_set_n_ticks:
        from scitex.plt.ax._style._set_n_ticks import set_n_ticks

        # For categorical y-axis (trials/channels), use appropriate tick count
        if n_yticks is None:
            n_yticks = min(len(spike_times_list), 8)  # Max 8 ticks for readability

        # Only apply if we have reasonable numeric ranges
        try:
            x_range = ax.get_xlim()
            y_range = ax.get_ylim()

            # Apply x-ticks if we have a reasonable numeric range
            if x_range[1] - x_range[0] > 0:
                set_n_ticks(ax, n_xticks=n_xticks, n_yticks=None)

            # Apply y-ticks only if we don't have categorical labels
            if labels is None and y_range[1] - y_range[0] > 0:
                set_n_ticks(ax, n_xticks=None, n_yticks=n_yticks)

        except Exception:
            # Skip set_n_ticks if there are issues (e.g., categorical data)
            pass

    # Legend
    if labels is not None:
        ax.legend()

    # Return spike_times in a useful format
    spike_times_digital_df = _event_times_to_digital_df(
        spike_times_list, time, lineoffsets
    )

    return ax, spike_times_digital_df


def _ensure_list(event_times):
    return [[pos] if isinstance(pos, (int, float)) else pos for pos in event_times]


def _define_label(labels, ii):
    if (labels is not None) and (ii < len(labels)):
        return labels[ii]
    else:
        return None


def _handle_colors(colors, event_times_list):
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if len(colors) < len(event_times_list):
        colors = colors * (len(event_times_list) // len(colors) + 1)
    return colors


def _event_times_to_digital_df(event_times_list, time, lineoffsets=None):
    if time is None:
        time = np.linspace(0, np.max([np.max(pos) for pos in event_times_list]), 1000)

    digi = np.full((len(event_times_list), len(time)), np.nan, dtype=float)

    for i_ch, posis_ch in enumerate(event_times_list):
        for posi_ch in posis_ch:
            i_insert = bisect_left(time, posi_ch)
            if i_insert == len(time):
                i_insert -= 1
            # Use lineoffset position if available, otherwise use channel index
            if lineoffsets is not None and i_ch < len(lineoffsets):
                digi[i_ch, i_insert] = lineoffsets[i_ch]
            else:
                digi[i_ch, i_insert] = i_ch

    return pd.DataFrame(digi.T, index=time)


# EOF
