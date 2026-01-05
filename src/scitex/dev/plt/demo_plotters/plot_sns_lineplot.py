#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_sns_lineplot.py - sns_lineplot demo

"""sns_lineplot: DataFrame time series."""

import numpy as np
import pandas as pd


def plot_sns_lineplot(plt, rng, ax=None):
    """sns_lineplot - DataFrame time series.

    Demonstrates: ax.sns_lineplot(data=df, ...)
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex

    n = 100
    times = np.tile(np.arange(20), 30)
    groups = np.repeat(['A', 'B'], 300)
    values = np.where(groups == 'A', np.sin(times * 0.3), np.cos(times * 0.3)) + rng.normal(0, 0.3, 600)
    df = pd.DataFrame({'time': times, 'value': values, 'group': groups})
    ax.sns_lineplot(data=df, x='time', y='value', hue='group')
    ax.set_xyt("X", "Y", "sns_lineplot")
    return fig, ax


# EOF
