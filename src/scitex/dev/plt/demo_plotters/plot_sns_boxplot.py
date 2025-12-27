#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_sns_boxplot.py - sns_boxplot demo

"""sns_boxplot: DataFrame grouping."""

import numpy as np
import pandas as pd


def plot_sns_boxplot(plt, rng, ax=None):
    """sns_boxplot - DataFrame grouping.

    Demonstrates: ax.sns_boxplot(data=df, ...)
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex

    n = 100
    groups = rng.choice(['A', 'B', 'C'], n)
    df = pd.DataFrame({'group': groups, 'value': rng.normal(0, 1, n) + np.where(groups == 'A', 0, np.where(groups == 'B', 1, 2))})
    ax.sns_boxplot(data=df, x='group', y='value')
    ax.set_xyt("X", "Y", "sns_boxplot")
    return fig, ax


# EOF
