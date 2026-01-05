#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_sns_kdeplot.py - sns_kdeplot demo

"""sns_kdeplot: DataFrame KDE."""

import numpy as np
import pandas as pd


def plot_sns_kdeplot(plt, rng, ax=None):
    """sns_kdeplot - DataFrame KDE.

    Demonstrates: ax.sns_kdeplot(data=df, ...)
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex

    n = 100
    groups = rng.choice(['A', 'B'], n)
    df = pd.DataFrame({'group': groups, 'value': rng.normal(0, 1, n) + np.where(groups == 'A', 0, 2)})
    ax.sns_kdeplot(data=df, x='value', hue='group')
    ax.set_xyt("X", "Y", "sns_kdeplot")
    return fig, ax


# EOF
