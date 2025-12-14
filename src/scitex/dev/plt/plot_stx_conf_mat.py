#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_stx_conf_mat.py - stx_conf_mat demo

"""stx_conf_mat: confusion matrix."""

import numpy as np


def plot_stx_conf_mat(plt, rng, ax=None):
    """stx_conf_mat - confusion matrix.

    Demonstrates: ax.stx_conf_mat()
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex

    data = rng.integers(0, 100, (4, 4))
    ax.stx_conf_mat(data, x_labels=['A', 'B', 'C', 'D'], y_labels=['A', 'B', 'C', 'D'])
    ax.set_xyt("X", "Y", "stx_conf_mat")
    if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
        ax.legend()
    return fig, ax


# EOF
