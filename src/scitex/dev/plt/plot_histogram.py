#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: plot_histogram.py - Single histogram

import numpy as np
import scitex as stx


def plot_histogram(plt, rng):
    """Histogram - all bins should be grouped as one element."""
    fig, ax = plt.subplots(figsize=(8, 6))

    data = rng.standard_normal(2000)
    ax.hist(data, bins=40, color="#1f77b4", edgecolor="white", alpha=0.8)

    ax.set_xyt("Value", "Frequency", "Histogram (All Bins Grouped)")
    return fig, ax


