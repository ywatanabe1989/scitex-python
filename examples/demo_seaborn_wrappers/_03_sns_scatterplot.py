#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:55:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_seaborn_wrappers/_03_sns_scatterplot.py

"""ax.sns_scatterplot() - Seaborn scatter plot with correlation."""

import numpy as np
import pandas as pd
from scitex.stats.tests.correlation import test_pearson


def demo_sns_scatterplot(fig, ax, stx):
    """ax.sns_scatterplot() - Seaborn scatter plot with correlation."""
    np.random.seed(42)
    n = 100
    x = np.random.normal(0, 1, n)
    y = 0.7 * x + np.random.normal(0, 0.5, n)
    category = np.random.choice(["Group A", "Group B"], n)

    df = pd.DataFrame({"x": x, "y": y, "category": category})

    ax.sns_scatterplot(x="x", y="y", hue="category", data=df, id="sns_scatter")

    # Statistical test: Pearson correlation (using scitex.stats)
    stats_result = test_pearson(x, y)

    ax.text(0.05, 0.95, stats_result.format_text("compact"),
            transform=ax.transAxes, ha="left", va="top", fontsize=5)

    ax.set_xyt( x="X [a.u.]", y="Y [a.u.]", t="Scatter with Pearson Correlation")
    ax.legend(frameon=False, title="", fontsize=6)

    return fig, ax


# EOF
