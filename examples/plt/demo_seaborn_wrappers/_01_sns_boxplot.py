#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:55:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_seaborn_wrappers/_01_sns_boxplot.py

"""ax.sns_boxplot() - Seaborn box plot with statistical test."""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def format_pvalue(p):
    """Format p-value with significance stars."""
    if p < 0.001:
        return f"p < 0.001 ***"
    elif p < 0.01:
        return f"p = {p:.3f} **"
    elif p < 0.05:
        return f"p = {p:.3f} *"
    else:
        return f"p = {p:.3f} (ns)"


def demo_sns_boxplot(fig, ax, stx):
    """ax.sns_boxplot() - Seaborn box plot with statistical test."""
    np.random.seed(42)
    df = pd.DataFrame({
        "category": np.repeat(["Control", "Treatment A", "Treatment B"], 50),
        "value": np.concatenate([
            np.random.normal(0, 1, 50),
            np.random.normal(1, 1, 50),
            np.random.normal(2, 1.5, 50),
        ]),
    })

    ax.sns_boxplot(x="category", y="value", data=df, id="sns_box")

    # Statistical test: One-way ANOVA
    groups = [df[df["category"] == cat]["value"].values for cat in df["category"].unique()]
    f_stat, p_val = sp_stats.f_oneway(*groups)

    # Display stats on plot
    stats_text = f"ANOVA: F = {f_stat:.2f}, {format_pvalue(p_val)}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, ha="right", va="top", fontsize=5)

    ax.set_xyt( x="Group", y="Value [a.u.]", t="ax.sns_boxplot(x, y, data)")

    return fig, ax


# EOF
