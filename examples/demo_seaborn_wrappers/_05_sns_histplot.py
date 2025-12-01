#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:55:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_seaborn_wrappers/_05_sns_histplot.py

"""ax.sns_histplot() - Seaborn histogram with KDE."""

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


def demo_sns_histplot(fig, ax, stx):
    """ax.sns_histplot() - Seaborn histogram with KDE."""
    np.random.seed(42)
    df = pd.DataFrame({
        "value": np.concatenate([
            np.random.normal(0, 1, 300),
            np.random.normal(3, 1, 200),
        ]),
        "category": np.repeat(["Control", "Treatment"], [300, 200]),
    })

    ax.sns_histplot(x="value", hue="category", data=df, kde=True, id="sns_hist")

    # Statistical test: Two-sample t-test
    ctrl = df[df["category"] == "Control"]["value"].values
    treat = df[df["category"] == "Treatment"]["value"].values
    t_stat, p_val = sp_stats.ttest_ind(ctrl, treat)

    stats_text = f"t-test: t = {t_stat:.2f}, {format_pvalue(p_val)}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, ha="right", va="top", fontsize=5)

    ax.set_xyt( x="Value [a.u.]", y="Count", t="Histogram with t-test")
    ax.legend(frameon=False, title="", fontsize=6)

    return fig, ax


# EOF
