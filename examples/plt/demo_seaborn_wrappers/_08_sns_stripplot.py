#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:55:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_seaborn_wrappers/_08_sns_stripplot.py

"""ax.sns_stripplot() - Seaborn strip plot."""

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


def demo_sns_stripplot(fig, ax, stx):
    """ax.sns_stripplot() - Seaborn strip plot."""
    np.random.seed(42)
    df = pd.DataFrame({
        "category": np.repeat(["A", "B", "C"], 40),
        "value": np.concatenate([
            np.random.normal(0, 1, 40),
            np.random.normal(1, 1, 40),
            np.random.normal(0.5, 1.5, 40),
        ]),
    })

    ax.sns_stripplot(x="category", y="value", data=df, id="sns_strip")

    # Mann-Whitney U test between A and B
    a_vals = df[df["category"] == "A"]["value"].values
    b_vals = df[df["category"] == "B"]["value"].values
    u_stat, p_val = sp_stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")

    stats_text = f"Mann-Whitney: U = {u_stat:.1f}, {format_pvalue(p_val)}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, ha="right", va="top", fontsize=5)

    ax.set_xyt( x="Category", y="Value [a.u.]", t="ax.sns_stripplot(x, y, data)")

    return fig, ax


# EOF
