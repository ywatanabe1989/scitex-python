#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:55:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_seaborn_wrappers/_02_sns_violinplot.py

"""ax.sns_violinplot() - Seaborn violin plot with statistical test."""

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


def demo_sns_violinplot(fig, ax, stx):
    """ax.sns_violinplot() - Seaborn violin plot with statistical test."""
    np.random.seed(42)
    df = pd.DataFrame({
        "category": np.repeat(["A", "B", "C"], 50),
        "value": np.concatenate([
            np.random.normal(0, 1, 50),
            np.random.normal(0.5, 1, 50),
            np.random.normal(1, 1.5, 50),
        ]),
    })

    ax.sns_violinplot(x="category", y="value", data=df, id="sns_violin")

    # Statistical test: Kruskal-Wallis (non-parametric)
    groups = [df[df["category"] == cat]["value"].values for cat in df["category"].unique()]
    h_stat, p_val = sp_stats.kruskal(*groups)

    stats_text = f"Kruskal: H = {h_stat:.2f}, {format_pvalue(p_val)}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, ha="right", va="top", fontsize=5)

    ax.set_xyt( x="Category", y="Value [a.u.]", t="Violin Plot with Kruskal-Wallis")

    return fig, ax


# EOF
