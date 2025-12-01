#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:55:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_seaborn_wrappers/_06_sns_kdeplot.py

"""ax.sns_kdeplot() - Seaborn KDE plot with normality test."""

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


def demo_sns_kdeplot(fig, ax, stx):
    """ax.sns_kdeplot() - Seaborn KDE plot with normality test."""
    np.random.seed(42)
    df = pd.DataFrame({
        "value": np.concatenate([
            np.random.normal(0, 1, 300),
            np.random.exponential(2, 200),
        ]),
        "category": np.repeat(["Normal", "Exponential"], [300, 200]),
    })

    ax.sns_kdeplot(x="value", hue="category", data=df)

    # Statistical test: Shapiro-Wilk normality test on Normal group
    normal_data = df[df["category"] == "Normal"]["value"].values[:50]  # subset for Shapiro
    w_stat, p_val = sp_stats.shapiro(normal_data)

    stats_text = f"Shapiro: W = {w_stat:.3f}, {format_pvalue(p_val)}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, ha="right", va="top", fontsize=5)

    ax.set_xyt( x="Value [a.u.]", y="Density", t="ax.sns_kdeplot(x, hue, data)")
    ax.legend(frameon=False, title="", fontsize=6)

    return fig, ax


# EOF
