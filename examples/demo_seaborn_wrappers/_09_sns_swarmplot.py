#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:55:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_seaborn_wrappers/_09_sns_swarmplot.py

"""ax.sns_swarmplot() - Seaborn swarm plot."""

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


def demo_sns_swarmplot(fig, ax, stx):
    """ax.sns_swarmplot() - Seaborn swarm plot."""
    np.random.seed(42)
    df = pd.DataFrame({
        "category": np.repeat(["Control", "Treatment"], 30),
        "value": np.concatenate([
            np.random.normal(5, 1, 30),
            np.random.normal(7, 1.2, 30),
        ]),
    })

    ax.sns_swarmplot(x="category", y="value", data=df, id="sns_swarm")

    # Two-sample t-test
    ctrl = df[df["category"] == "Control"]["value"].values
    treat = df[df["category"] == "Treatment"]["value"].values
    t_stat, p_val = sp_stats.ttest_ind(ctrl, treat)

    stats_text = f"t-test: t = {t_stat:.2f}, {format_pvalue(p_val)}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, ha="right", va="top", fontsize=5)

    ax.set_xyt( x="Group", y="Value [a.u.]", t="Swarm Plot with t-test")

    return fig, ax


# EOF
