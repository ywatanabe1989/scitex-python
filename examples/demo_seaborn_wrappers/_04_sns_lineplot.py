#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:55:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_seaborn_wrappers/_04_sns_lineplot.py

"""ax.sns_lineplot() - Seaborn line plot."""

import numpy as np
import pandas as pd


def demo_sns_lineplot(fig, ax, stx):
    """ax.sns_lineplot() - Seaborn line plot."""
    np.random.seed(42)
    x = np.tile(np.arange(20), 30)
    y = np.sin(x * 0.3) + np.random.normal(0, 0.3, len(x))
    group = np.repeat(["Condition A", "Condition B"], len(x) // 2)

    df = pd.DataFrame({"x": x, "y": y, "group": group})

    ax.sns_lineplot(x="x", y="y", hue="group", data=df, id="sns_line")

    ax.set_xyt( x="Time [s]", y="Signal [a.u.]", t="Line Plot with CI")
    ax.legend(frameon=False, title="", fontsize=6)

    return fig, ax


# EOF
