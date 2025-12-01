#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 09:55:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_seaborn_wrappers/_10_sns_heatmap.py

"""ax.sns_heatmap() - Seaborn heatmap."""

import numpy as np


def demo_sns_heatmap(fig, ax, stx):
    """ax.sns_heatmap() - Seaborn heatmap."""
    np.random.seed(42)
    # Create correlation matrix
    data = np.random.rand(5, 100)
    corr_matrix = np.corrcoef(data)

    ax.sns_heatmap(corr_matrix, annot=True, id="sns_heatmap")

    ax.set_xyt( t="ax.sns_heatmap(data, annot)")

    return fig, ax


# EOF
