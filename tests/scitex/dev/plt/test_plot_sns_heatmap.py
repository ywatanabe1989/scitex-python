# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_sns_heatmap.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: plot_sns_heatmap.py - sns_heatmap demo
# 
# """sns_heatmap: DataFrame heatmap."""
# 
# import numpy as np
# import pandas as pd
# 
# 
# def plot_sns_heatmap(plt, rng, ax=None):
#     """sns_heatmap - DataFrame heatmap.
# 
#     Demonstrates: ax.sns_heatmap(data=df, ...)
#     """
#     if ax is None:
#         fig, ax = plt.subplots()
#     else:
#         fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex
# 
#     n = 100
#     data = pd.DataFrame(rng.uniform(0, 1, (5, 5)), index=['A', 'B', 'C', 'D', 'E'], columns=['V1', 'V2', 'V3', 'V4', 'V5'])
#     ax.sns_heatmap(data, annot=True, fmt='.2f')
#     ax.set_xyt("X", "Y", "sns_heatmap")
#     return fig, ax
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_sns_heatmap.py
# --------------------------------------------------------------------------------
