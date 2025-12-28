# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_sns_violinplot.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: plot_sns_violinplot.py - sns_violinplot demo
# 
# """sns_violinplot: DataFrame grouping."""
# 
# import numpy as np
# import pandas as pd
# 
# 
# def plot_sns_violinplot(plt, rng, ax=None):
#     """sns_violinplot - DataFrame grouping.
# 
#     Demonstrates: ax.sns_violinplot(data=df, ...)
#     """
#     if ax is None:
#         fig, ax = plt.subplots()
#     else:
#         fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex
# 
#     n = 100
#     groups = rng.choice(['A', 'B', 'C'], n)
#     df = pd.DataFrame({'group': groups, 'value': rng.normal(0, 1, n) + np.where(groups == 'A', 0, np.where(groups == 'B', 1, 2))})
#     ax.sns_violinplot(data=df, x='group', y='value')
#     ax.set_xyt("X", "Y", "sns_violinplot")
#     return fig, ax
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_sns_violinplot.py
# --------------------------------------------------------------------------------
