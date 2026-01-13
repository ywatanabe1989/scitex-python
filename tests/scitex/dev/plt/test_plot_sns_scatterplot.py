# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_sns_scatterplot.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: plot_sns_scatterplot.py - sns_scatterplot demo
# 
# """sns_scatterplot: DataFrame with hue."""
# 
# import numpy as np
# import pandas as pd
# 
# 
# def plot_sns_scatterplot(plt, rng, ax=None):
#     """sns_scatterplot - DataFrame with hue.
# 
#     Demonstrates: ax.sns_scatterplot(data=df, ...)
#     """
#     if ax is None:
#         fig, ax = plt.subplots()
#     else:
#         fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex
# 
#     n = 100
#     groups = rng.choice(['A', 'B', 'C'], n)
#     df = pd.DataFrame({'group': groups, 'x': rng.uniform(0, 10, n), 'y': rng.uniform(0, 10, n)})
#     ax.sns_scatterplot(data=df, x='x', y='y', hue='group')
#     ax.set_xyt("X", "Y", "sns_scatterplot")
#     return fig, ax
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_sns_scatterplot.py
# --------------------------------------------------------------------------------
