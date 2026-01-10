# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_mpl_boxplot.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: plot_mpl_boxplot.py - mpl_boxplot demo
# 
# """mpl_boxplot: box plot."""
# 
# import numpy as np
# 
# 
# def plot_mpl_boxplot(plt, rng, ax=None):
#     """mpl_boxplot - box plot.
# 
#     Demonstrates: ax.mpl_boxplot() - identical to ax.boxplot()
#     """
#     if ax is None:
#         fig, ax = plt.subplots()
#     else:
#         fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex
# 
#     data = [rng.normal(i, 1, 100) for i in range(4)]
#     ax.mpl_boxplot(data, tick_labels=['A', 'B', 'C', 'D'])
#     ax.set_xyt("X", "Y", "mpl_boxplot")
#     if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
#         ax.legend()
#     return fig, ax
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_mpl_boxplot.py
# --------------------------------------------------------------------------------
