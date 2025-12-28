# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_mpl_scatter.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: plot_mpl_scatter.py - mpl_scatter demo
# 
# """mpl_scatter: scatter plot."""
# 
# import numpy as np
# 
# 
# def plot_mpl_scatter(plt, rng, ax=None):
#     """mpl_scatter - scatter plot.
# 
#     Demonstrates: ax.mpl_scatter() - identical to ax.scatter()
#     """
#     if ax is None:
#         fig, ax = plt.subplots()
#     else:
#         fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex
# 
#     x, y = rng.uniform(0, 10, 50), rng.uniform(0, 10, 50)
#     ax.mpl_scatter(x, y, c=rng.uniform(0, 1, 50), cmap='viridis')
#     ax.set_xyt("X", "Y", "mpl_scatter")
#     if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
#         ax.legend()
#     return fig, ax
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_mpl_scatter.py
# --------------------------------------------------------------------------------
