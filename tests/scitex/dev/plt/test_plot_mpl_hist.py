# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_mpl_hist.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: plot_mpl_hist.py - mpl_hist demo
# 
# """mpl_hist: histogram."""
# 
# import numpy as np
# 
# 
# def plot_mpl_hist(plt, rng, ax=None):
#     """mpl_hist - histogram.
# 
#     Demonstrates: ax.mpl_hist() - identical to ax.hist()
#     """
#     if ax is None:
#         fig, ax = plt.subplots()
#     else:
#         fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex
# 
#     data = rng.standard_normal(1000)
#     ax.mpl_hist(data, bins=30, edgecolor='white', alpha=0.8)
#     ax.set_xyt("X", "Y", "mpl_hist")
#     if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
#         ax.legend()
#     return fig, ax
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_mpl_hist.py
# --------------------------------------------------------------------------------
