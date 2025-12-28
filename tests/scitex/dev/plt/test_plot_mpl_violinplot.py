# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_mpl_violinplot.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: plot_mpl_violinplot.py - mpl_violinplot demo
# 
# """mpl_violinplot: violin plot."""
# 
# import numpy as np
# 
# 
# def plot_mpl_violinplot(plt, rng, ax=None):
#     """mpl_violinplot - violin plot.
# 
#     Demonstrates: ax.mpl_violinplot() - identical to ax.violinplot()
#     """
#     if ax is None:
#         fig, ax = plt.subplots()
#     else:
#         fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex
# 
#     data = [rng.normal(i, 1, 100) for i in range(4)]
#     ax.mpl_violinplot(data)
#     ax.set_xyt("X", "Y", "mpl_violinplot")
#     if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
#         ax.legend()
#     return fig, ax
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_mpl_violinplot.py
# --------------------------------------------------------------------------------
