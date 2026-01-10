# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_mpl_pie.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: plot_mpl_pie.py - mpl_pie demo
# 
# """mpl_pie: pie chart."""
# 
# import numpy as np
# 
# 
# def plot_mpl_pie(plt, rng, ax=None):
#     """mpl_pie - pie chart.
# 
#     Demonstrates: ax.mpl_pie() - identical to ax.pie()
#     """
#     if ax is None:
#         fig, ax = plt.subplots()
#     else:
#         fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex
# 
#     sizes = rng.uniform(10, 30, 5)
#     labels = ['A', 'B', 'C', 'D', 'E']
#     ax.mpl_pie(sizes, labels=labels, autopct='%1.1f%%')
#     ax.set_xyt("X", "Y", "mpl_pie")
#     if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
#         ax.legend()
#     return fig, ax
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_mpl_pie.py
# --------------------------------------------------------------------------------
