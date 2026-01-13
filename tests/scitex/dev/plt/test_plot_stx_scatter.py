# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_stx_scatter.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: plot_stx_scatter.py - stx_scatter demo
# 
# """stx_scatter: x, y arrays."""
# 
# import numpy as np
# 
# 
# def plot_stx_scatter(plt, rng, ax=None):
#     """stx_scatter - x, y arrays.
# 
#     Demonstrates: ax.stx_scatter()
#     """
#     if ax is None:
#         fig, ax = plt.subplots()
#     else:
#         fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex
# 
#     x = rng.uniform(0, 10, 50)
#     y = 2*x + rng.normal(0, 2, 50)
#     ax.stx_scatter(x, y, label='Data')
#     ax.set_xyt("X", "Y", "stx_scatter")
#     if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
#         ax.legend()
#     return fig, ax
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_stx_scatter.py
# --------------------------------------------------------------------------------
