# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_mpl_fill_between.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: plot_mpl_fill_between.py - mpl_fill_between demo
# 
# """mpl_fill_between: fill between."""
# 
# import numpy as np
# 
# 
# def plot_mpl_fill_between(plt, rng, ax=None):
#     """mpl_fill_between - fill between.
# 
#     Demonstrates: ax.mpl_fill_between() - identical to ax.fill_between()
#     """
#     if ax is None:
#         fig, ax = plt.subplots()
#     else:
#         fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex
# 
#     x = np.linspace(0, 2*np.pi, 100)
#     y1, y2 = np.sin(x), np.sin(x) + 0.5
#     ax.mpl_fill_between(x, y1, y2, alpha=0.3)
#     ax.plot(x, y1)
#     ax.plot(x, y2)
#     ax.set_xyt("X", "Y", "mpl_fill_between")
#     if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
#         ax.legend()
#     return fig, ax
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_mpl_fill_between.py
# --------------------------------------------------------------------------------
