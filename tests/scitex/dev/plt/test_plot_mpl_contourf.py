# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_mpl_contourf.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: plot_mpl_contourf.py - mpl_contourf demo
# 
# """mpl_contourf: filled contour."""
# 
# import numpy as np
# 
# 
# def plot_mpl_contourf(plt, rng, ax=None):
#     """mpl_contourf - filled contour.
# 
#     Demonstrates: ax.mpl_contourf() - identical to ax.contourf()
#     """
#     if ax is None:
#         fig, ax = plt.subplots()
#     else:
#         fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex
# 
#     x = np.linspace(-3, 3, 50)
#     y = np.linspace(-3, 3, 50)
#     X, Y = np.meshgrid(x, y)
#     Z = np.exp(-(X**2 + Y**2))
#     ax.mpl_contourf(X, Y, Z, levels=10)
#     ax.set_xyt("X", "Y", "mpl_contourf")
#     if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
#         ax.legend()
#     return fig, ax
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_mpl_contourf.py
# --------------------------------------------------------------------------------
