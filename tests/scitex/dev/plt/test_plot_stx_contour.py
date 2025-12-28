# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_stx_contour.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: plot_stx_contour.py - stx_contour demo
# 
# """stx_contour: 2D array contour."""
# 
# import numpy as np
# 
# 
# def plot_stx_contour(plt, rng, ax=None):
#     """stx_contour - 2D array contour.
# 
#     Demonstrates: ax.stx_contour()
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
#     ax.stx_contour(X, Y, Z, levels=10)
#     ax.set_xyt("X", "Y", "stx_contour")
#     if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
#         ax.legend()
#     return fig, ax
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_stx_contour.py
# --------------------------------------------------------------------------------
