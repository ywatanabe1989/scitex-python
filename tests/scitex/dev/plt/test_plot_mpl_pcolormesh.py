# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_mpl_pcolormesh.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: plot_mpl_pcolormesh.py - mpl_pcolormesh demo
# 
# """mpl_pcolormesh: pseudocolor mesh."""
# 
# import numpy as np
# 
# 
# def plot_mpl_pcolormesh(plt, rng, ax=None):
#     """mpl_pcolormesh - pseudocolor mesh.
# 
#     Demonstrates: ax.mpl_pcolormesh() - identical to ax.pcolormesh()
#     """
#     if ax is None:
#         fig, ax = plt.subplots()
#     else:
#         fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex
# 
#     x = np.linspace(0, 5, 20)
#     y = np.linspace(0, 5, 20)
#     X, Y = np.meshgrid(x, y)
#     Z = np.sin(X) * np.cos(Y)
#     ax.mpl_pcolormesh(X, Y, Z, shading='auto')
#     ax.set_xyt("X", "Y", "mpl_pcolormesh")
#     if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
#         ax.legend()
#     return fig, ax
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_mpl_pcolormesh.py
# --------------------------------------------------------------------------------
