# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_stx_mean_std.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: plot_stx_mean_std.py - stx_mean_std demo
# 
# """stx_mean_std: 2D array (samples x time)."""
# 
# import numpy as np
# 
# 
# def plot_stx_mean_std(plt, rng, ax=None):
#     """stx_mean_std - 2D array (samples x time).
# 
#     Demonstrates: ax.stx_mean_std()
#     """
#     if ax is None:
#         fig, ax = plt.subplots()
#     else:
#         fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex
# 
#     data = rng.normal(0, 1, (100, 50)) + np.linspace(0, 2, 50)
#     ax.stx_mean_std(data, label='Mean +/- std')
#     ax.set_xyt("X", "Y", "stx_mean_std")
#     if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
#         ax.legend()
#     return fig, ax
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_stx_mean_std.py
# --------------------------------------------------------------------------------
