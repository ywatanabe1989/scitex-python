# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_stx_errorbar.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: plot_stx_errorbar.py - stx_errorbar demo
# 
# """stx_errorbar: x, y, yerr arrays."""
# 
# import numpy as np
# 
# 
# def plot_stx_errorbar(plt, rng, ax=None):
#     """stx_errorbar - x, y, yerr arrays.
# 
#     Demonstrates: ax.stx_errorbar()
#     """
#     if ax is None:
#         fig, ax = plt.subplots()
#     else:
#         fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex
# 
#     x = np.arange(1, 6)
#     y = rng.uniform(2, 8, 5)
#     yerr = rng.uniform(0.5, 1.5, 5)
#     ax.stx_errorbar(x, y, yerr=yerr, label='Measurements')
#     ax.set_xyt("X", "Y", "stx_errorbar")
#     if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
#         ax.legend()
#     return fig, ax
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_stx_errorbar.py
# --------------------------------------------------------------------------------
