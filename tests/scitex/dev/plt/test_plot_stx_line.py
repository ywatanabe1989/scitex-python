# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_stx_line.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: plot_stx_line.py - stx_line demo
# 
# """stx_line: 1D array."""
# 
# import numpy as np
# 
# 
# def plot_stx_line(plt, rng, ax=None):
#     """stx_line - 1D array.
# 
#     Demonstrates: ax.stx_line()
#     """
#     if ax is None:
#         fig, ax = plt.subplots()
#     else:
#         fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex
# 
#     y = np.sin(np.linspace(0, 4*np.pi, 100)) + rng.normal(0, 0.1, 100)
#     ax.stx_line(y, label='Signal')
#     ax.set_xyt("X", "Y", "stx_line")
#     if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
#         ax.legend()
#     return fig, ax
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_stx_line.py
# --------------------------------------------------------------------------------
