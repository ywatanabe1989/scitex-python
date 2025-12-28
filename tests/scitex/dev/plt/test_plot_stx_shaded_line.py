# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_stx_shaded_line.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: plot_stx_shaded_line.py - stx_shaded_line demo
# 
# """stx_shaded_line: line with shading."""
# 
# import numpy as np
# 
# 
# def plot_stx_shaded_line(plt, rng, ax=None):
#     """stx_shaded_line - line with shading.
# 
#     Demonstrates: ax.stx_shaded_line()
#     """
#     if ax is None:
#         fig, ax = plt.subplots()
#     else:
#         fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex
# 
#     x = np.linspace(0, 10, 100)
#     y = np.sin(x)
#     ax.stx_shaded_line(x, y, y - 0.2, y + 0.2, label='Shaded')
#     ax.set_xyt("X", "Y", "stx_shaded_line")
#     if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
#         ax.legend()
#     return fig, ax
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_stx_shaded_line.py
# --------------------------------------------------------------------------------
