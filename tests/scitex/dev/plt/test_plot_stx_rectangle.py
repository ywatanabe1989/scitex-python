# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_stx_rectangle.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: plot_stx_rectangle.py - stx_rectangle demo
# 
# """stx_rectangle: rectangle annotation."""
# 
# import numpy as np
# 
# 
# def plot_stx_rectangle(plt, rng, ax=None):
#     """stx_rectangle - rectangle annotation.
# 
#     Demonstrates: ax.stx_rectangle()
#     """
#     if ax is None:
#         fig, ax = plt.subplots()
#     else:
#         fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex
# 
#     ax.plot(np.sin(np.linspace(0, 4*np.pi, 100)))
#     ax.stx_rectangle(20, 40, -0.5, 0.5, alpha=0.3)
#     ax.set_xyt("X", "Y", "stx_rectangle")
#     if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
#         ax.legend()
#     return fig, ax
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_stx_rectangle.py
# --------------------------------------------------------------------------------
