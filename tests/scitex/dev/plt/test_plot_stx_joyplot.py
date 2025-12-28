# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_stx_joyplot.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: plot_stx_joyplot.py - stx_joyplot demo
# 
# """stx_joyplot: list of distributions."""
# 
# import numpy as np
# 
# 
# def plot_stx_joyplot(plt, rng, ax=None):
#     """stx_joyplot - list of distributions.
# 
#     Demonstrates: ax.stx_joyplot()
#     """
#     if ax is None:
#         fig, ax = plt.subplots()
#     else:
#         fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex
# 
#     data = [rng.normal(i, 1, 200) for i in range(5)]
#     ax.stx_joyplot(data, labels=['A', 'B', 'C', 'D', 'E'])
#     ax.set_xyt("X", "Y", "stx_joyplot")
#     if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
#         ax.legend()
#     return fig, ax
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_stx_joyplot.py
# --------------------------------------------------------------------------------
