# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_stx_barh.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: plot_stx_barh.py - stx_barh demo
# 
# """stx_barh: y, width arrays."""
# 
# import numpy as np
# 
# 
# def plot_stx_barh(plt, rng, ax=None):
#     """stx_barh - y, width arrays.
# 
#     Demonstrates: ax.stx_barh()
#     """
#     if ax is None:
#         fig, ax = plt.subplots()
#     else:
#         fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex
# 
#     y = [1, 2, 3, 4, 5]
#     width = rng.uniform(2, 8, 5)
#     ax.stx_barh(y, width, label='Values')
#     ax.set_xyt("X", "Y", "stx_barh")
#     if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
#         ax.legend()
#     return fig, ax
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_stx_barh.py
# --------------------------------------------------------------------------------
