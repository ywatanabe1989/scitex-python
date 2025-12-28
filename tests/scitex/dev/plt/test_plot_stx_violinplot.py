# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_stx_violinplot.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: plot_stx_violinplot.py - stx_violinplot demo
# 
# """stx_violinplot: matplotlib violinplot."""
# 
# import numpy as np
# 
# 
# def plot_stx_violinplot(plt, rng, ax=None):
#     """stx_violinplot - matplotlib violinplot.
# 
#     Demonstrates: ax.stx_violinplot()
#     """
#     if ax is None:
#         fig, ax = plt.subplots()
#     else:
#         fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex
# 
#     data = [rng.normal(i, 1, 100) for i in range(4)]
#     ax.stx_violinplot(data)
#     ax.set_xyt("X", "Y", "stx_violinplot")
#     if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
#         ax.legend()
#     return fig, ax
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_stx_violinplot.py
# --------------------------------------------------------------------------------
