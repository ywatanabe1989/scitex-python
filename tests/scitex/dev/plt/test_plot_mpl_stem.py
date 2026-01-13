# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_mpl_stem.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: plot_mpl_stem.py - mpl_stem demo
# 
# """mpl_stem: stem plot."""
# 
# import numpy as np
# 
# 
# def plot_mpl_stem(plt, rng, ax=None):
#     """mpl_stem - stem plot.
# 
#     Demonstrates: ax.mpl_stem() - identical to ax.stem()
#     """
#     if ax is None:
#         fig, ax = plt.subplots()
#     else:
#         fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex
# 
#     x = np.arange(20)
#     y = rng.uniform(-1, 1, 20)
#     ax.mpl_stem(x, y)
#     ax.set_xyt("X", "Y", "mpl_stem")
#     if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
#         ax.legend()
#     return fig, ax
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_mpl_stem.py
# --------------------------------------------------------------------------------
