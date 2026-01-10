# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_mpl_axhline.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: plot_mpl_axhline.py - mpl_axhline demo
# 
# """mpl_axhline: horizontal line."""
# 
# import numpy as np
# 
# 
# def plot_mpl_axhline(plt, rng, ax=None):
#     """mpl_axhline - horizontal line.
# 
#     Demonstrates: ax.mpl_axhline() - identical to ax.axhline()
#     """
#     if ax is None:
#         fig, ax = plt.subplots()
#     else:
#         fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex
# 
#     ax.plot(rng.uniform(0, 10, 20))
#     ax.mpl_axhline(y=5, color='r', linestyle='--', label='threshold')
#     ax.set_xyt("X", "Y", "mpl_axhline")
#     if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
#         ax.legend()
#     return fig, ax
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_mpl_axhline.py
# --------------------------------------------------------------------------------
