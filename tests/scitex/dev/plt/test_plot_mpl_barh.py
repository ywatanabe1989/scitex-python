# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_mpl_barh.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: plot_mpl_barh.py - mpl_barh demo
# 
# """mpl_barh: horizontal bar."""
# 
# import numpy as np
# 
# 
# def plot_mpl_barh(plt, rng, ax=None):
#     """mpl_barh - horizontal bar.
# 
#     Demonstrates: ax.mpl_barh() - identical to ax.barh()
#     """
#     if ax is None:
#         fig, ax = plt.subplots()
#     else:
#         fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex
# 
#     y = [1, 2, 3, 4, 5]
#     width = rng.uniform(2, 8, 5)
#     ax.mpl_barh(y, width)
#     ax.set_xyt("X", "Y", "mpl_barh")
#     if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
#         ax.legend()
#     return fig, ax
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_mpl_barh.py
# --------------------------------------------------------------------------------
