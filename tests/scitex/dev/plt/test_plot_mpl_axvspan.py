# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_mpl_axvspan.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: plot_mpl_axvspan.py - mpl_axvspan demo
# 
# """mpl_axvspan: vertical span."""
# 
# import numpy as np
# 
# 
# def plot_mpl_axvspan(plt, rng, ax=None):
#     """mpl_axvspan - vertical span.
# 
#     Demonstrates: ax.mpl_axvspan() - identical to ax.axvspan()
#     """
#     if ax is None:
#         fig, ax = plt.subplots()
#     else:
#         fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex
# 
#     ax.plot(rng.uniform(0, 10, 20))
#     ax.mpl_axvspan(5, 15, alpha=0.3, color='yellow', label='range')
#     ax.set_xyt("X", "Y", "mpl_axvspan")
#     if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
#         ax.legend()
#     return fig, ax
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_mpl_axvspan.py
# --------------------------------------------------------------------------------
