# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_mpl_step.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: plot_mpl_step.py - mpl_step demo
# 
# """mpl_step: step plot."""
# 
# import numpy as np
# 
# 
# def plot_mpl_step(plt, rng, ax=None):
#     """mpl_step - step plot.
# 
#     Demonstrates: ax.mpl_step() - identical to ax.step()
#     """
#     if ax is None:
#         fig, ax = plt.subplots()
#     else:
#         fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex
# 
#     x = np.arange(20)
#     y = rng.integers(0, 10, 20)
#     ax.mpl_step(x, y, where='mid', label='step')
#     ax.set_xyt("X", "Y", "mpl_step")
#     if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
#         ax.legend()
#     return fig, ax
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_mpl_step.py
# --------------------------------------------------------------------------------
