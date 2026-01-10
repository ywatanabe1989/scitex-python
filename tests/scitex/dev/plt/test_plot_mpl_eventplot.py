# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_mpl_eventplot.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: plot_mpl_eventplot.py - mpl_eventplot demo
# 
# """mpl_eventplot: event plot."""
# 
# import numpy as np
# 
# 
# def plot_mpl_eventplot(plt, rng, ax=None):
#     """mpl_eventplot - event plot.
# 
#     Demonstrates: ax.mpl_eventplot() - identical to ax.eventplot()
#     """
#     if ax is None:
#         fig, ax = plt.subplots()
#     else:
#         fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex
# 
#     positions = [rng.uniform(0, 10, rng.integers(5, 15)) for _ in range(5)]
#     ax.mpl_eventplot(positions, orientation='horizontal')
#     ax.set_xyt("X", "Y", "mpl_eventplot")
#     if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
#         ax.legend()
#     return fig, ax
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_mpl_eventplot.py
# --------------------------------------------------------------------------------
