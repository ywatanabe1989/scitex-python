# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_stx_raster.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: plot_stx_raster.py - stx_raster demo
# 
# """stx_raster: spike times list."""
# 
# import numpy as np
# 
# 
# def plot_stx_raster(plt, rng, ax=None):
#     """stx_raster - spike times list.
# 
#     Demonstrates: ax.stx_raster()
#     """
#     if ax is None:
#         fig, ax = plt.subplots()
#     else:
#         fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex
# 
#     spikes = [rng.uniform(0, 100, rng.integers(10, 30)) for _ in range(10)]
#     ax.stx_raster(spikes)
#     ax.set_xyt("X", "Y", "stx_raster")
#     if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
#         ax.legend()
#     return fig, ax
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_stx_raster.py
# --------------------------------------------------------------------------------
