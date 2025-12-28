# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_mpl_imshow.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: plot_mpl_imshow.py - mpl_imshow demo
# 
# """mpl_imshow: image show."""
# 
# import numpy as np
# 
# 
# def plot_mpl_imshow(plt, rng, ax=None):
#     """mpl_imshow - image show.
# 
#     Demonstrates: ax.mpl_imshow() - identical to ax.imshow()
#     """
#     if ax is None:
#         fig, ax = plt.subplots()
#     else:
#         fig = ax.get_figure() if hasattr(ax, "get_figure") else ax._fig_scitex
# 
#     data = rng.uniform(0, 1, (10, 10))
#     im = ax.mpl_imshow(data, cmap='viridis', aspect='auto')
#     fig.colorbar(im, ax=ax)
#     ax.set_xyt("X", "Y", "mpl_imshow")
#     if hasattr(ax, 'legend') and ax.get_legend_handles_labels()[0]:
#         ax.legend()
#     return fig, ax
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/plot_mpl_imshow.py
# --------------------------------------------------------------------------------
