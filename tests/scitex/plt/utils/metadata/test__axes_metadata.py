# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/metadata/_axes_metadata.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: scitex/plt/utils/metadata/_axes_metadata.py
# 
# """
# Axes-level metadata collection.
# 
# This module provides functions to collect comprehensive metadata for individual
# axes objects, including dimensions, axis info, and plot content.
# """
# 
# from typing import Dict
# from ._dimensions import _extract_axes_dimensions, _extract_axis_info
# from ._label_parsing import _parse_label_unit
# 
# 
# def _collect_single_axes_metadata(fig, ax, ax_index: int) -> dict:
#     """
#     Collect metadata for a single axes object.
# 
#     Parameters
#     ----------
#     fig : matplotlib.figure.Figure
#         The parent figure
#     ax : matplotlib.axes.Axes
#         The axes to collect metadata from
#     ax_index : int
#         Index of this axes in the figure (for position tracking)
# 
#     Returns
#     -------
#     dict
#         Metadata dictionary for this axes containing:
#         - size_mm, size_inch, size_px
#         - position_ratio
#         - position_in_grid
#         - margins_mm, margins_inch
#         - bbox_mm, bbox_inch, bbox_px
#         - xaxis, yaxis (axis info)
#     """
#     # Extract dimension information
#     ax_metadata = _extract_axes_dimensions(fig, ax, ax_index)
# 
#     # Extract axes labels and units
#     ax_metadata["xaxis"] = _extract_axis_info(ax, "xaxis")
#     ax_metadata["yaxis"] = _extract_axis_info(ax, "yaxis")
# 
#     return ax_metadata
# 
# 
# def _collect_all_axes_metadata(all_axes, fig, grid_shape) -> dict:
#     """
#     Collect metadata for all axes in a figure.
# 
#     Parameters
#     ----------
#     all_axes : list
#         List of (ax, row, col) tuples
#     fig : matplotlib.figure.Figure
#         The parent figure
#     grid_shape : tuple
#         (n_rows, n_cols) shape of the axes grid
# 
#     Returns
#     -------
#     dict
#         Nested dictionary with ax_00, ax_01, etc. keys containing axes metadata
#     """
#     from ._plot_content import _extract_artists, _extract_legend_info
# 
#     axes_metadata = {}
# 
#     for ax_index, (ax_item, row, col) in enumerate(all_axes):
#         # Get the underlying matplotlib axes
#         mpl_ax = ax_item._axis_mpl if hasattr(ax_item, "_axis_mpl") else ax_item
# 
#         # Use ax_item (wrapper) for metadata, mpl_ax for matplotlib properties
#         ax_data = _collect_single_axes_metadata(fig, ax_item, ax_index)
# 
#         # Store position in grid
#         ax_data["position_in_grid"] = [row, col]
# 
#         # Extract plot content (artists and legend)
#         ax_data["artists"] = _extract_artists(ax_item)
#         legend_info = _extract_legend_info(mpl_ax)
#         if legend_info:
#             ax_data["legend"] = legend_info
# 
#         # Store with formatted key: ax_00, ax_01, ax_10, etc.
#         ax_key = f"ax_{row:02d}_{col:02d}"
#         axes_metadata[ax_key] = ax_data
# 
#     return axes_metadata

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/metadata/_axes_metadata.py
# --------------------------------------------------------------------------------
