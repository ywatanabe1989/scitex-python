# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/metadata/_dimensions.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: scitex/plt/utils/metadata/_dimensions.py
# 
# """
# Dimension extraction utilities for axes and figure metadata.
# 
# This module provides functions to extract size, position, and bounding box
# information from matplotlib figures and axes in multiple units (mm, inch, px).
# """
# 
# from typing import Dict
# 
# from scitex import logging
# 
# logger = logging.getLogger(__name__)
# 
# 
# def _extract_axes_dimensions(fig, ax, ax_index: int) -> dict:
#     """
#     Extract dimension information for a single axes.
# 
#     Parameters
#     ----------
#     fig : matplotlib.figure.Figure
#         The parent figure
#     ax : matplotlib.axes.Axes
#         The axes to extract dimensions from
#     ax_index : int
#         Index of this axes in the figure (for position tracking)
# 
#     Returns
#     -------
#     dict
#         Dimension metadata containing:
#         - size_mm, size_inch, size_px
#         - bounds_figure_fraction
#         - position_in_grid
#         - margins_mm, margins_inch
#         - bbox_mm, bbox_inch, bbox_px
#     """
#     ax_metadata = {}
# 
#     try:
#         from .._figure_from_axes_mm import get_dimension_info
# 
#         dim_info = get_dimension_info(fig, ax)
# 
#         # Size in multiple units
#         ax_metadata["size_mm"] = dim_info.get("axes_size_mm", [])
#         if "axes_size_inch" in dim_info:
#             ax_metadata["size_inch"] = dim_info["axes_size_inch"]
#         if "axes_size_px" in dim_info:
#             ax_metadata["size_px"] = dim_info["axes_size_px"]
# 
#         # Position in figure coordinates (normalized 0-1 values)
#         # Uses matplotlib terminology: bounds_figure_fraction
#         if "axes_position" in dim_info:
#             ax_metadata["bounds_figure_fraction"] = list(dim_info["axes_position"])
# 
#         # Position in grid (row, col)
#         if hasattr(ax, "_scitex_metadata") and "position_in_grid" in ax._scitex_metadata:
#             ax_metadata["position_in_grid"] = ax._scitex_metadata["position_in_grid"]
#         else:
#             # Calculate from ax_index if we have grid info
#             ax_metadata["position_in_grid"] = [ax_index, 0]  # Default single column
# 
#         # Margins in mm and inch
#         if "margins_mm" in dim_info:
#             ax_metadata["margins_mm"] = dim_info["margins_mm"]
#         if "margins_inch" in dim_info:
#             ax_metadata["margins_inch"] = dim_info["margins_inch"]
# 
#         # Bounding box with intuitive keys
#         if "axes_bbox_px" in dim_info:
#             bbox = dim_info["axes_bbox_px"]
#             ax_metadata["bbox_px"] = _normalize_bbox(bbox)
#         if "axes_bbox_mm" in dim_info:
#             bbox = dim_info["axes_bbox_mm"]
#             ax_metadata["bbox_mm"] = _normalize_bbox(bbox)
#         if "axes_bbox_inch" in dim_info:
#             bbox = dim_info["axes_bbox_inch"]
#             ax_metadata["bbox_inch"] = _normalize_bbox(bbox)
# 
#     except Exception as e:
#         logger.warning(f"Could not extract dimension info for axes {ax_index}: {e}")
# 
#     return ax_metadata
# 
# 
# def _normalize_bbox(bbox: dict) -> dict:
#     """
#     Normalize bounding box dictionary to consistent format.
# 
#     Converts from x0/y0/x1/y1 to x_left/y_bottom/x_right/y_top format.
# 
#     Parameters
#     ----------
#     bbox : dict
#         Bounding box with either x0/y0/x1/y1 or x_left/y_bottom/x_right/y_top keys
# 
#     Returns
#     -------
#     dict
#         Normalized bounding box with x_left, x_right, y_top, y_bottom, width, height
#     """
#     return {
#         "x_left": bbox.get("x0", bbox.get("x_left", 0)),
#         "x_right": bbox.get("x1", bbox.get("x_right", 0)),
#         "y_top": bbox.get("y0", bbox.get("y_top", 0)),
#         "y_bottom": bbox.get("y1", bbox.get("y_bottom", 0)),
#         "width": bbox.get("width", 0),
#         "height": bbox.get("height", 0),
#     }
# 
# 
# def _extract_axis_info(ax, axis_name: str = "xaxis") -> dict:
#     """
#     Extract axis information (label, unit, scale, limits).
# 
#     Parameters
#     ----------
#     ax : matplotlib.axes.Axes
#         The axes to extract axis info from
#     axis_name : str
#         Either 'xaxis' or 'yaxis'
# 
#     Returns
#     -------
#     dict
#         Axis metadata with label, unit, scale, and lim
#     """
#     from ._label_parsing import _parse_label_unit
# 
#     if axis_name == "xaxis":
#         label_text = ax.get_xlabel()
#         scale = ax.get_xscale()
#         lim = list(ax.get_xlim())
#     else:  # yaxis
#         label_text = ax.get_ylabel()
#         scale = ax.get_yscale()
#         lim = list(ax.get_ylim())
# 
#     label, unit = _parse_label_unit(label_text)
# 
#     return {
#         "label": label,
#         "unit": unit,
#         "scale": scale,
#         "lim": lim,
#     }
# 
# 
# def _extract_figure_dimensions(fig) -> dict:
#     """
#     Extract figure-level dimensions.
# 
#     Parameters
#     ----------
#     fig : matplotlib.figure.Figure
#         The figure to extract dimensions from
# 
#     Returns
#     -------
#     dict
#         Figure dimensions with size_mm, size_inch, size_px, and dpi
#     """
#     fig_metadata = {}
# 
#     try:
#         # Get figure size from metadata if available
#         if hasattr(fig, "_scitex_metadata"):
#             scitex_meta = fig._scitex_metadata
#             if "size_mm" in scitex_meta:
#                 fig_metadata["size_mm"] = list(scitex_meta["size_mm"])
#             if "size_inch" in scitex_meta:
#                 fig_metadata["size_inch"] = list(scitex_meta["size_inch"])
# 
#         # Get figure size in inches (always available)
#         width_inch, height_inch = fig.get_size_inches()
#         if "size_inch" not in fig_metadata:
#             fig_metadata["size_inch"] = [width_inch, height_inch]
# 
#         # Convert to mm if not in metadata
#         if "size_mm" not in fig_metadata:
#             fig_metadata["size_mm"] = [width_inch * 25.4, height_inch * 25.4]
# 
#         # Get DPI
#         dpi = fig.get_dpi()
#         fig_metadata["dpi"] = dpi
# 
#         # Calculate pixel size
#         fig_metadata["size_px"] = [width_inch * dpi, height_inch * dpi]
# 
#     except Exception as e:
#         logger.warning(f"Could not extract figure dimensions: {e}")
# 
#     return fig_metadata
# 
# 
# def _collect_grid_info(ax_wrapper) -> tuple:
#     """
#     Extract grid shape and axes positions from an axes wrapper.
# 
#     Parameters
#     ----------
#     ax_wrapper
#         AxesWrapper or similar object with _axes_scitex attribute
# 
#     Returns
#     -------
#     tuple
#         (all_axes, grid_shape) where all_axes is a list of (ax, row, col) tuples
#     """
#     import numpy as np
# 
#     all_axes = []
#     grid_shape = (1, 1)
# 
#     if hasattr(ax_wrapper, "_axes_scitex"):
#         axes_array = ax_wrapper._axes_scitex
#         if isinstance(axes_array, np.ndarray):
#             grid_shape = axes_array.shape
#             for i, ax_row in enumerate(axes_array):
#                 if axes_array.ndim == 1:
#                     # 1D array: treat as single column
#                     all_axes.append((ax_row, i, 0))
#                 else:
#                     # 2D array
#                     for j, ax_item in enumerate(ax_row):
#                         all_axes.append((ax_item, i, j))
#         else:
#             # Single axes
#             all_axes.append((axes_array, 0, 0))
#     elif hasattr(ax_wrapper, "_ax"):
#         # Single AxisWrapper
#         all_axes.append((ax_wrapper, 0, 0))
#     else:
#         # Fallback: matplotlib axes
#         all_axes.append((ax_wrapper, 0, 0))
# 
#     return all_axes, grid_shape

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/metadata/_dimensions.py
# --------------------------------------------------------------------------------
