# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/metadata/_recipe_extraction.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: scitex/plt/utils/metadata/_recipe_extraction.py
# 
# """
# Recipe metadata extraction utilities.
# 
# This module provides functions to extract reconstruction recipes from axes history,
# allowing figures to be recreated from metadata and CSV data.
# """
# 
# from typing import List, Dict
# 
# 
# def collect_recipe_metadata(
#     fig,
#     ax=None,
#     include_recipe: bool = True,
#     auto_crop: bool = True,
#     crop_margin_mm: float = 1.0,
# ) -> dict:
#     """
#     Collect minimal "recipe" metadata - much smaller than verbose schema.
# 
#     Uses scitex.plt.figure.recipe schema which only includes:
#     - Figure/axes dimensions
#     - Method calls with arguments (from ax.history)
#     - Data column references for CSV linkage
#     - Top-level axes_bbox_px for canvas alignment
# 
#     Parameters
#     ----------
#     fig : matplotlib.figure.Figure
#         Figure to collect metadata from
#     ax : optional
#         Axes or AxesWrapper
#     include_recipe : bool
#         Whether to include reconstruction recipe
#     auto_crop : bool
#         Whether auto_crop was enabled when saving
#     crop_margin_mm : float
#         Crop margin in mm (if auto_crop was used)
# 
#     Returns
#     -------
#     dict
#         Minimal metadata with recipe schema (~95% smaller than verbose)
#     """
#     # Import the recipe schema collector from _collect_figure_metadata
#     # This uses scitex.plt.figure.recipe schema (minimal)
#     from .._collect_figure_metadata import collect_recipe_metadata as _collect_recipe
# 
#     return _collect_recipe(fig, ax, auto_crop=auto_crop, crop_margin_mm=crop_margin_mm)
# 
# 
# def _extract_calls_from_history(ax, ax_index: int) -> List[dict]:
#     """
#     Extract method call records from axis history.
# 
#     Parameters
#     ----------
#     ax : AxisWrapper or matplotlib.axes.Axes
#         Axis to extract history from
#     ax_index : int
#         Index of axis in figure (for CSV column naming)
# 
#     Returns
#     -------
#     list
#         List of call records: [{id, method, data_ref, kwargs}, ...]
#     """
#     calls = []
# 
#     # Check for scitex wrapper with history
#     if not hasattr(ax, 'history') and not hasattr(ax, '_ax_history'):
#         return calls
# 
#     # Get history dict
#     history = getattr(ax, 'history', None)
#     if history is None:
#         history = getattr(ax, '_ax_history', {})
# 
#     # Get grid position
#     ax_row = 0
#     ax_col = 0
#     if hasattr(ax, "_scitex_metadata"):
#         pos = ax._scitex_metadata.get("position_in_grid", [0, 0])
#         ax_row, ax_col = pos[0], pos[1]
# 
#     for trace_id, record in history.items():
#         # record format: (id, method_name, tracked_dict, kwargs)
#         if not isinstance(record, (list, tuple)) or len(record) < 3:
#             continue
# 
#         call_id, method_name, tracked_dict = record[0], record[1], record[2]
#         kwargs = record[3] if len(record) > 3 else {}
# 
#         call = {
#             "id": str(call_id),
#             "method": method_name,
#         }
# 
#         # Build data_ref from tracked_dict to CSV column names
#         data_ref = _build_data_ref(call_id, method_name, tracked_dict, ax_row, ax_col)
#         if data_ref:
#             call["data_ref"] = data_ref
# 
#         # Filter kwargs to only style-relevant ones (not data)
#         style_kwargs = _filter_style_kwargs(kwargs, method_name)
#         if style_kwargs:
#             call["kwargs"] = style_kwargs
# 
#         calls.append(call)
# 
#     return calls
# 
# 
# def _build_data_ref(trace_id, method_name: str, tracked_dict: dict,
#                     ax_row: int, ax_col: int) -> dict:
#     """
#     Build data_ref mapping from tracked_dict to CSV column names.
# 
#     Parameters
#     ----------
#     trace_id : str
#         Trace identifier
#     method_name : str
#         Name of the method called
#     tracked_dict : dict
#         Data tracked by the method (contains arrays, dataframes)
#     ax_row, ax_col : int
#         Axis position in grid
# 
#     Returns
#     -------
#     dict
#         Mapping of variable names to CSV column names
#     """
#     prefix = f"ax-row-{ax_row}-col-{ax_col}_trace-id-{trace_id}_variable-"
# 
#     data_ref = {}
# 
#     # Method-specific column naming
#     if method_name == 'hist':
#         data_ref["raw_data"] = f"{prefix}raw-data"
#         data_ref["bin_centers"] = f"{prefix}bin-centers"
#         data_ref["bin_counts"] = f"{prefix}bin-counts"
#     elif method_name in ('plot', 'scatter', 'step', 'errorbar'):
#         data_ref["x"] = f"{prefix}x"
#         data_ref["y"] = f"{prefix}y"
#         if tracked_dict and 'yerr' in tracked_dict:
#             data_ref["yerr"] = f"{prefix}yerr"
#         if tracked_dict and 'xerr' in tracked_dict:
#             data_ref["xerr"] = f"{prefix}xerr"
#     elif method_name in ('bar', 'barh'):
#         data_ref["x"] = f"{prefix}x"
#         data_ref["y"] = f"{prefix}y"
#     elif method_name == 'stem':
#         data_ref["x"] = f"{prefix}x"
#         data_ref["y"] = f"{prefix}y"
#     elif method_name in ('fill_between', 'fill_betweenx'):
#         data_ref["x"] = f"{prefix}x"
#         data_ref["y1"] = f"{prefix}y1"
#         data_ref["y2"] = f"{prefix}y2"
#     elif method_name in ('imshow', 'matshow', 'pcolormesh'):
#         data_ref["data"] = f"{prefix}data"
#     elif method_name in ('contour', 'contourf'):
#         data_ref["x"] = f"{prefix}x"
#         data_ref["y"] = f"{prefix}y"
#         data_ref["z"] = f"{prefix}z"
#     elif method_name in ('boxplot', 'violinplot'):
#         data_ref["data"] = f"{prefix}data"
#     elif method_name == 'pie':
#         data_ref["x"] = f"{prefix}x"
#     elif method_name in ('quiver', 'streamplot'):
#         data_ref["x"] = f"{prefix}x"
#         data_ref["y"] = f"{prefix}y"
#         data_ref["u"] = f"{prefix}u"
#         data_ref["v"] = f"{prefix}v"
#     elif method_name == 'hexbin':
#         data_ref["x"] = f"{prefix}x"
#         data_ref["y"] = f"{prefix}y"
#     elif method_name == 'hist2d':
#         data_ref["x"] = f"{prefix}x"
#         data_ref["y"] = f"{prefix}y"
#     elif method_name == 'kde':
#         data_ref["x"] = f"{prefix}x"
#         data_ref["y"] = f"{prefix}y"
#     # SciTeX custom methods (stx_*)
#     elif method_name == 'stx_line':
#         data_ref["x"] = f"{prefix}x"
#         data_ref["y"] = f"{prefix}y"
#     elif method_name in ('stx_mean_std', 'stx_mean_ci', 'stx_median_iqr', 'stx_shaded_line'):
#         data_ref["x"] = f"{prefix}x"
#         data_ref["y_lower"] = f"{prefix}y-lower"
#         data_ref["y_middle"] = f"{prefix}y-middle"
#         data_ref["y_upper"] = f"{prefix}y-upper"
#     elif method_name in ('stx_box', 'stx_violin'):
#         data_ref["data"] = f"{prefix}data"
#     elif method_name == 'stx_scatter_hist':
#         data_ref["x"] = f"{prefix}x"
#         data_ref["y"] = f"{prefix}y"
#     elif method_name in ('stx_heatmap', 'stx_conf_mat', 'stx_image', 'stx_raster'):
#         data_ref["data"] = f"{prefix}data"
#     elif method_name in ('stx_kde', 'stx_ecdf'):
#         data_ref["x"] = f"{prefix}x"
#         data_ref["y"] = f"{prefix}y"
#     elif method_name.startswith('stx_'):
#         data_ref["x"] = f"{prefix}x"
#         data_ref["y"] = f"{prefix}y"
#     else:
#         # Generic fallback
#         if tracked_dict:
#             if 'x' in tracked_dict or 'args' in tracked_dict:
#                 data_ref["x"] = f"{prefix}x"
#                 data_ref["y"] = f"{prefix}y"
# 
#     return data_ref
# 
# 
# def _filter_style_kwargs(kwargs: dict, method_name: str) -> dict:
#     """
#     Filter kwargs to only include style-relevant parameters.
# 
#     Removes data arrays and internal parameters, keeps style settings
#     that affect appearance (color, linewidth, etc.).
# 
#     Parameters
#     ----------
#     kwargs : dict
#         Original keyword arguments
#     method_name : str
#         Name of the method
# 
#     Returns
#     -------
#     dict
#         Filtered kwargs with only style parameters
#     """
#     if not kwargs:
#         return {}
# 
#     # Style-relevant kwargs to keep
#     style_keys = {
#         'color', 'c', 'facecolor', 'edgecolor', 'linecolor',
#         'linewidth', 'lw', 'linestyle', 'ls',
#         'marker', 'markersize', 'ms', 'markerfacecolor', 'markeredgecolor',
#         'alpha', 'zorder',
#         'label',
#         'bins', 'density', 'histtype', 'orientation',
#         'width', 'height', 'align',
#         'cmap', 'vmin', 'vmax', 'norm',
#         'levels', 'extend',
#         'scale', 'units',
#         'autopct', 'explode', 'shadow', 'startangle',
#     }
# 
#     filtered = {}
#     for key, value in kwargs.items():
#         if key in style_keys:
#             # Skip if value is a large array (data, not style)
#             if hasattr(value, '__len__') and not isinstance(value, str):
#                 if len(value) > 10:
#                     continue
#             filtered[key] = value
# 
#     return filtered

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/metadata/_recipe_extraction.py
# --------------------------------------------------------------------------------
