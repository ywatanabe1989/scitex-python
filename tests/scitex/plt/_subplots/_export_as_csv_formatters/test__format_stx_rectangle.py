# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_stx_rectangle.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-10 12:00:00 (ywatanabe)"
# # File: ./src/scitex/plt/_subplots/_export_as_csv_formatters/_format_stx_rectangle.py
# 
# """CSV formatter for stx_rectangle() calls - uses standard column naming."""
# 
# import numpy as np
# import pandas as pd
# 
# from scitex.plt.utils._csv_column_naming import get_csv_column_name
# 
# from ._format_plot import _parse_tracking_id
# 
# 
# def _format_plot_rectangle(id, tracked_dict, kwargs):
#     """Format data from a stx_rectangle call.
# 
#     Uses standard column naming convention:
#     (ax-row-{r}-col-{c}_trace-id-{id}_variable-{var}).
# 
#     Args:
#         id (str): Identifier for the plot
#         tracked_dict (dict): Dictionary containing tracked data
#         kwargs (dict): Keyword arguments passed to stx_rectangle
# 
#     Returns:
#         pd.DataFrame: Formatted rectangle data
#     """
#     # Check if tracked_dict is empty or not a dictionary
#     if not tracked_dict or not isinstance(tracked_dict, dict):
#         return pd.DataFrame()
# 
#     # Parse tracking ID to get axes position
#     ax_row, ax_col, trace_id = _parse_tracking_id(id)
# 
#     # Get standard column names
#     x_col = get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id)
#     y_col = get_csv_column_name("y", ax_row, ax_col, trace_id=trace_id)
#     width_col = get_csv_column_name("width", ax_row, ax_col, trace_id=trace_id)
#     height_col = get_csv_column_name("height", ax_row, ax_col, trace_id=trace_id)
# 
#     # Try to get rectangle parameters directly from tracked_dict
#     x = tracked_dict.get("x")
#     y = tracked_dict.get("y")
#     width = tracked_dict.get("width")
#     height = tracked_dict.get("height")
# 
#     # If direct parameters aren't available, try the args
#     if any(param is None for param in [x, y, width, height]):
#         args = tracked_dict.get("args", [])
# 
#         # Rectangles defined by [x, y, width, height]
#         if len(args) >= 4:
#             x, y, width, height = args[0], args[1], args[2], args[3]
# 
#     # If we have all required parameters, create the DataFrame
#     if all(param is not None for param in [x, y, width, height]):
#         try:
#             # Handle single rectangle
#             if all(
#                 isinstance(val, (int, float, np.number))
#                 for val in [x, y, width, height]
#             ):
#                 return pd.DataFrame(
#                     {
#                         x_col: [x],
#                         y_col: [y],
#                         width_col: [width],
#                         height_col: [height],
#                     }
#                 )
# 
#             # Handle multiple rectangles (arrays)
#             elif all(
#                 isinstance(val, (np.ndarray, list)) for val in [x, y, width, height]
#             ):
#                 try:
#                     return pd.DataFrame(
#                         {
#                             x_col: x,
#                             y_col: y,
#                             width_col: width,
#                             height_col: height,
#                         }
#                     )
#                 except ValueError:
#                     # Handle case where arrays might be different lengths
#                     result = pd.DataFrame()
#                     result[x_col] = pd.Series(x)
#                     result[y_col] = pd.Series(y)
#                     result[width_col] = pd.Series(width)
#                     result[height_col] = pd.Series(height)
#                     return result
#         except Exception:
#             # Fallback for rectangle in case of any errors
#             try:
#                 return pd.DataFrame(
#                     {
#                         x_col: [float(x) if x is not None else 0],
#                         y_col: [float(y) if y is not None else 0],
#                         width_col: [float(width) if width is not None else 0],
#                         height_col: [float(height) if height is not None else 0],
#                     }
#                 )
#             except (TypeError, ValueError):
#                 pass
# 
#     # Check directly in the kwargs for the parameters
#     rect_x = kwargs.get("x")
#     rect_y = kwargs.get("y")
#     rect_w = kwargs.get("width")
#     rect_h = kwargs.get("height")
# 
#     if all(param is not None for param in [rect_x, rect_y, rect_w, rect_h]):
#         try:
#             return pd.DataFrame(
#                 {
#                     x_col: [float(rect_x)],
#                     y_col: [float(rect_y)],
#                     width_col: [float(rect_w)],
#                     height_col: [float(rect_h)],
#                 }
#             )
#         except (TypeError, ValueError):
#             pass
# 
#     # Default empty DataFrame if nothing could be processed
#     return pd.DataFrame()

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_stx_rectangle.py
# --------------------------------------------------------------------------------
