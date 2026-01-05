# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_bar.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-19 15:45:51 (ywatanabe)"
# # File: /data/gpfs/projects/punim2354/ywatanabe/scitex_repo/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_bar.py
# # ----------------------------------------
# import os
# 
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# import pandas as pd
# import numpy as np
# 
# from scitex.plt.utils._csv_column_naming import get_csv_column_name
# from ._format_plot import _parse_tracking_id
# 
# 
# def _format_bar(id, tracked_dict, kwargs):
#     """Format data from a bar call for CSV export.
# 
#     Includes x, y values and optional yerr for error bars.
# 
#     Args:
#         id: The identifier for the plot
#         tracked_dict: Dictionary of tracked data
#         kwargs: Original keyword arguments (may contain yerr)
# 
#     Returns:
#         pd.DataFrame: Formatted data ready for CSV export with x, y, and optional yerr
#     """
#     # Check if tracked_dict is empty or not a dictionary
#     if not tracked_dict or not isinstance(tracked_dict, dict):
#         return pd.DataFrame()
# 
#     # Parse tracking ID to get axes position and trace ID
#     ax_row, ax_col, trace_id = _parse_tracking_id(id)
# 
#     # Get structured column names
#     col_x = get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id)
#     col_y = get_csv_column_name("y", ax_row, ax_col, trace_id=trace_id)
#     col_yerr = get_csv_column_name("yerr", ax_row, ax_col, trace_id=trace_id)
# 
#     # Extract yerr from kwargs if present
#     yerr = kwargs.get("yerr") if kwargs else None
# 
#     # Check if we have the newer format with bar_data
#     if "bar_data" in tracked_dict and isinstance(
#         tracked_dict["bar_data"], pd.DataFrame
#     ):
#         # Use the pre-formatted DataFrame but keep only x and height (y)
#         df = tracked_dict["bar_data"].copy()
# 
#         # Keep only essential columns
#         essential_cols = [col for col in df.columns if col in ["x", "height"]]
#         if essential_cols:
#             df = df[essential_cols]
# 
#             # Rename using structured naming
#             rename_map = {}
#             if "x" in df.columns:
#                 rename_map["x"] = col_x
#             if "height" in df.columns:
#                 rename_map["height"] = col_y
# 
#             df = df.rename(columns=rename_map)
# 
#             # Add yerr if present
#             if yerr is not None:
#                 try:
#                     yerr_array = np.asarray(yerr)
#                     if len(yerr_array) == len(df):
#                         df[col_yerr] = yerr_array
#                 except (TypeError, ValueError):
#                     pass
# 
#             return df
# 
#     # Legacy format - get the args from tracked_dict
#     args = tracked_dict.get("args", [])
# 
#     # Extract x and y data if available
#     if len(args) >= 2:
#         x, y = args[0], args[1]
# 
#         # Convert to arrays if possible for consistent handling
#         try:
#             x_array = np.asarray(x)
#             y_array = np.asarray(y)
# 
#             # Create DataFrame with structured column names
#             data = {
#                 col_x: x_array,
#                 col_y: y_array,
#             }
# 
#             # Add yerr if present
#             if yerr is not None:
#                 try:
#                     yerr_array = np.asarray(yerr)
#                     if len(yerr_array) == len(x_array):
#                         data[col_yerr] = yerr_array
#                 except (TypeError, ValueError):
#                     pass
# 
#             return pd.DataFrame(data)
# 
#         except (TypeError, ValueError):
#             # Fall back to direct values if conversion fails
#             result = {col_x: x, col_y: y}
#             if yerr is not None:
#                 result[col_yerr] = yerr
#             return pd.DataFrame(result)
# 
#     # If we have tracked data in another format (like our MatplotlibPlotMixin bar method)
#     result = {}
# 
#     # Check for x position (might be in different keys)
#     for x_key in ["x", "xs", "positions"]:
#         if x_key in tracked_dict:
#             result[col_x] = tracked_dict[x_key]
#             break
# 
#     # Check for y values (might be in different keys)
#     for y_key in ["y", "ys", "height", "heights", "values"]:
#         if y_key in tracked_dict:
#             result[col_y] = tracked_dict[y_key]
#             break
# 
#     # Add yerr if present in kwargs
#     if yerr is not None and result:
#         try:
#             yerr_array = np.asarray(yerr)
#             result[col_yerr] = yerr_array
#         except (TypeError, ValueError):
#             pass
# 
#     return pd.DataFrame(result) if result else pd.DataFrame()

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_bar.py
# --------------------------------------------------------------------------------
