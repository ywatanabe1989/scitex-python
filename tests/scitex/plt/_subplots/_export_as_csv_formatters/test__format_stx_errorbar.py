# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_stx_errorbar.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """CSV formatter for stx_errorbar() calls."""
# 
# import pandas as pd
# import numpy as np
# from scitex.plt.utils._csv_column_naming import get_csv_column_name
# from ._format_plot import _parse_tracking_id
# 
# 
# def _format_stx_errorbar(id, tracked_dict, kwargs):
#     """Format data from stx_errorbar call for CSV export.
# 
#     Parameters
#     ----------
#     id : str
#         Tracking identifier
#     tracked_dict : dict
#         Dictionary containing tracked data with 'errorbar_df' key
#     kwargs : dict
#         Additional keyword arguments (may contain yerr, xerr)
# 
#     Returns
#     -------
#     pd.DataFrame
#         Formatted errorbar data with standardized column names
#     """
#     if not tracked_dict or not isinstance(tracked_dict, dict):
#         return pd.DataFrame()
# 
#     ax_row, ax_col, trace_id = _parse_tracking_id(id)
# 
#     # Get errorbar_df from tracked data
#     errorbar_df = tracked_dict.get("errorbar_df")
#     if errorbar_df is not None and isinstance(errorbar_df, pd.DataFrame):
#         result = errorbar_df.copy()
#         renamed = {}
# 
#         # Map columns to standardized names
#         for col in result.columns:
#             if col == "x":
#                 renamed[col] = get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id)
#             elif col == "y":
#                 renamed[col] = get_csv_column_name("y", ax_row, ax_col, trace_id=trace_id)
#             elif col == "yerr":
#                 # Check if yerr is asymmetric (tuple/list of 2)
#                 yerr_value = result[col].iloc[0] if len(result) > 0 else None
#                 if isinstance(yerr_value, (list, tuple)) and len(yerr_value) == 2:
#                     # Handle asymmetric yerr separately below
#                     continue
#                 else:
#                     renamed[col] = get_csv_column_name("yerr", ax_row, ax_col, trace_id=trace_id)
#             elif col == "xerr":
#                 # Check if xerr is asymmetric (tuple/list of 2)
#                 xerr_value = result[col].iloc[0] if len(result) > 0 else None
#                 if isinstance(xerr_value, (list, tuple)) and len(xerr_value) == 2:
#                     # Handle asymmetric xerr separately below
#                     continue
#                 else:
#                     renamed[col] = get_csv_column_name("xerr", ax_row, ax_col, trace_id=trace_id)
#             else:
#                 renamed[col] = get_csv_column_name(col, ax_row, ax_col, trace_id=trace_id)
# 
#         result = result.rename(columns=renamed)
# 
#         # Handle asymmetric error bars if needed from kwargs
#         yerr = kwargs.get("yerr") if kwargs else None
#         xerr = kwargs.get("xerr") if kwargs else None
# 
#         if yerr is not None and isinstance(yerr, (list, tuple)) and len(yerr) == 2:
#             col_yerr_neg = get_csv_column_name("yerr-neg", ax_row, ax_col, trace_id=trace_id)
#             col_yerr_pos = get_csv_column_name("yerr-pos", ax_row, ax_col, trace_id=trace_id)
#             result[col_yerr_neg] = yerr[0]
#             result[col_yerr_pos] = yerr[1]
# 
#         if xerr is not None and isinstance(xerr, (list, tuple)) and len(xerr) == 2:
#             col_xerr_neg = get_csv_column_name("xerr-neg", ax_row, ax_col, trace_id=trace_id)
#             col_xerr_pos = get_csv_column_name("xerr-pos", ax_row, ax_col, trace_id=trace_id)
#             result[col_xerr_neg] = xerr[0]
#             result[col_xerr_pos] = xerr[1]
# 
#         return result
# 
#     # Fallback to args if errorbar_df not found
#     args = tracked_dict.get("args", [])
#     if len(args) >= 2:
#         col_x = get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id)
#         col_y = get_csv_column_name("y", ax_row, ax_col, trace_id=trace_id)
#         data = {col_x: args[0], col_y: args[1]}
# 
#         # Add error bars if present
#         yerr = kwargs.get("yerr") if kwargs else None
#         xerr = kwargs.get("xerr") if kwargs else None
# 
#         if yerr is not None:
#             if isinstance(yerr, (list, tuple)) and len(yerr) == 2:
#                 col_yerr_neg = get_csv_column_name("yerr-neg", ax_row, ax_col, trace_id=trace_id)
#                 col_yerr_pos = get_csv_column_name("yerr-pos", ax_row, ax_col, trace_id=trace_id)
#                 data[col_yerr_neg] = yerr[0]
#                 data[col_yerr_pos] = yerr[1]
#             else:
#                 col_yerr = get_csv_column_name("yerr", ax_row, ax_col, trace_id=trace_id)
#                 data[col_yerr] = yerr
# 
#         if xerr is not None:
#             if isinstance(xerr, (list, tuple)) and len(xerr) == 2:
#                 col_xerr_neg = get_csv_column_name("xerr-neg", ax_row, ax_col, trace_id=trace_id)
#                 col_xerr_pos = get_csv_column_name("xerr-pos", ax_row, ax_col, trace_id=trace_id)
#                 data[col_xerr_neg] = xerr[0]
#                 data[col_xerr_pos] = xerr[1]
#             else:
#                 col_xerr = get_csv_column_name("xerr", ax_row, ax_col, trace_id=trace_id)
#                 data[col_xerr] = xerr
# 
#         return pd.DataFrame(data)
# 
#     return pd.DataFrame()
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_stx_errorbar.py
# --------------------------------------------------------------------------------
