# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_stx_bar.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """CSV formatter for stx_bar() calls."""
# 
# import pandas as pd
# import numpy as np
# from scitex.plt.utils._csv_column_naming import get_csv_column_name
# from ._format_plot import _parse_tracking_id
# 
# 
# def _format_stx_bar(id, tracked_dict, kwargs):
#     """Format data from stx_bar call for CSV export.
# 
#     Parameters
#     ----------
#     id : str
#         Tracking identifier
#     tracked_dict : dict
#         Dictionary containing tracked data with 'bar_df' key
#     kwargs : dict
#         Additional keyword arguments (may contain yerr)
# 
#     Returns
#     -------
#     pd.DataFrame
#         Formatted bar data with standardized column names
#     """
#     if not tracked_dict or not isinstance(tracked_dict, dict):
#         return pd.DataFrame()
# 
#     ax_row, ax_col, trace_id = _parse_tracking_id(id)
# 
#     # Get bar_df from tracked data
#     bar_df = tracked_dict.get("bar_df")
#     if bar_df is not None and isinstance(bar_df, pd.DataFrame):
#         result = bar_df.copy()
#         renamed = {}
#         # Map 'x' and 'height' to standardized column names
#         for col in result.columns:
#             if col == "x":
#                 renamed[col] = get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id)
#             elif col == "height":
#                 renamed[col] = get_csv_column_name("y", ax_row, ax_col, trace_id=trace_id)
#             else:
#                 renamed[col] = get_csv_column_name(col, ax_row, ax_col, trace_id=trace_id)
# 
#         result = result.rename(columns=renamed)
# 
#         # Add yerr if present in kwargs
#         yerr = kwargs.get("yerr") if kwargs else None
#         if yerr is not None:
#             try:
#                 yerr_array = np.asarray(yerr)
#                 if len(yerr_array) == len(result):
#                     col_yerr = get_csv_column_name("yerr", ax_row, ax_col, trace_id=trace_id)
#                     result[col_yerr] = yerr_array
#             except (TypeError, ValueError):
#                 pass
# 
#         return result
# 
#     # Fallback to args if bar_df not found
#     args = tracked_dict.get("args", [])
#     if len(args) >= 2:
#         col_x = get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id)
#         col_y = get_csv_column_name("y", ax_row, ax_col, trace_id=trace_id)
#         data = {col_x: args[0], col_y: args[1]}
# 
#         # Add yerr if present
#         yerr = kwargs.get("yerr") if kwargs else None
#         if yerr is not None:
#             try:
#                 yerr_array = np.asarray(yerr)
#                 col_yerr = get_csv_column_name("yerr", ax_row, ax_col, trace_id=trace_id)
#                 data[col_yerr] = yerr_array
#             except (TypeError, ValueError):
#                 pass
# 
#         return pd.DataFrame(data)
# 
#     return pd.DataFrame()
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_stx_bar.py
# --------------------------------------------------------------------------------
