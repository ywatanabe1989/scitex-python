# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_stx_barh.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """CSV formatter for stx_barh() calls."""
# 
# import pandas as pd
# import numpy as np
# from scitex.plt.utils._csv_column_naming import get_csv_column_name
# from ._format_plot import _parse_tracking_id
# 
# 
# def _format_stx_barh(id, tracked_dict, kwargs):
#     """Format data from stx_barh call for CSV export.
# 
#     Parameters
#     ----------
#     id : str
#         Tracking identifier
#     tracked_dict : dict
#         Dictionary containing tracked data with 'barh_df' key
#     kwargs : dict
#         Additional keyword arguments (may contain xerr)
# 
#     Returns
#     -------
#     pd.DataFrame
#         Formatted barh data with standardized column names
#     """
#     if not tracked_dict or not isinstance(tracked_dict, dict):
#         return pd.DataFrame()
# 
#     ax_row, ax_col, trace_id = _parse_tracking_id(id)
# 
#     # Get barh_df from tracked data
#     barh_df = tracked_dict.get("barh_df")
#     if barh_df is not None and isinstance(barh_df, pd.DataFrame):
#         result = barh_df.copy()
#         renamed = {}
#         # Map 'y' and 'width' to standardized column names
#         for col in result.columns:
#             if col == "y":
#                 renamed[col] = get_csv_column_name("y", ax_row, ax_col, trace_id=trace_id)
#             elif col == "width":
#                 renamed[col] = get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id)
#             else:
#                 renamed[col] = get_csv_column_name(col, ax_row, ax_col, trace_id=trace_id)
# 
#         result = result.rename(columns=renamed)
# 
#         # Add xerr if present in kwargs
#         xerr = kwargs.get("xerr") if kwargs else None
#         if xerr is not None:
#             try:
#                 xerr_array = np.asarray(xerr)
#                 if len(xerr_array) == len(result):
#                     col_xerr = get_csv_column_name("xerr", ax_row, ax_col, trace_id=trace_id)
#                     result[col_xerr] = xerr_array
#             except (TypeError, ValueError):
#                 pass
# 
#         return result
# 
#     # Fallback to args if barh_df not found
#     args = tracked_dict.get("args", [])
#     if len(args) >= 2:
#         # Note: in barh, first arg is y positions, second is widths (x values)
#         col_y = get_csv_column_name("y", ax_row, ax_col, trace_id=trace_id)
#         col_x = get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id)
#         data = {col_y: args[0], col_x: args[1]}
# 
#         # Add xerr if present
#         xerr = kwargs.get("xerr") if kwargs else None
#         if xerr is not None:
#             try:
#                 xerr_array = np.asarray(xerr)
#                 col_xerr = get_csv_column_name("xerr", ax_row, ax_col, trace_id=trace_id)
#                 data[col_xerr] = xerr_array
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
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_stx_barh.py
# --------------------------------------------------------------------------------
