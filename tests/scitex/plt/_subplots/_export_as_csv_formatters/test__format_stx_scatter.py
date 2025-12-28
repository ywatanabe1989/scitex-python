# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_stx_scatter.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """CSV formatter for stx_scatter() calls."""
# 
# import pandas as pd
# from scitex.plt.utils._csv_column_naming import get_csv_column_name
# from ._format_plot import _parse_tracking_id
# 
# 
# def _format_stx_scatter(id, tracked_dict, kwargs):
#     """Format data from stx_scatter call for CSV export.
# 
#     Parameters
#     ----------
#     id : str
#         Tracking identifier
#     tracked_dict : dict
#         Dictionary containing tracked data with 'scatter_df' key
#     kwargs : dict
#         Additional keyword arguments (unused)
# 
#     Returns
#     -------
#     pd.DataFrame
#         Formatted scatter data with standardized column names
#     """
#     if not tracked_dict or not isinstance(tracked_dict, dict):
#         return pd.DataFrame()
# 
#     ax_row, ax_col, trace_id = _parse_tracking_id(id)
# 
#     # Get scatter_df from tracked data
#     scatter_df = tracked_dict.get("scatter_df")
#     if scatter_df is not None and isinstance(scatter_df, pd.DataFrame):
#         result = scatter_df.copy()
#         renamed = {}
#         for col in result.columns:
#             renamed[col] = get_csv_column_name(col, ax_row, ax_col, trace_id=trace_id)
#         return result.rename(columns=renamed)
# 
#     # Fallback to args if scatter_df not found
#     args = tracked_dict.get("args", [])
#     if len(args) >= 2:
#         col_x = get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id)
#         col_y = get_csv_column_name("y", ax_row, ax_col, trace_id=trace_id)
#         return pd.DataFrame({col_x: args[0], col_y: args[1]})
# 
#     return pd.DataFrame()
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_stx_scatter.py
# --------------------------------------------------------------------------------
