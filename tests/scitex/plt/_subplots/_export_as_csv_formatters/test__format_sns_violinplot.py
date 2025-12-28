# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_sns_violinplot.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-10 02:30:00 (ywatanabe)"
# # File: ./src/scitex/plt/_subplots/_export_as_csv_formatters/_format_sns_violinplot.py
# 
# """CSV formatter for sns.violinplot() calls - uses standard column naming."""
# 
# import numpy as np
# import pandas as pd
# 
# from scitex.plt.utils._csv_column_naming import get_csv_column_name
# 
# from ._format_plot import _parse_tracking_id
# 
# 
# def _format_sns_violinplot(id, tracked_dict, kwargs):
#     """Format data from a sns_violinplot call.
# 
#     Uses standard column naming: ax-row-{r}-col-{c}_trace-id-{id}_variable-{var}
# 
#     Args:
#         id (str): Identifier for the plot
#         tracked_dict (dict): Dictionary containing tracked data
#         kwargs (dict): Keyword arguments passed to sns_violinplot
# 
#     Returns:
#         pd.DataFrame: Formatted data with standard column names
#     """
#     # Check if tracked_dict is empty
#     if not tracked_dict:
#         return pd.DataFrame()
# 
#     # Parse tracking ID to get axes position
#     ax_row, ax_col, trace_id = _parse_tracking_id(id)
# 
#     def _format_dataframe(df):
#         result = pd.DataFrame()
#         for col in df.columns:
#             col_name = get_csv_column_name(f"data-{col}", ax_row, ax_col, trace_id=trace_id)
#             result[col_name] = df[col]
#         return result
# 
#     def _format_list_of_arrays(data):
#         result = pd.DataFrame()
#         for i, group_data in enumerate(data):
#             col_name = get_csv_column_name(f"data-{i}", ax_row, ax_col, trace_id=trace_id)
#             result[col_name] = pd.Series(group_data)
#         return result
# 
#     # If tracked_dict is a dictionary
#     if isinstance(tracked_dict, dict):
#         if "data" in tracked_dict:
#             data = tracked_dict["data"]
# 
#             if isinstance(data, pd.DataFrame):
#                 try:
#                     return _format_dataframe(data)
#                 except Exception:
#                     try:
#                         x_var = kwargs.get("x") if kwargs else None
#                         y_var = kwargs.get("y") if kwargs else None
# 
#                         if x_var and y_var and x_var in data.columns and y_var in data.columns:
#                             return pd.DataFrame({
#                                 get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id): data[x_var],
#                                 get_csv_column_name("y", ax_row, ax_col, trace_id=trace_id): data[y_var],
#                             })
#                         elif len(data.columns) > 0:
#                             first_col = data.columns[0]
#                             return pd.DataFrame({
#                                 get_csv_column_name("data", ax_row, ax_col, trace_id=trace_id): data[first_col]
#                             })
#                     except Exception:
#                         return pd.DataFrame()
# 
#             elif isinstance(data, (list, np.ndarray)):
#                 try:
#                     if isinstance(data, list) and len(data) > 0 and all(
#                         isinstance(item, (list, np.ndarray)) for item in data
#                     ):
#                         return _format_list_of_arrays(data)
#                     else:
#                         return pd.DataFrame({
#                             get_csv_column_name("data", ax_row, ax_col, trace_id=trace_id): data
#                         })
#                 except Exception:
#                     return pd.DataFrame()
# 
#         # Legacy handling for args
#         args = tracked_dict.get("args", [])
#         if len(args) > 0:
#             data = args[0]
# 
#             if isinstance(data, pd.DataFrame):
#                 return _format_dataframe(data)
# 
#             elif isinstance(data, (list, np.ndarray)):
#                 try:
#                     if all(isinstance(item, (list, np.ndarray)) for item in data):
#                         return _format_list_of_arrays(data)
#                     else:
#                         return pd.DataFrame({
#                             get_csv_column_name("data", ax_row, ax_col, trace_id=trace_id): data
#                         })
#                 except Exception:
#                     return pd.DataFrame()
# 
#     # If tracked_dict is a DataFrame directly
#     elif isinstance(tracked_dict, pd.DataFrame):
#         try:
#             return _format_dataframe(tracked_dict)
#         except Exception:
#             try:
#                 if len(tracked_dict.columns) > 0:
#                     first_col = tracked_dict.columns[0]
#                     return pd.DataFrame({
#                         get_csv_column_name("data", ax_row, ax_col, trace_id=trace_id): tracked_dict[first_col]
#                     })
#             except Exception:
#                 return pd.DataFrame()
# 
#     # If tracked_dict is a list or numpy array directly
#     elif isinstance(tracked_dict, (list, np.ndarray)):
#         try:
#             if all(isinstance(item, (list, np.ndarray)) for item in tracked_dict):
#                 return _format_list_of_arrays(tracked_dict)
#             else:
#                 return pd.DataFrame({
#                     get_csv_column_name("data", ax_row, ax_col, trace_id=trace_id): tracked_dict
#                 })
#         except Exception:
#             return pd.DataFrame()
# 
#     return pd.DataFrame()

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_sns_violinplot.py
# --------------------------------------------------------------------------------
