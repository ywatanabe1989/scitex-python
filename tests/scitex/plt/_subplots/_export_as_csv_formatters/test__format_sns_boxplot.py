# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_sns_boxplot.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-10 02:30:00 (ywatanabe)"
# # File: ./src/scitex/plt/_subplots/_export_as_csv_formatters/_format_sns_boxplot.py
# 
# """CSV formatter for sns.boxplot() calls - uses standard column naming."""
# 
# import numpy as np
# import pandas as pd
# 
# from scitex.plt.utils._csv_column_naming import get_csv_column_name
# 
# from ._format_plot import _parse_tracking_id
# 
# 
# def _format_sns_boxplot(id, tracked_dict, kwargs):
#     """Format data from a sns_boxplot call.
# 
#     Uses standard column naming: ax-row-{r}-col-{c}_trace-id-{id}_variable-{var}
# 
#     Args:
#         id (str): Identifier for the plot
#         tracked_dict (dict): Dictionary containing tracked data
#         kwargs (dict): Keyword arguments passed to sns_boxplot
# 
#     Returns:
#         pd.DataFrame: Formatted boxplot data with standard column names
#     """
#     # Check if tracked_dict is empty or not a dictionary
#     if not tracked_dict:
#         return pd.DataFrame()
# 
#     # Parse tracking ID to get axes position
#     ax_row, ax_col, trace_id = _parse_tracking_id(id)
# 
#     # If tracked_dict is a dictionary, try to extract the data from it
#     if isinstance(tracked_dict, dict):
#         # First try to get 'data' key which is used in seaborn functions
#         if "data" in tracked_dict:
#             data = tracked_dict["data"]
#             if isinstance(data, pd.DataFrame):
#                 result = pd.DataFrame()
#                 for col in data.columns:
#                     col_name = get_csv_column_name(
#                         f"data-{col}", ax_row, ax_col, trace_id=trace_id
#                     )
#                     result[col_name] = data[col]
#                 return result
# 
#         # If no 'data' key, try to get data from args
#         args = tracked_dict.get("args", [])
#         if len(args) > 0:
#             data = args[0]
#             if isinstance(data, pd.DataFrame):
#                 result = pd.DataFrame()
#                 for col in data.columns:
#                     col_name = get_csv_column_name(
#                         f"data-{col}", ax_row, ax_col, trace_id=trace_id
#                     )
#                     result[col_name] = data[col]
#                 return result
# 
#             # Handle list or array data
#             elif isinstance(data, (list, np.ndarray)):
#                 try:
#                     if all(isinstance(item, (list, np.ndarray)) for item in data):
#                         result = pd.DataFrame()
#                         for i, group_data in enumerate(data):
#                             col_name = get_csv_column_name(
#                                 f"data-{i}", ax_row, ax_col, trace_id=trace_id
#                             )
#                             result[col_name] = pd.Series(group_data)
#                         return result
#                     else:
#                         col_name = get_csv_column_name(
#                             "data", ax_row, ax_col, trace_id=trace_id
#                         )
#                         return pd.DataFrame({col_name: data})
#                 except Exception:
#                     pass
# 
#     # If tracked_dict is a DataFrame already, use it directly
#     elif isinstance(tracked_dict, pd.DataFrame):
#         result = pd.DataFrame()
#         for col in tracked_dict.columns:
#             col_name = get_csv_column_name(
#                 f"data-{col}", ax_row, ax_col, trace_id=trace_id
#             )
#             result[col_name] = tracked_dict[col]
#         return result
# 
#     # If tracked_dict is list-like, try to convert it to a DataFrame
#     elif hasattr(tracked_dict, "__iter__") and not isinstance(tracked_dict, str):
#         try:
#             if all(isinstance(item, (list, np.ndarray)) for item in tracked_dict):
#                 result = pd.DataFrame()
#                 for i, group_data in enumerate(tracked_dict):
#                     col_name = get_csv_column_name(
#                         f"data-{i}", ax_row, ax_col, trace_id=trace_id
#                     )
#                     result[col_name] = pd.Series(group_data)
#                 return result
#             else:
#                 col_name = get_csv_column_name(
#                     "data", ax_row, ax_col, trace_id=trace_id
#                 )
#                 return pd.DataFrame({col_name: tracked_dict})
#         except Exception:
#             pass
# 
#     # Return empty DataFrame if we couldn't extract useful data
#     return pd.DataFrame()

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_sns_boxplot.py
# --------------------------------------------------------------------------------
