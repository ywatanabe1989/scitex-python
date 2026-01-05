# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_sns_scatterplot.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-10 02:30:00 (ywatanabe)"
# # File: ./src/scitex/plt/_subplots/_export_as_csv_formatters/_format_sns_scatterplot.py
# 
# """CSV formatter for sns.scatterplot() calls - uses standard column naming."""
# 
# import numpy as np
# import pandas as pd
# 
# from scitex.plt.utils._csv_column_naming import get_csv_column_name
# 
# from ._format_plot import _parse_tracking_id
# 
# 
# def _format_sns_scatterplot(id, tracked_dict, kwargs=None):
#     """Format data from a sns_scatterplot call.
# 
#     Uses standard column naming: ax-row-{r}-col-{c}_trace-id-{id}_variable-{var}
# 
#     Args:
#         id (str): Identifier for the plot
#         tracked_dict (dict): Tracked data dictionary
#         kwargs (dict): Keyword arguments from the record tuple
# 
#     Returns:
#         pd.DataFrame: Formatted data with standard column names
#     """
#     # Parse tracking ID to get axes position
#     ax_row, ax_col, trace_id = _parse_tracking_id(id)
# 
#     # Look for the DataFrame in the kwargs dictionary if provided
#     if kwargs and isinstance(kwargs, dict) and "data" in kwargs:
#         data = kwargs["data"]
#         if isinstance(data, pd.DataFrame):
#             result = pd.DataFrame()
# 
#             # If x and y variables are specified in kwargs, use them
#             x_var = kwargs.get("x")
#             y_var = kwargs.get("y")
# 
#             if x_var and y_var and x_var in data.columns and y_var in data.columns:
#                 result[get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id)] = data[x_var]
#                 result[get_csv_column_name("y", ax_row, ax_col, trace_id=trace_id)] = data[y_var]
# 
#                 # Also extract hue, size, style if specified
#                 for extra_var in ["hue", "size", "style"]:
#                     var_name = kwargs.get(extra_var)
#                     if var_name and var_name in data.columns:
#                         result[get_csv_column_name(extra_var, ax_row, ax_col, trace_id=trace_id)] = data[var_name]
# 
#                 return result
#             else:
#                 # If columns aren't specified, include all columns
#                 for col in data.columns:
#                     col_name = get_csv_column_name(f"data-{col}", ax_row, ax_col, trace_id=trace_id)
#                     result[col_name] = data[col]
#                 return result
# 
#     # Alternative: try to find a DataFrame in tracked_dict
#     if tracked_dict and isinstance(tracked_dict, dict):
#         if "data" in tracked_dict and isinstance(tracked_dict["data"], pd.DataFrame):
#             data = tracked_dict["data"]
#             result = pd.DataFrame()
# 
#             for col in data.columns:
#                 col_name = get_csv_column_name(f"data-{col}", ax_row, ax_col, trace_id=trace_id)
#                 result[col_name] = data[col]
# 
#             return result
# 
#     return pd.DataFrame()

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_sns_scatterplot.py
# --------------------------------------------------------------------------------
