# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_sns_swarmplot.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-10 02:30:00 (ywatanabe)"
# # File: ./src/scitex/plt/_subplots/_export_as_csv_formatters/_format_sns_swarmplot.py
# 
# """CSV formatter for sns.swarmplot() calls - uses standard column naming."""
# 
# import pandas as pd
# 
# from scitex.plt.utils._csv_column_naming import get_csv_column_name
# 
# from ._format_plot import _parse_tracking_id
# 
# 
# def _format_sns_swarmplot(id, tracked_dict, kwargs):
#     """Format data from a sns_swarmplot call.
# 
#     Uses standard column naming: ax-row-{r}-col-{c}_trace-id-{id}_variable-{var}
# 
#     Args:
#         id (str): Identifier for the plot
#         tracked_dict (dict): Dictionary containing tracked data
#         kwargs (dict): Keyword arguments passed to sns_swarmplot
# 
#     Returns:
#         pd.DataFrame: Formatted data with standard column names
#     """
#     # Check if tracked_dict is empty or not a dictionary
#     if not tracked_dict or not isinstance(tracked_dict, dict):
#         return pd.DataFrame()
# 
#     # Parse tracking ID to get axes position
#     ax_row, ax_col, trace_id = _parse_tracking_id(id)
# 
#     # If 'data' key is in tracked_dict, use it
#     if "data" in tracked_dict:
#         data = tracked_dict["data"]
# 
#         if isinstance(data, pd.DataFrame):
#             result = pd.DataFrame()
# 
#             # Extract variables from kwargs
#             x_var = kwargs.get("x") if kwargs else None
#             y_var = kwargs.get("y") if kwargs else None
#             hue_var = kwargs.get("hue") if kwargs else None
# 
#             # Add x variable if specified
#             if x_var and x_var in data.columns:
#                 result[get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id)] = data[x_var]
# 
#             # Add y variable if specified
#             if y_var and y_var in data.columns:
#                 result[get_csv_column_name("y", ax_row, ax_col, trace_id=trace_id)] = data[y_var]
# 
#             # Add grouping variable if present
#             if hue_var and hue_var in data.columns:
#                 result[get_csv_column_name("hue", ax_row, ax_col, trace_id=trace_id)] = data[hue_var]
# 
#             # If we've added columns, return the result
#             if not result.empty:
#                 return result
# 
#             # If no columns were explicitly specified, return all columns
#             for col in data.columns:
#                 col_name = get_csv_column_name(f"data-{col}", ax_row, ax_col, trace_id=trace_id)
#                 result[col_name] = data[col]
#             return result
# 
#     # Legacy handling for args
#     if "args" in tracked_dict and len(tracked_dict["args"]) >= 1:
#         data = tracked_dict["args"][0]
# 
#         if isinstance(data, pd.DataFrame):
#             result = pd.DataFrame()
#             for col in data.columns:
#                 col_name = get_csv_column_name(f"data-{col}", ax_row, ax_col, trace_id=trace_id)
#                 result[col_name] = data[col]
#             return result
# 
#     return pd.DataFrame()

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_sns_swarmplot.py
# --------------------------------------------------------------------------------
