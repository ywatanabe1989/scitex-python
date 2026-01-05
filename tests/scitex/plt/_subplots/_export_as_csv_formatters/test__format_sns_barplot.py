# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_sns_barplot.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-10 02:30:00 (ywatanabe)"
# # File: ./src/scitex/plt/_subplots/_export_as_csv_formatters/_format_sns_barplot.py
# 
# """CSV formatter for sns.barplot() calls - uses standard column naming."""
# 
# import numpy as np
# import pandas as pd
# 
# from scitex.plt.utils._csv_column_naming import get_csv_column_name
# 
# from ._format_plot import _parse_tracking_id
# 
# 
# def _format_sns_barplot(id, tracked_dict, kwargs):
#     """Format data from a sns_barplot call.
# 
#     Uses standard column naming: ax-row-{r}-col-{c}_trace-id-{id}_variable-{var}
# 
#     Args:
#         id (str): Identifier for the plot
#         tracked_dict (dict): Dictionary containing tracked data
#         kwargs (dict): Keyword arguments passed to sns_barplot
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
#         df = tracked_dict["data"]
#         if isinstance(df, pd.DataFrame):
#             result = pd.DataFrame()
#             for col in df.columns:
#                 col_name = get_csv_column_name(f"data-{col}", ax_row, ax_col, trace_id=trace_id)
#                 result[col_name] = df[col]
#             return result
# 
#     # Legacy handling for args
#     if "args" in tracked_dict:
#         df = tracked_dict["args"]
#         if isinstance(df, pd.DataFrame):
#             try:
#                 processed_df = pd.DataFrame(
#                     pd.Series(np.array(df).diagonal(), index=df.columns)
#                 ).T
#                 result = pd.DataFrame()
#                 for col in processed_df.columns:
#                     col_name = get_csv_column_name(f"data-{col}", ax_row, ax_col, trace_id=trace_id)
#                     result[col_name] = processed_df[col]
#                 return result
#             except (ValueError, TypeError, IndexError):
#                 result = pd.DataFrame()
#                 for col in df.columns:
#                     col_name = get_csv_column_name(f"data-{col}", ax_row, ax_col, trace_id=trace_id)
#                     result[col_name] = df[col]
#                 return result
# 
#     return pd.DataFrame()

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_sns_barplot.py
# --------------------------------------------------------------------------------
