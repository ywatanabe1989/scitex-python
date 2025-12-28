# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_sns_lineplot.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-18 18:14:26 (ywatanabe)"
# # File: /data/gpfs/projects/punim2354/ywatanabe/scitex_repo/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_sns_lineplot.py
# # ----------------------------------------
# import os
# 
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# import numpy as np
# import pandas as pd
# 
# from scitex.plt.utils._csv_column_naming import get_csv_column_name
# from ._format_plot import _parse_tracking_id
# 
# 
# def _format_sns_lineplot(id, tracked_dict, kwargs):
#     """Format data from a sns_lineplot call."""
#     # Check if tracked_dict is empty or not a dictionary
#     if not tracked_dict or not isinstance(tracked_dict, dict):
#         return pd.DataFrame()
# 
#     # Parse the tracking ID to get axes position and trace ID
#     ax_row, ax_col, trace_id = _parse_tracking_id(id)
# 
#     # Get data from tracked_dict - can be in "data" (from _sns_base_xyhue) or "args"
#     data = tracked_dict.get("data")
#     args = tracked_dict.get("args", [])
# 
#     # If data is None, try to get it from args
#     if data is None and len(args) >= 1:
#         data = args[0]
# 
#     x_var = kwargs.get("x")
#     y_var = kwargs.get("y")
# 
#     # Handle DataFrame input with x, y variables
#     if isinstance(data, pd.DataFrame):
#         # If data has been pre-processed by _sns_prepare_xyhue, it may be pivoted
#         # Just export all columns with proper naming
#         if data.empty:
#             return pd.DataFrame()
# 
#         result = {}
#         for col in data.columns:
#             col_name = str(col) if not isinstance(col, str) else col
#             result[get_csv_column_name(col_name, ax_row, ax_col, trace_id=trace_id)] = data[col].values
#         return pd.DataFrame(result)
# 
#     # Handle direct x, y data arrays from args
#     elif (
#         len(args) > 1
#         and isinstance(args[0], (np.ndarray, list))
#         and isinstance(args[1], (np.ndarray, list))
#     ):
#         x_data, y_data = args[0], args[1]
#         return pd.DataFrame({
#             get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id): x_data,
#             get_csv_column_name("y", ax_row, ax_col, trace_id=trace_id): y_data
#         })
# 
#     return pd.DataFrame()

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_sns_lineplot.py
# --------------------------------------------------------------------------------
