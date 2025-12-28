# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_fill_between.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-09 12:00:00 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_fill_between.py
# 
# import pandas as pd
# 
# from scitex.plt.utils._csv_column_naming import get_csv_column_name
# from ._format_plot import _parse_tracking_id
# 
# 
# def _format_fill_between(id, tracked_dict, kwargs):
#     """Format data from a fill_between call.
# 
#     Args:
#         id (str): Identifier for the plot
#         tracked_dict (dict): Dictionary containing tracked data
#         kwargs (dict): Keyword arguments passed to fill_between
# 
#     Returns:
#         pd.DataFrame: Formatted data from fill_between plot
#     """
#     if not tracked_dict or not isinstance(tracked_dict, dict):
#         return pd.DataFrame()
# 
#     # Parse tracking ID to get axes position and trace ID
#     ax_row, ax_col, trace_id = _parse_tracking_id(id)
# 
#     args = tracked_dict.get("args", [])
# 
#     # Typical args: x, y1, y2
#     if len(args) >= 3:
#         x, y1, y2 = args[:3]
# 
#         # Get column names from single source of truth
#         col_x = get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id)
#         col_y1 = get_csv_column_name("y1", ax_row, ax_col, trace_id=trace_id)
#         col_y2 = get_csv_column_name("y2", ax_row, ax_col, trace_id=trace_id)
# 
#         df = pd.DataFrame({col_x: x, col_y1: y1, col_y2: y2})
#         return df
# 
#     return pd.DataFrame()
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_fill_between.py
# --------------------------------------------------------------------------------
