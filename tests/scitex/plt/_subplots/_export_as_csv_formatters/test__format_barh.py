# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_barh.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-18 18:14:26 (ywatanabe)"
# # File: /data/gpfs/projects/punim2354/ywatanabe/scitex_repo/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_barh.py
# # ----------------------------------------
# import os
# 
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# import pandas as pd
# 
# from scitex.plt.utils._csv_column_naming import get_csv_column_name
# from ._format_plot import _parse_tracking_id
# 
# 
# def _format_barh(id, tracked_dict, kwargs):
#     """Format data from a barh call."""
#     # Check if tracked_dict is empty or not a dictionary
#     if not tracked_dict or not isinstance(tracked_dict, dict):
#         return pd.DataFrame()
# 
#     # Parse tracking ID to get axes position and trace ID
#     ax_row, ax_col, trace_id = _parse_tracking_id(id)
# 
#     # Get the args from tracked_dict
#     args = tracked_dict.get("args", [])
# 
#     # Extract x and y data if available
#     if len(args) >= 2:
#         # Note: in barh, first arg is y positions, second is widths (x values)
#         y_pos, x_width = args[0], args[1]
# 
#         # Get xerr from kwargs
#         xerr = kwargs.get("xerr")
# 
#         # Convert single values to Series
#         if isinstance(y_pos, (int, float)):
#             y_pos = pd.Series(y_pos, name="y")
#         if isinstance(x_width, (int, float)):
#             x_width = pd.Series(x_width, name="x")
#     else:
#         # Not enough arguments
#         return pd.DataFrame()
# 
#     # Use structured column naming: ax-row-{row}-col-{col}_trace-id-{id}_variable-{var}
#     col_x = get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id)
#     col_y = get_csv_column_name("y", ax_row, ax_col, trace_id=trace_id)
# 
#     df = pd.DataFrame({col_y: y_pos, col_x: x_width})
# 
#     if xerr is not None:
#         if isinstance(xerr, (int, float)):
#             xerr = pd.Series(xerr, name="xerr")
#         col_xerr = get_csv_column_name("xerr", ax_row, ax_col, trace_id=trace_id)
#         df[col_xerr] = xerr
#     return df

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_barh.py
# --------------------------------------------------------------------------------
