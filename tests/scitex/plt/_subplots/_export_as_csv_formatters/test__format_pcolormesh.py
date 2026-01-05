# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_pcolormesh.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-21 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_pcolormesh.py
# 
# import numpy as np
# import pandas as pd
# 
# from scitex.plt.utils._csv_column_naming import get_csv_column_name
# from ._format_plot import _parse_tracking_id
# 
# 
# def _format_pcolormesh(id, tracked_dict, kwargs):
#     """Format data from a pcolormesh call.
# 
#     Args:
#         id (str): Identifier for the plot
#         tracked_dict (dict): Dictionary containing tracked data
#         kwargs (dict): Keyword arguments passed to pcolormesh
# 
#     Returns:
#         pd.DataFrame: Formatted data from pcolormesh (x, y, value columns)
#     """
#     if not tracked_dict or not isinstance(tracked_dict, dict):
#         return pd.DataFrame()
# 
#     # Parse tracking ID to get axes position and trace ID
#     ax_row, ax_col, trace_id = _parse_tracking_id(id)
# 
#     args = tracked_dict.get("args", ())
# 
#     if len(args) == 0:
#         return pd.DataFrame()
# 
#     # pcolormesh can be called as:
#     # pcolormesh(C) - just color values
#     # pcolormesh(X, Y, C) - with coordinates
#     if len(args) == 1:
#         # Just C provided
#         C = np.asarray(args[0])
#         rows, cols = C.shape
#         Y, X = np.meshgrid(range(rows), range(cols), indexing="ij")
#     elif len(args) >= 3:
#         # X, Y, C provided
#         X = np.asarray(args[0])
#         Y = np.asarray(args[1])
#         C = np.asarray(args[2])
#     else:
#         return pd.DataFrame()
# 
#     # Get column names
#     col_x = get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id)
#     col_y = get_csv_column_name("y", ax_row, ax_col, trace_id=trace_id)
#     col_value = get_csv_column_name("value", ax_row, ax_col, trace_id=trace_id)
# 
#     # Flatten for CSV format
#     df = pd.DataFrame({
#         col_x: X.flatten(),
#         col_y: Y.flatten(),
#         col_value: C.flatten(),
#     })
# 
#     return df
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_pcolormesh.py
# --------------------------------------------------------------------------------
