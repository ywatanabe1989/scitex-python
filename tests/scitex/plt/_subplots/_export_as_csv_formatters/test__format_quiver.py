# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_quiver.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-01 12:20:00 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_quiver.py
# 
# import numpy as np
# import pandas as pd
# 
# from scitex.plt.utils._csv_column_naming import get_csv_column_name
# from ._format_plot import _parse_tracking_id
# 
# 
# def _format_quiver(id, tracked_dict, kwargs):
#     """Format data from a quiver (vector field) call.
# 
#     Args:
#         id (str): Identifier for the plot
#         tracked_dict (dict): Dictionary containing tracked data
#         kwargs (dict): Keyword arguments passed to quiver
# 
#     Returns:
#         pd.DataFrame: Formatted data from quiver (X, Y positions and U, V vectors)
#     """
#     if not tracked_dict or not isinstance(tracked_dict, dict):
#         return pd.DataFrame()
# 
#     # Parse the tracking ID to get axes position and trace ID
#     ax_row, ax_col, trace_id = _parse_tracking_id(id)
# 
#     if "args" in tracked_dict:
#         args = tracked_dict["args"]
#         if isinstance(args, tuple):
#             # quiver can be called as:
#             # quiver(U, V) - positions auto-generated
#             # quiver(X, Y, U, V) - explicit positions
#             if len(args) == 2:
#                 U = np.asarray(args[0])
#                 V = np.asarray(args[1])
#                 X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]))
#             elif len(args) >= 4:
#                 X = np.asarray(args[0])
#                 Y = np.asarray(args[1])
#                 U = np.asarray(args[2])
#                 V = np.asarray(args[3])
#             else:
#                 return pd.DataFrame()
# 
#             df = pd.DataFrame(
#                 {
#                     get_csv_column_name("quiver-x", ax_row, ax_col, trace_id=trace_id): X.flatten(),
#                     get_csv_column_name("quiver-y", ax_row, ax_col, trace_id=trace_id): Y.flatten(),
#                     get_csv_column_name("quiver-u", ax_row, ax_col, trace_id=trace_id): U.flatten(),
#                     get_csv_column_name("quiver-v", ax_row, ax_col, trace_id=trace_id): V.flatten(),
#                 }
#             )
#             return df
# 
#     return pd.DataFrame()
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_quiver.py
# --------------------------------------------------------------------------------
