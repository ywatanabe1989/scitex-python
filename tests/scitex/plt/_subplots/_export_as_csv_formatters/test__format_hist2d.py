# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_hist2d.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-09 12:00:00 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_hist2d.py
# 
# import numpy as np
# import pandas as pd
# 
# from scitex.plt.utils._csv_column_naming import get_csv_column_name
# from ._format_plot import _parse_tracking_id
# 
# 
# def _format_hist2d(id, tracked_dict, kwargs):
#     """Format data from a 2D histogram call.
# 
#     Args:
#         id (str): Identifier for the plot
#         tracked_dict (dict): Dictionary containing tracked data
#         kwargs (dict): Keyword arguments passed to hist2d
# 
#     Returns:
#         pd.DataFrame: Formatted data from 2D histogram (input x, y data)
#     """
#     if not tracked_dict or not isinstance(tracked_dict, dict):
#         return pd.DataFrame()
# 
#     # Parse tracking ID to get axes position and trace ID
#     ax_row, ax_col, trace_id = _parse_tracking_id(id)
# 
#     if "args" in tracked_dict:
#         args = tracked_dict["args"]
#         if isinstance(args, tuple) and len(args) >= 2:
#             x = np.asarray(args[0]).flatten()
#             y = np.asarray(args[1]).flatten()
# 
#             # Ensure same length
#             min_len = min(len(x), len(y))
#             x = x[:min_len]
#             y = y[:min_len]
# 
#             # Get column names from single source of truth
#             col_x = get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id)
#             col_y = get_csv_column_name("y", ax_row, ax_col, trace_id=trace_id)
# 
#             df = pd.DataFrame({col_x: x, col_y: y})
#             return df
# 
#     return pd.DataFrame()
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_hist2d.py
# --------------------------------------------------------------------------------
