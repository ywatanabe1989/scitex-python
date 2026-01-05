# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_contour.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-09 12:00:00 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_contour.py
# 
# import numpy as np
# import pandas as pd
# 
# from scitex.plt.utils._csv_column_naming import get_csv_column_name
# from ._format_plot import _parse_tracking_id
# 
# 
# def _format_contour(id, tracked_dict, kwargs):
#     """Format data from a contour call.
# 
#     Args:
#         id (str): Identifier for the plot
#         tracked_dict (dict): Dictionary containing tracked data
#         kwargs (dict): Keyword arguments passed to contour
# 
#     Returns:
#         pd.DataFrame: Formatted data from contour plot (flattened X, Y, Z grids)
#     """
#     if not tracked_dict or not isinstance(tracked_dict, dict):
#         return pd.DataFrame()
# 
#     # Parse tracking ID to get axes position and trace ID
#     ax_row, ax_col, trace_id = _parse_tracking_id(id)
# 
#     args = tracked_dict.get("args", [])
# 
#     # Typical args: X, Y, Z where X and Y are 2D coordinate arrays and Z is the height array
#     if len(args) >= 3:
#         X, Y, Z = args[:3]
#         X_flat = np.asarray(X).flatten()
#         Y_flat = np.asarray(Y).flatten()
#         Z_flat = np.asarray(Z).flatten()
# 
#         # Get column names from single source of truth
#         col_x = get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id)
#         col_y = get_csv_column_name("y", ax_row, ax_col, trace_id=trace_id)
#         col_z = get_csv_column_name("z", ax_row, ax_col, trace_id=trace_id)
# 
#         df = pd.DataFrame({col_x: X_flat, col_y: Y_flat, col_z: Z_flat})
#         return df
# 
#     return pd.DataFrame()
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_contour.py
# --------------------------------------------------------------------------------
