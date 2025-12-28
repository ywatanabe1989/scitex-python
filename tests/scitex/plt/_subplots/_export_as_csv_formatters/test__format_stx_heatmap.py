# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_stx_heatmap.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-18 18:14:26 (ywatanabe)"
# # File: /data/gpfs/projects/punim2354/ywatanabe/scitex_repo/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_plot_heatmap.py
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
# def _format_plot_heatmap(id, tracked_dict, kwargs):
#     """Format data from a stx_heatmap call.
# 
#     Exports heatmap data in xyz format (x, y, value) for better compatibility.
# 
#     Args:
#         id (str): Identifier for the plot
#         tracked_dict (dict): Dictionary containing tracked data
#         kwargs (dict): Keyword arguments passed to stx_heatmap
# 
#     Returns:
#         pd.DataFrame: Formatted heatmap data in xyz format
#     """
#     # Check if tracked_dict is empty or not a dictionary
#     if not tracked_dict or not isinstance(tracked_dict, dict):
#         return pd.DataFrame()
# 
#     # Extract data from tracked_dict
#     data = tracked_dict.get("data")
#     x_labels = tracked_dict.get("x_labels")
#     y_labels = tracked_dict.get("y_labels")
# 
#     if data is not None and hasattr(data, "shape") and len(data.shape) == 2:
#         rows, cols = data.shape
#         row_indices, col_indices = np.meshgrid(range(rows), range(cols), indexing="ij")
# 
#         # Parse the tracking ID to get axes position and trace ID
#         ax_row, ax_col, trace_id = _parse_tracking_id(id)
# 
#         # Format data in xyz format (x, y, value) using single source of truth
#         df = pd.DataFrame(
#             {
#                 get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id): col_indices.flatten(),  # x is column
#                 get_csv_column_name("y", ax_row, ax_col, trace_id=trace_id): row_indices.flatten(),  # y is row
#                 get_csv_column_name("value", ax_row, ax_col, trace_id=trace_id): data.flatten(),  # z is intensity/value
#             }
#         )
# 
#         # Add label information if available
#         if x_labels is not None and len(x_labels) == cols:
#             # Map column indices to x labels (columns are x)
#             x_label_map = {i: label for i, label in enumerate(x_labels)}
#             x_col_name = get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id)
#             x_label_col_name = get_csv_column_name("x_label", ax_row, ax_col, trace_id=trace_id)
#             df[x_label_col_name] = df[x_col_name].map(x_label_map)
# 
#         if y_labels is not None and len(y_labels) == rows:
#             # Map row indices to y labels (rows are y)
#             y_label_map = {i: label for i, label in enumerate(y_labels)}
#             y_col_name = get_csv_column_name("y", ax_row, ax_col, trace_id=trace_id)
#             y_label_col_name = get_csv_column_name("y_label", ax_row, ax_col, trace_id=trace_id)
#             df[y_label_col_name] = df[y_col_name].map(y_label_map)
# 
#         return df
# 
#     return pd.DataFrame()

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_stx_heatmap.py
# --------------------------------------------------------------------------------
