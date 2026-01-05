# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_stackplot.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-21 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_stackplot.py
# 
# import numpy as np
# import pandas as pd
# 
# from scitex.plt.utils._csv_column_naming import get_csv_column_name
# from ._format_plot import _parse_tracking_id
# 
# 
# def _format_stackplot(id, tracked_dict, kwargs):
#     """Format data from a stackplot call.
# 
#     Args:
#         id (str): Identifier for the plot
#         tracked_dict (dict): Dictionary containing tracked data
#         kwargs (dict): Keyword arguments passed to stackplot
# 
#     Returns:
#         pd.DataFrame: Formatted data from stackplot (x and multiple y columns)
#     """
#     if not tracked_dict or not isinstance(tracked_dict, dict):
#         return pd.DataFrame()
# 
#     # Parse tracking ID to get axes position and trace ID
#     ax_row, ax_col, trace_id = _parse_tracking_id(id)
# 
#     args = tracked_dict.get("args", ())
# 
#     # stackplot(x, y1, y2, y3, ...) or stackplot(x, [y1, y2, y3], ...)
#     if len(args) < 2:
#         return pd.DataFrame()
# 
#     x = np.asarray(args[0])
#     col_x = get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id)
#     data = {col_x: x}
# 
#     # Get labels from kwargs if available
#     labels = kwargs.get("labels", [])
# 
#     # Handle remaining args as y arrays
#     y_arrays = args[1:]
# 
#     # If first y arg is a 2D array, treat rows as separate series
#     if len(y_arrays) == 1 and hasattr(y_arrays[0], "ndim"):
#         y_data = np.asarray(y_arrays[0])
#         if y_data.ndim == 2:
#             y_arrays = [y_data[i] for i in range(y_data.shape[0])]
# 
#     for i, y in enumerate(y_arrays):
#         y = np.asarray(y)
#         # Use label if available, otherwise use index
#         label = labels[i] if i < len(labels) else f"y{i:02d}"
#         col_y = get_csv_column_name(label, ax_row, ax_col, trace_id=trace_id)
#         data[col_y] = y
# 
#     return pd.DataFrame(data)
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_stackplot.py
# --------------------------------------------------------------------------------
