# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_boxplot.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-09 12:00:00 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_boxplot.py
# 
# import numpy as np
# import pandas as pd
# 
# from scitex.plt.utils._csv_column_naming import get_csv_column_name
# from ._format_plot import _parse_tracking_id
# 
# 
# def _format_boxplot(id, tracked_dict, kwargs):
#     """Format data from a boxplot call.
# 
#     Args:
#         id (str): Identifier for the plot
#         tracked_dict (dict): Dictionary containing tracked data
#         kwargs (dict): Keyword arguments passed to boxplot
# 
#     Returns:
#         pd.DataFrame: Formatted data from boxplot
#     """
#     if not tracked_dict or not isinstance(tracked_dict, dict):
#         return pd.DataFrame()
# 
#     # Parse tracking ID to get axes position and trace ID
#     ax_row, ax_col, trace_id = _parse_tracking_id(id)
# 
#     args = tracked_dict.get("args", [])
#     call_kwargs = tracked_dict.get("kwargs", {})
# 
#     # Get labels if provided (for consistent naming with stats)
#     labels = call_kwargs.get("labels", None)
# 
#     if len(args) >= 1:
#         x = args[0]
# 
#         # One box plot
#         from scitex.types import is_listed_X as scitex_types_is_listed_X
# 
#         if isinstance(x, np.ndarray) or scitex_types_is_listed_X(x, [float, int]):
#             df = pd.DataFrame(x)
#             # Use label if single box and labels provided
#             if labels and len(labels) == 1:
#                 col_name = get_csv_column_name(labels[0], ax_row, ax_col, trace_id=trace_id)
#             else:
#                 col_name = get_csv_column_name("data-0", ax_row, ax_col, trace_id=trace_id)
#             df.columns = [col_name]
#         else:
#             # Multiple boxes
#             import scitex.pd
# 
#             df = scitex.pd.force_df({i_x: _x for i_x, _x in enumerate(x)})
# 
#             # Use labels if provided, otherwise use numeric indices
#             if labels and len(labels) == len(df.columns):
#                 df.columns = [
#                     get_csv_column_name(label, ax_row, ax_col, trace_id=trace_id)
#                     for label in labels
#                 ]
#             else:
#                 df.columns = [
#                     get_csv_column_name(f"data-{col}", ax_row, ax_col, trace_id=trace_id)
#                     for col in range(len(df.columns))
#                 ]
# 
#         df = df.apply(lambda col: col.dropna().reset_index(drop=True))
#         return df
# 
#     return pd.DataFrame()
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_boxplot.py
# --------------------------------------------------------------------------------
