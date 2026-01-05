# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_imshow2d.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-13 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_imshow2d.py
# 
# import pandas as pd
# 
# from scitex.plt.utils._csv_column_naming import get_csv_column_name
# from ._format_plot import _parse_tracking_id
# 
# 
# def _format_imshow2d(id, tracked_dict, kwargs):
#     """Format data from an imshow2d call.
# 
#     Args:
#         id (str): Identifier for the plot
#         tracked_dict (dict): Dictionary containing tracked data
#         kwargs (dict): Keyword arguments passed to imshow2d
# 
#     Returns:
#         pd.DataFrame: Formatted data from imshow2d
#     """
#     # Check if tracked_dict is empty or not a dictionary
#     if not tracked_dict or not isinstance(tracked_dict, dict):
#         return pd.DataFrame()
# 
#     # Parse the tracking ID to get axes position and trace ID
#     ax_row, ax_col, trace_id = _parse_tracking_id(id)
# 
#     # Get the args from tracked_dict
#     args = tracked_dict.get("args", [])
# 
#     # Extract data if available
#     if len(args) >= 1 and isinstance(args[0], pd.DataFrame):
#         df = args[0].copy()
#         # Rename columns using the single source of truth
#         renamed_cols = {}
#         for col in df.columns:
#             renamed_cols[col] = get_csv_column_name(
#                 f"imshow2d_{col}", ax_row, ax_col, trace_id=trace_id
#             )
#         df = df.rename(columns=renamed_cols)
#         return df
# 
#     return pd.DataFrame()
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_imshow2d.py
# --------------------------------------------------------------------------------
