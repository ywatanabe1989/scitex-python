# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_stx_ecdf.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-18 18:14:26 (ywatanabe)"
# # File: /data/gpfs/projects/punim2354/ywatanabe/scitex_repo/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_plot_ecdf.py
# # ----------------------------------------
# import os
# import pandas as pd
# 
# from scitex.plt.utils._csv_column_naming import get_csv_column_name
# from ._format_plot import _parse_tracking_id
# 
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# 
# def _format_plot_ecdf(id, tracked_dict, kwargs):
#     """Format data from a stx_ecdf call.
# 
#     Args:
#         id (str): Identifier for the plot
#         tracked_dict (dict): Dictionary containing 'ecdf_df' key with ECDF data
#         kwargs (dict): Keyword arguments passed to stx_ecdf
# 
#     Returns:
#         pd.DataFrame: Formatted ECDF data
#     """
#     # Check if tracked_dict is empty or not a dictionary
#     if not tracked_dict or not isinstance(tracked_dict, dict):
#         return pd.DataFrame()
# 
#     # Get the ecdf_df from tracked_dict
#     ecdf_df = tracked_dict.get("ecdf_df")
# 
#     if ecdf_df is None or not isinstance(ecdf_df, pd.DataFrame):
#         return pd.DataFrame()
# 
#     # Create a copy to avoid modifying the original
#     result = ecdf_df.copy()
# 
#     # Add prefix to column names if ID is provided
#     if id is not None:
#         # Parse the tracking ID to get axes position and trace ID
#         ax_row, ax_col, trace_id = _parse_tracking_id(id)
# 
#         # Rename columns using single source of truth
#         renamed = {}
#         for col in result.columns:
#             # Use the original column name as the variable (e.g., "ecdf_value", "ecdf_prob")
#             renamed[col] = get_csv_column_name(
#                 f"ecdf_{col}", ax_row, ax_col, trace_id=trace_id
#             )
#         result = result.rename(columns=renamed)
# 
#     return result

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_stx_ecdf.py
# --------------------------------------------------------------------------------
