# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_stx_conf_mat.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-18 18:14:26 (ywatanabe)"
# # File: /data/gpfs/projects/punim2354/ywatanabe/scitex_repo/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_plot_conf_mat.py
# # ----------------------------------------
# import os
# 
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# import numpy as np
# import pandas as pd
# from scitex.plt.utils._csv_column_naming import get_csv_column_name
# from ._format_plot import _parse_tracking_id
# 
# 
# def _format_plot_conf_mat(id, tracked_dict, kwargs):
#     """Format data from a stx_conf_mat call.
# 
#     Args:
#         id (str): Identifier for the plot
#         tracked_dict (dict): Dictionary containing tracked data
#         kwargs (dict): Keyword arguments passed to stx_conf_mat
# 
#     Returns:
#         pd.DataFrame: Formatted confusion matrix data
#     """
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
#     # Extract confusion matrix if available in args
#     if len(args) >= 1 and isinstance(args[0], (np.ndarray, list)):
#         conf_mat = np.array(args[0])
# 
#         # Convert to DataFrame
#         if conf_mat.ndim == 2:
#             # Create column and index names
#             n_classes = conf_mat.shape[0]
#             columns = [f"Predicted_{i}" for i in range(n_classes)]
#             index = [f"True_{i}" for i in range(n_classes)]
# 
#             # Create DataFrame with proper labels
#             df = pd.DataFrame(conf_mat, columns=columns, index=index)
# 
#             # Reset index to make it a regular column
#             df = df.reset_index().rename(columns={"index": "True_Class"})
# 
#             # Add prefix to all columns using single source of truth
#             df.columns = [
#                 get_csv_column_name(f"conf-mat-{col}", ax_row, ax_col, trace_id=trace_id)
#                 for col in df.columns
#             ]
# 
#             return df
# 
#     # Extract balanced accuracy if available as fallback
#     bacc = tracked_dict.get("balanced_accuracy")
# 
#     # Create DataFrame with the balanced accuracy
#     if bacc is not None:
#         col_name = get_csv_column_name(
#             "conf-mat-balanced-accuracy", ax_row, ax_col, trace_id=trace_id
#         )
#         df = pd.DataFrame({col_name: [bacc]})
#         return df
# 
#     return pd.DataFrame()

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_stx_conf_mat.py
# --------------------------------------------------------------------------------
