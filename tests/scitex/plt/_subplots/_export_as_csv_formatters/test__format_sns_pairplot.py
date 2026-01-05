# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_sns_pairplot.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-18 18:14:26 (ywatanabe)"
# # File: /data/gpfs/projects/punim2354/ywatanabe/scitex_repo/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_sns_pairplot.py
# # ----------------------------------------
# import os
# 
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# import pandas as pd
# 
# from scitex.plt.utils._csv_column_naming import get_csv_column_name
# from ._format_plot import _parse_tracking_id
# 
# 
# def _format_sns_pairplot(id, tracked_dict, kwargs):
#     """Format data from a sns_pairplot call."""
#     # Check if tracked_dict is empty or not a dictionary
#     if not tracked_dict or not isinstance(tracked_dict, dict):
#         return pd.DataFrame()
# 
#     # Parse tracking ID to extract axes position and trace ID
#     ax_row, ax_col, trace_id = _parse_tracking_id(id)
# 
#     # Get the args from tracked_dict
#     args = tracked_dict.get("args", [])
# 
#     # Grid of plots showing pairwise relationships
#     if len(args) >= 1:
#         data = args[0]
# 
#         # Handle DataFrame input
#         if isinstance(data, pd.DataFrame):
#             # For pairplot, just return the full DataFrame since it uses all variables
#             result = data.copy()
#             if id is not None:
#                 result.columns = [
#                     get_csv_column_name(f"pair_{col}", ax_row, ax_col, trace_id=trace_id)
#                     for col in result.columns
#                 ]
# 
#             # Add vars or hue columns if specified
#             vars_list = kwargs.get("vars")
#             if vars_list and all(var in data.columns for var in vars_list):
#                 # Keep only the specified columns
#                 result = pd.DataFrame(
#                     {
#                         get_csv_column_name(f"pair_{col}", ax_row, ax_col, trace_id=trace_id): data[col]
#                         for col in vars_list
#                     }
#                 )
# 
#             return result
# 
#     return pd.DataFrame()

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_sns_pairplot.py
# --------------------------------------------------------------------------------
