# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_stx_shaded_line.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-10 03:00:00 (ywatanabe)"
# # File: ./src/scitex/plt/_subplots/_export_as_csv_formatters/_format_stx_shaded_line.py
# 
# """CSV formatter for stx_shaded_line() calls - uses standard column naming."""
# 
# import pandas as pd
# 
# from scitex.plt.utils._csv_column_naming import get_csv_column_name
# 
# from ._format_plot import _parse_tracking_id
# 
# 
# def _format_plot_shaded_line(id, tracked_dict, kwargs):
#     """Format data from a stx_shaded_line call.
# 
#     Processes stx_shaded_line data for CSV export using standard column naming
#     (ax-row-{r}-col-{c}_trace-id-{id}_variable-{var}).
# 
#     Args:
#         id (str): Identifier for the plot
#         tracked_dict (dict): Dictionary containing tracked data
#         kwargs (dict): Keyword arguments passed to stx_shaded_line
# 
#     Returns:
#         pd.DataFrame: Formatted shaded line data with standard column names
#     """
#     # Check if tracked_dict is empty or not a dictionary
#     if not tracked_dict or not isinstance(tracked_dict, dict):
#         return pd.DataFrame()
# 
#     # Parse tracking ID to get axes position
#     ax_row, ax_col, trace_id = _parse_tracking_id(id)
# 
#     # If we have a plot_df from plotting methods, use that directly
#     if "plot_df" in tracked_dict and isinstance(tracked_dict["plot_df"], pd.DataFrame):
#         plot_df = tracked_dict["plot_df"]
#         # Rename columns using standard naming convention
#         renamed = {}
#         for col in plot_df.columns:
#             renamed[col] = get_csv_column_name(col, ax_row, ax_col, trace_id=trace_id)
#         return plot_df.rename(columns=renamed)
# 
#     # Try getting the individual components
#     x = tracked_dict.get("x")
#     y_middle = tracked_dict.get("y_middle")
#     y_lower = tracked_dict.get("y_lower")
#     y_upper = tracked_dict.get("y_upper")
# 
#     # If we have all necessary components
#     if x is not None and y_middle is not None and y_lower is not None:
#         x_col = get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id)
#         y_col = get_csv_column_name("y-middle", ax_row, ax_col, trace_id=trace_id)
#         lower_col = get_csv_column_name("y-lower", ax_row, ax_col, trace_id=trace_id)
#         upper_col = get_csv_column_name("y-upper", ax_row, ax_col, trace_id=trace_id)
# 
#         data = {
#             x_col: x,
#             y_col: y_middle,
#             lower_col: y_lower,
#         }
# 
#         if y_upper is not None:
#             data[upper_col] = y_upper
#         else:
#             # If only y_lower is provided, assume it's symmetric around y_middle
#             data[upper_col] = y_middle + (y_middle - y_lower)
# 
#         return pd.DataFrame(data)
# 
#     return pd.DataFrame()

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_stx_shaded_line.py
# --------------------------------------------------------------------------------
