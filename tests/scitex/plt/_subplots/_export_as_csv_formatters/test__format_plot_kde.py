# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_plot_kde.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-18 18:14:26 (ywatanabe)"
# # File: /data/gpfs/projects/punim2354/ywatanabe/scitex_repo/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_plot_kde.py
# # ----------------------------------------
# import os
# 
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# import pandas as pd
# from scitex.pd import force_df
# from scitex.plt.utils._csv_column_naming import get_csv_column_name
# from ._format_plot import _parse_tracking_id
# 
# 
# def _format_plot_kde(id, tracked_dict, kwargs):
#     """Format data from a stx_kde call.
# 
#     Processes kernel density estimation plot data.
# 
#     Args:
#         id (str): Identifier for the plot
#         tracked_dict (dict): Dictionary containing 'x', 'kde', and 'n' keys
#         kwargs (dict): Keyword arguments passed to stx_kde
# 
#     Returns:
#         pd.DataFrame: Formatted KDE data
#     """
#     # Check if tracked_dict is empty or not a dictionary
#     if not tracked_dict or not isinstance(tracked_dict, dict):
#         return pd.DataFrame()
# 
#     x = tracked_dict.get("x")
#     kde = tracked_dict.get("kde")
#     n = tracked_dict.get("n")
# 
#     if x is None or kde is None:
#         return pd.DataFrame()
# 
#     # Parse tracking ID to extract axes position and trace ID
#     ax_row, ax_col, trace_id = _parse_tracking_id(id)
# 
#     # Use standardized column naming
#     x_col = get_csv_column_name("kde_x", ax_row, ax_col, trace_id=trace_id)
#     density_col = get_csv_column_name("kde_density", ax_row, ax_col, trace_id=trace_id)
# 
#     df = pd.DataFrame({x_col: x, density_col: kde})
# 
#     # Add sample count if available
#     if n is not None:
#         # If n is a scalar, create a list with the same length as x
#         if not hasattr(n, "__len__"):
#             n = [n] * len(x)
#         n_col = get_csv_column_name("kde_n", ax_row, ax_col, trace_id=trace_id)
#         df[n_col] = n
# 
#     return df

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_plot_kde.py
# --------------------------------------------------------------------------------
