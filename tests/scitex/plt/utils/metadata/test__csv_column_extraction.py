# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/metadata/_csv_column_extraction.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: scitex/plt/utils/metadata/_csv_column_extraction.py
# 
# """
# CSV column name extraction from plot history.
# 
# This module provides functions to extract CSV column names from scitex history
# for linking JSON metadata to CSV data files.
# """
# 
# from typing import List
# 
# 
# def _extract_csv_columns_from_history(ax) -> list:
#     """
#     Extract CSV column names from scitex history for all plot types.
# 
#     This function generates the exact column names that will be produced
#     by export_as_csv(), providing a mapping between JSON metadata and CSV data.
# 
#     Parameters
#     ----------
#     ax : AxisWrapper or matplotlib.axes.Axes
#         The axes to extract CSV column info from
# 
#     Returns
#     -------
#     list
#         List of dictionaries containing CSV column mappings for each tracked plot,
#         e.g., [{"id": "boxplot_0", "method": "boxplot", "columns": ["ax_00_boxplot_0_boxplot_0", "ax_00_boxplot_0_boxplot_1"]}]
#     """
#     # Get axes position for CSV column naming
#     ax_row, ax_col = 0, 0
#     if hasattr(ax, "_scitex_metadata") and "position_in_grid" in ax._scitex_metadata:
#         pos = ax._scitex_metadata["position_in_grid"]
#         ax_row, ax_col = pos[0], pos[1]
# 
#     csv_columns_list = []
# 
#     # Check if we have scitex history
#     if not hasattr(ax, "history") or len(ax.history) == 0:
#         return csv_columns_list
# 
#     # Iterate through history to extract column names
#     for trace_index, (record_id, record) in enumerate(ax.history.items()):
#         if not isinstance(record, tuple) or len(record) < 4:
#             continue
# 
#         id_val, method, tracked_dict, kwargs = record
# 
#         # Generate column names
#         columns = _get_csv_columns_for_method_with_index(
#             id_val, method, tracked_dict, kwargs, ax_row, ax_col, trace_index
#         )
# 
#         if columns:
#             csv_columns_list.append({
#                 "id": id_val,
#                 "method": method,
#                 "columns": columns,
#             })
# 
#     return csv_columns_list
# 
# 
# def _get_csv_columns_for_method_with_index(
#     id_val, method, tracked_dict, kwargs, ax_row: int, ax_col: int, trace_index: int
# ) -> list:
#     """
#     Get CSV column names for a specific plotting method using trace index.
# 
#     This function uses the same naming convention as _extract_traces to ensure
#     consistency between plot.traces.csv_columns and data.columns.
# 
#     Parameters
#     ----------
#     id_val : str
#         The plot ID (e.g., "sine", "cosine")
#     method : str
#         The plotting method name (e.g., "plot", "scatter")
#     tracked_dict : dict
#         The tracked data dictionary
#     kwargs : dict
#         The keyword arguments passed to the plot
#     ax_row : int
#         Row index of axes in grid
#     ax_col : int
#         Column index of axes in grid
#     trace_index : int
#         Index of this trace (for deduplication)
# 
#     Returns
#     -------
#     list
#         List of column names that will be in the CSV
#     """
#     from .._csv_column_naming import get_csv_column_name
# 
#     columns = []
# 
#     # Use simplified variable names (x, y, bins, counts, etc.)
#     if method in ("plot", "stx_line"):
#         columns = [
#             get_csv_column_name("x", ax_row, ax_col, trace_index=trace_index),
#             get_csv_column_name("y", ax_row, ax_col, trace_index=trace_index),
#         ]
#     elif method in ("scatter", "plot_scatter"):
#         columns = [
#             get_csv_column_name("x", ax_row, ax_col, trace_index=trace_index),
#             get_csv_column_name("y", ax_row, ax_col, trace_index=trace_index),
#         ]
#     elif method in ("bar", "barh"):
#         columns = [
#             get_csv_column_name("x", ax_row, ax_col, trace_index=trace_index),
#             get_csv_column_name("height", ax_row, ax_col, trace_index=trace_index),
#         ]
#     elif method == "hist":
#         columns = [
#             get_csv_column_name("bins", ax_row, ax_col, trace_index=trace_index),
#             get_csv_column_name("counts", ax_row, ax_col, trace_index=trace_index),
#         ]
#     elif method in ("boxplot", "stx_box"):
#         columns = [
#             get_csv_column_name("data", ax_row, ax_col, trace_index=trace_index),
#         ]
#     elif method in ("violinplot", "stx_violin"):
#         columns = [
#             get_csv_column_name("data", ax_row, ax_col, trace_index=trace_index),
#         ]
#     elif method == "errorbar":
#         columns = [
#             get_csv_column_name("x", ax_row, ax_col, trace_index=trace_index),
#             get_csv_column_name("y", ax_row, ax_col, trace_index=trace_index),
#             get_csv_column_name("yerr", ax_row, ax_col, trace_index=trace_index),
#         ]
#     elif method == "fill_between":
#         columns = [
#             get_csv_column_name("x", ax_row, ax_col, trace_index=trace_index),
#             get_csv_column_name("y1", ax_row, ax_col, trace_index=trace_index),
#             get_csv_column_name("y2", ax_row, ax_col, trace_index=trace_index),
#         ]
#     elif method in ("imshow", "stx_heatmap", "stx_image"):
#         columns = [
#             get_csv_column_name("data", ax_row, ax_col, trace_index=trace_index),
#         ]
#     elif method in ("stx_kde", "stx_ecdf"):
#         columns = [
#             get_csv_column_name("x", ax_row, ax_col, trace_index=trace_index),
#             get_csv_column_name("y", ax_row, ax_col, trace_index=trace_index),
#         ]
#     elif method in ("stx_mean_std", "stx_mean_ci", "stx_median_iqr", "stx_shaded_line"):
#         columns = [
#             get_csv_column_name("x", ax_row, ax_col, trace_index=trace_index),
#             get_csv_column_name("y", ax_row, ax_col, trace_index=trace_index),
#             get_csv_column_name("lower", ax_row, ax_col, trace_index=trace_index),
#             get_csv_column_name("upper", ax_row, ax_col, trace_index=trace_index),
#         ]
#     elif method.startswith("sns_"):
#         sns_type = method.replace("sns_", "")
#         if sns_type in ("boxplot", "violinplot"):
#             columns = [
#                 get_csv_column_name("data", ax_row, ax_col, trace_index=trace_index),
#             ]
#         elif sns_type in ("scatterplot", "lineplot"):
#             columns = [
#                 get_csv_column_name("x", ax_row, ax_col, trace_index=trace_index),
#                 get_csv_column_name("y", ax_row, ax_col, trace_index=trace_index),
#             ]
#         elif sns_type == "barplot":
#             columns = [
#                 get_csv_column_name("x", ax_row, ax_col, trace_index=trace_index),
#                 get_csv_column_name("y", ax_row, ax_col, trace_index=trace_index),
#             ]
#         elif sns_type == "histplot":
#             columns = [
#                 get_csv_column_name("bins", ax_row, ax_col, trace_index=trace_index),
#                 get_csv_column_name("counts", ax_row, ax_col, trace_index=trace_index),
#             ]
#         elif sns_type == "kdeplot":
#             columns = [
#                 get_csv_column_name("x", ax_row, ax_col, trace_index=trace_index),
#                 get_csv_column_name("y", ax_row, ax_col, trace_index=trace_index),
#             ]
# 
#     return columns

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/metadata/_csv_column_extraction.py
# --------------------------------------------------------------------------------
