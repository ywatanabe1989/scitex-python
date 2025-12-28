# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_stx_image.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-18 18:14:26 (ywatanabe)"
# # File: /data/gpfs/projects/punim2354/ywatanabe/scitex_repo/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_plot_image.py
# # ----------------------------------------
# import os
# import numpy as np
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
# def _format_plot_image(id, tracked_dict, kwargs):
#     """Format data from a stx_image call.
# 
#     Exports image data in long-format xyz format for better compatibility.
#     Also saves channel data for RGB/RGBA images.
# 
#     Args:
#         id (str or int): Identifier for the plot
#         tracked_dict (dict): Dictionary containing tracked data
#         kwargs (dict): Keyword arguments passed to stx_image
# 
#     Returns:
#         pd.DataFrame: Formatted image data in xyz format
#     """
#     # Check if tracked_dict is not a dictionary or is empty
#     if not tracked_dict or not isinstance(tracked_dict, dict):
#         return pd.DataFrame()
# 
#     # Parse the tracking ID to get axes position and trace ID
#     ax_row, ax_col, trace_id = _parse_tracking_id(id)
# 
#     # Check if image_df is available and use it if present
#     if "image_df" in tracked_dict:
#         image_df = tracked_dict.get("image_df")
#         if isinstance(image_df, pd.DataFrame):
#             # Add prefix if ID is provided
#             if id is not None:
#                 image_df = image_df.copy()
#                 # Rename columns using single source of truth
#                 renamed = {}
#                 for col in image_df.columns:
#                     # Convert to string to handle integer column names
#                     col_str = str(col)
#                     renamed[col] = get_csv_column_name(
#                         col_str, ax_row, ax_col, trace_id=trace_id
#                     )
#                 image_df = image_df.rename(columns=renamed)
#             return image_df
# 
#     # If we have image data
#     if "image" in tracked_dict:
#         img = tracked_dict["image"]
# 
#         # Handle 2D grayscale images - create xyz format (x, y, value)
#         if isinstance(img, np.ndarray) and img.ndim == 2:
#             rows, cols = img.shape
#             row_indices, col_indices = np.meshgrid(
#                 range(rows), range(cols), indexing="ij"
#             )
# 
#             # Create xyz format using single source of truth
#             df = pd.DataFrame(
#                 {
#                     get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id): col_indices.flatten(),  # x is column
#                     get_csv_column_name("y", ax_row, ax_col, trace_id=trace_id): row_indices.flatten(),  # y is row
#                     get_csv_column_name("value", ax_row, ax_col, trace_id=trace_id): img.flatten(),  # z is intensity
#                 }
#             )
#             return df
# 
#         # Handle RGB/RGBA images - create xyz format with additional channel information
#         elif isinstance(img, np.ndarray) and img.ndim == 3:
#             rows, cols, channels = img.shape
# 
#             # Create a list to hold rows for a long-format DataFrame
#             data_rows = []
#             channel_names = ["r", "g", "b", "a"]
# 
#             # Get column names using single source of truth
#             x_col = get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id)
#             y_col = get_csv_column_name("y", ax_row, ax_col, trace_id=trace_id)
#             channel_col = get_csv_column_name("channel", ax_row, ax_col, trace_id=trace_id)
#             value_col = get_csv_column_name("value", ax_row, ax_col, trace_id=trace_id)
# 
#             # Create long-format data (x, y, channel, value)
#             for r in range(rows):
#                 for c in range(cols):
#                     for ch in range(min(channels, len(channel_names))):
#                         data_rows.append(
#                             {
#                                 x_col: c,  # x is column
#                                 y_col: r,  # y is row
#                                 channel_col: channel_names[ch],  # channel name
#                                 value_col: img[r, c, ch],  # channel value
#                             }
#                         )
# 
#             # Return long-format DataFrame
#             return pd.DataFrame(data_rows)
# 
#     # Skip CSV export if no suitable data format found
#     return pd.DataFrame()

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_stx_image.py
# --------------------------------------------------------------------------------
