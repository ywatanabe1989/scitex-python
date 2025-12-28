# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_annotate.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-04 02:30:00 (ywatanabe)"
# # File: /data/gpfs/projects/punim2354/ywatanabe/SciTeX-Code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_annotate.py
# # ----------------------------------------
# from __future__ import annotations
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
# def _format_annotate(id, tracked_dict, kwargs):
#     """Format data from an annotate call.
# 
#     matplotlib annotate signature: annotate(text, xy, xytext=None, **kwargs)
#     - text: The text of the annotation
#     - xy: The point (x, y) to annotate
#     - xytext: The position (x, y) to place the text at (optional)
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
#     # Extract text and xy coordinates if available
#     if len(args) >= 2:
#         text_content = args[0]
#         xy = args[1]
# 
#         # xy should be a tuple (x, y)
#         if hasattr(xy, "__len__") and len(xy) >= 2:
#             x, y = xy[0], xy[1]
#         else:
#             return pd.DataFrame()
# 
#         data = {
#             get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id): [x],
#             get_csv_column_name("y", ax_row, ax_col, trace_id=trace_id): [y],
#             get_csv_column_name("content", ax_row, ax_col, trace_id=trace_id): [text_content],
#         }
# 
#         # Check if xytext was provided (either as third arg or in kwargs)
#         xytext = None
#         if len(args) >= 3:
#             xytext = args[2]
#         elif "xytext" in kwargs:
#             xytext = kwargs["xytext"]
# 
#         if xytext is not None and hasattr(xytext, "__len__") and len(xytext) >= 2:
#             data[get_csv_column_name("text_x", ax_row, ax_col, trace_id=trace_id)] = [xytext[0]]
#             data[get_csv_column_name("text_y", ax_row, ax_col, trace_id=trace_id)] = [xytext[1]]
# 
#         # Create DataFrame with proper column names (use dict with list values)
#         df = pd.DataFrame(data)
#         return df
# 
#     return pd.DataFrame()
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_annotate.py
# --------------------------------------------------------------------------------
