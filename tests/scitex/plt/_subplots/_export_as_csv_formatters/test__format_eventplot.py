# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_eventplot.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-18 18:14:26 (ywatanabe)"
# # File: /data/gpfs/projects/punim2354/ywatanabe/scitex_repo/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_eventplot.py
# # ----------------------------------------
# import os
# 
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# import numpy as np
# import pandas as pd
# import scitex
# 
# from scitex import logging
# from scitex.plt.utils._csv_column_naming import get_csv_column_name
# from ._format_plot import _parse_tracking_id
# 
# logger = logging.getLogger(__name__)
# 
# 
# def _format_eventplot(id, tracked_dict, kwargs):
#     """Format data from an eventplot call."""
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
#     # Eventplot displays multiple sets of events as parallel lines
#     if len(args) >= 1:
#         positions = args[0]
# 
#         try:
#             # Try using scitex.pd.force_df if available
#             try:
#                 import scitex.pd
# 
#                 # If positions is a single array
#                 if isinstance(positions, (list, np.ndarray)) and not isinstance(
#                     positions[0], (list, np.ndarray)
#                 ):
#                     col_name = get_csv_column_name("eventplot-events", ax_row, ax_col, trace_id=trace_id)
#                     return pd.DataFrame({col_name: positions})
# 
#                 # If positions is a list of arrays (multiple event sets)
#                 elif isinstance(positions, (list, np.ndarray)):
#                     data = {}
#                     for i, events in enumerate(positions):
#                         col_name = get_csv_column_name(f"eventplot-events{i:02d}", ax_row, ax_col, trace_id=f"{trace_id}-{i}")
#                         data[col_name] = events
# 
#                     # Use force_df to handle different length arrays
#                     return scitex.pd.force_df(data)
# 
#             except (ImportError, AttributeError):
#                 # Fall back to pandas with manual Series creation
#                 # If positions is a single array
#                 if isinstance(positions, (list, np.ndarray)) and not isinstance(
#                     positions[0], (list, np.ndarray)
#                 ):
#                     col_name = get_csv_column_name("eventplot-events", ax_row, ax_col, trace_id=trace_id)
#                     return pd.DataFrame({col_name: positions})
# 
#                 # If positions is a list of arrays (multiple event sets)
#                 elif isinstance(positions, (list, np.ndarray)):
#                     # Create a DataFrame where each column is a Series that can handle varying lengths
#                     df = pd.DataFrame()
#                     for i, events in enumerate(positions):
#                         col_name = get_csv_column_name(f"eventplot-events{i:02d}", ax_row, ax_col, trace_id=f"{trace_id}-{i}")
#                         df[col_name] = pd.Series(events)
#                     return df
#         except Exception as e:
#             # If all else fails, return an empty DataFrame
#             logger.warning(f"Error formatting eventplot data: {str(e)}")
#             return pd.DataFrame()
# 
#     return pd.DataFrame()

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_eventplot.py
# --------------------------------------------------------------------------------
