# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_scatter.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-18 18:14:26 (ywatanabe)"
# # File: /data/gpfs/projects/punim2354/ywatanabe/scitex_repo/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_scatter.py
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
# def _format_scatter(id, tracked_dict, kwargs):
#     """Format data from a scatter call (matplotlib ax.scatter or seaborn scatter).
# 
#     Note: For plot_scatter (wrapper method), use _format_plot_scatter instead.
#     This formatter expects data in args format: tracked_dict['args'] = (x, y).
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
#     # Extract x and y data if available
#     if len(args) >= 2:
#         x, y = args[0], args[1]
#         # Use structured column naming: ax-row-{row}-col-{col}_trace-id-{id}_variable-{var}
#         col_x = get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id)
#         col_y = get_csv_column_name("y", ax_row, ax_col, trace_id=trace_id)
#         df = pd.DataFrame({col_x: x, col_y: y})
#         return df
# 
#     return pd.DataFrame()

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_scatter.py
# --------------------------------------------------------------------------------
