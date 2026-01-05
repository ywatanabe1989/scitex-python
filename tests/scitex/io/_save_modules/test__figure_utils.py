# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_save_modules/_figure_utils.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-19
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/io/_save_modules/_figure_utils.py
# 
# """Utility functions for extracting figure data for CSV export."""
# 
# 
# def get_figure_with_data(obj):
#     """
#     Extract figure or axes object that may contain plotting data for CSV export.
# 
#     Parameters
#     ----------
#     obj : various matplotlib objects
#         Could be Figure, Axes, FigWrapper, AxisWrapper, or other matplotlib objects
# 
#     Returns
#     -------
#     object or None
#         Figure or axes object that has export_as_csv methods, or None if not found
#     """
#     import matplotlib.axes
#     import matplotlib.figure
#     import matplotlib.pyplot as plt
# 
#     # Check if object already has export methods (SciTeX wrapped objects)
#     if hasattr(obj, "export_as_csv"):
#         return obj
# 
#     # Handle matplotlib Figure objects
#     if isinstance(obj, matplotlib.figure.Figure):
#         # Get the current axes that might be wrapped with SciTeX functionality
#         current_ax = plt.gca()
#         if hasattr(current_ax, "export_as_csv"):
#             return current_ax
# 
#         # Check all axes in the figure
#         for ax in obj.axes:
#             if hasattr(ax, "export_as_csv"):
#                 return ax
# 
#         return None
# 
#     # Handle matplotlib Axes objects
#     if isinstance(obj, matplotlib.axes.Axes):
#         if hasattr(obj, "export_as_csv"):
#             return obj
#         return None
# 
#     # Handle FigWrapper or similar SciTeX objects
#     if hasattr(obj, "figure") and hasattr(obj.figure, "axes"):
#         # Check if the wrapper itself has export methods
#         if hasattr(obj, "export_as_csv"):
#             return obj
# 
#         # Check the underlying figure's axes
#         for ax in obj.figure.axes:
#             if hasattr(ax, "export_as_csv"):
#                 return ax
# 
#         return None
# 
#     # Handle AxisWrapper or similar SciTeX objects
#     if hasattr(obj, "_axis_mpl") or hasattr(obj, "_ax"):
#         if hasattr(obj, "export_as_csv"):
#             return obj
#         return None
# 
#     # Try to get the current figure and its axes as fallback
#     try:
#         current_fig = plt.gcf()
#         current_ax = plt.gca()
# 
#         if hasattr(current_ax, "export_as_csv"):
#             return current_ax
#         elif hasattr(current_fig, "export_as_csv"):
#             return current_fig
# 
#         # Check all axes in current figure
#         for ax in current_fig.axes:
#             if hasattr(ax, "export_as_csv"):
#                 return ax
# 
#     except:
#         pass
# 
#     return None
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_save_modules/_figure_utils.py
# --------------------------------------------------------------------------------
