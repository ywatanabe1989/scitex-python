# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/ax/_style/_style_violinplot.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-01 20:00:00 (ywatanabe)"
# # File: ./src/scitex/plt/ax/_style/_style_violinplot.py
# 
# """Style violin plot elements with millimeter-based control.
# 
# Default values are loaded from SCITEX_STYLE.yaml via presets.py.
# """
# 
# from typing import Optional, Union
# 
# from matplotlib.axes import Axes
# 
# from scitex.plt.styles.presets import SCITEX_STYLE
# 
# # Get defaults from centralized config
# _DEFAULT_LINEWIDTH_MM = SCITEX_STYLE.get("trace_thickness_mm", 0.2)
# 
# 
# def style_violinplot(
#     ax: Union[Axes, "AxisWrapper"],
#     linewidth_mm: float = None,
#     edge_color: str = "black",
#     median_color: str = "black",
#     remove_caps: bool = True,
# ) -> Union[Axes, "AxisWrapper"]:
#     """Apply publication-quality styling to seaborn violin plots.
# 
#     This function modifies violin plots created by seaborn.violinplot() to:
#     - Add borders to the KDE (violin body) edges
#     - Remove caps from the internal boxplot whiskers
#     - Set median line to black for better visibility
#     - Apply consistent line widths
# 
#     Parameters
#     ----------
#     ax : matplotlib.axes.Axes or AxisWrapper
#         The axes containing the violin plot.
#     linewidth_mm : float, default 0.2
#         Line width in millimeters for violin edges and boxplot elements.
#     edge_color : str, default "black"
#         Color for the violin body edges.
#     median_color : str, default "black"
#         Color for the median line inside the boxplot.
#     remove_caps : bool, default True
#         Whether to remove the caps (horizontal lines) from boxplot whiskers.
# 
#     Returns
#     -------
#     ax : matplotlib.axes.Axes or AxisWrapper
#         The axes with styled violin plot.
# 
#     Examples
#     --------
#     >>> import seaborn as sns
#     >>> import scitex as stx
#     >>> fig, ax = stx.plt.subplots()
#     >>> sns.violinplot(data=df, x="group", y="value", ax=ax)
#     >>> stx.plt.ax.style_violinplot(ax)
#     """
#     from scitex.plt.utils import mm_to_pt
# 
#     # Use centralized default if not specified
#     if linewidth_mm is None:
#         linewidth_mm = _DEFAULT_LINEWIDTH_MM
# 
#     lw_pt = mm_to_pt(linewidth_mm)
# 
#     # Style violin bodies (PolyCollection)
#     for collection in ax.collections:
#         # Check if it's a violin body (PolyCollection with filled area)
#         if hasattr(collection, "set_edgecolor"):
#             collection.set_edgecolor(edge_color)
#             collection.set_linewidth(lw_pt)
# 
#     # Style internal boxplot elements (Line2D objects)
#     # Seaborn violin plot lines: whiskers (vertical), caps (horizontal), median (short horizontal)
#     lines = list(ax.lines)
#     n_violins = len(
#         [
#             c
#             for c in ax.collections
#             if hasattr(c, "get_paths") and len(c.get_paths()) > 0
#         ]
#     )
# 
#     for line in lines:
#         # Get line data to identify element type
#         xdata = line.get_xdata()
#         ydata = line.get_ydata()
# 
#         if len(ydata) != 2:
#             continue
# 
#         # Caps are horizontal lines (same y-value for both points) with wider x-span
#         is_horizontal = ydata[0] == ydata[1]
#         x_span = abs(xdata[1] - xdata[0]) if len(xdata) == 2 else 0
# 
#         if is_horizontal:
#             if remove_caps and x_span > 0.05:
#                 # This is likely a cap (wider horizontal line at whisker ends)
#                 line.set_visible(False)
#             else:
#                 # This is likely a median line (short horizontal line)
#                 line.set_color(median_color)
#                 line.set_linewidth(lw_pt)
#         else:
#             # Vertical lines (whiskers)
#             line.set_linewidth(lw_pt)
# 
#     return ax
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/ax/_style/_style_violinplot.py
# --------------------------------------------------------------------------------
