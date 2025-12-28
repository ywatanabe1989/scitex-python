# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/styles/_plot_postprocess.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-01 10:00:00 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/styles/_plot_postprocess.py
# 
# """Post-processing styling for plot methods.
# 
# This module centralizes all styling applied AFTER matplotlib methods
# are called. Each function modifies the plot result or axes in-place.
# 
# All default values are loaded from SCITEX_STYLE.yaml via presets.py.
# """
# 
# import numpy as np
# from matplotlib.ticker import MaxNLocator, FixedLocator
# from matplotlib.category import StrCategoryConverter, UnitData
# 
# from scitex.plt.utils import mm_to_pt
# from scitex.plt.styles.presets import SCITEX_STYLE
# 
# 
# # ============================================================================
# # Constants (loaded from centralized SCITEX_STYLE.yaml)
# # ============================================================================
# DEFAULT_LINE_WIDTH_MM = SCITEX_STYLE.get("trace_thickness_mm", 0.2)
# DEFAULT_MARKER_SIZE_MM = SCITEX_STYLE.get("marker_size_mm", 0.8)
# DEFAULT_N_TICKS = SCITEX_STYLE.get("n_ticks", 4) - 1  # nbins = n_ticks - 1
# SPINE_ZORDER = 1000
# CAP_WIDTH_RATIO = 1 / 3  # 33% of bar/box width
# 
# 
# # ============================================================================
# # Main post-processing function
# # ============================================================================
# def apply_plot_postprocess(method_name, result, ax, kwargs, args=None):
#     """Apply post-processing styling after matplotlib method call.
# 
#     Args:
#         method_name: Name of the matplotlib method that was called
#         result: Return value from the matplotlib method
#         ax: The matplotlib axes
#         kwargs: Original kwargs passed to the method
#         args: Original positional args passed to the method (needed for violinplot)
# 
#     Returns:
#         The result (possibly modified)
#     """
#     # Always ensure spines are on top
#     _ensure_spines_on_top(ax)
# 
#     # Apply tick locator for numerical axes
#     _apply_tick_locator(ax)
# 
#     # Method-specific post-processing
#     if method_name == "pie" and result is not None:
#         _postprocess_pie(result)
#     elif method_name == "stem" and result is not None:
#         _postprocess_stem(result)
#     elif method_name == "violinplot" and result is not None:
#         _postprocess_violin(result, ax, kwargs, args)
#     elif method_name == "boxplot" and result is not None:
#         _postprocess_boxplot(result, ax)
#     elif method_name == "scatter" and result is not None:
#         _postprocess_scatter(result, kwargs)
#     elif method_name == "bar" and result is not None:
#         _postprocess_bar(result, ax, kwargs)
#     elif method_name == "barh" and result is not None:
#         _postprocess_barh(result, ax, kwargs)
#     elif method_name == "errorbar" and result is not None:
#         _postprocess_errorbar(result)
#     elif method_name == "hist" and result is not None:
#         _postprocess_hist(result, ax)
#     elif method_name == "fill_between" and result is not None:
#         _postprocess_fill_between(result, kwargs)
# 
#     return result
# 
# 
# # ============================================================================
# # General post-processing
# # ============================================================================
# def _ensure_spines_on_top(ax):
#     """Ensure axes spines are always drawn in front of plot elements."""
#     try:
#         ax.set_axisbelow(False)
# 
#         # Set very high z-order for spines
#         for spine in ax.spines.values():
#             spine.set_zorder(SPINE_ZORDER)
# 
#         # Set z-order for tick marks
#         ax.tick_params(zorder=SPINE_ZORDER)
# 
#         # Ensure plot patches have lower z-order than spines
#         # But preserve intentionally set z-orders (e.g., boxplot in violin)
#         for patch in ax.patches:
#             current_z = patch.get_zorder()
#             # Only lower z-order if it's >= SPINE_ZORDER or is at matplotlib default (1)
#             if current_z >= SPINE_ZORDER:
#                 patch.set_zorder(current_z - SPINE_ZORDER)
#             elif current_z == 1:
#                 # Default matplotlib z-order, lower it
#                 patch.set_zorder(0.5)
#             # Otherwise, preserve the intentionally set z-order
# 
#         # Set axes patch behind everything
#         ax.patch.set_zorder(-1)
#     except Exception:
#         pass
# 
# 
# def _apply_tick_locator(ax):
#     """Apply MaxNLocator only to numerical (non-categorical) axes.
# 
#     Target: 3-4 ticks per axis for clean publication figures.
#     MaxNLocator's nbins=3 gives approximately 3-4 tick marks.
#     min_n_ticks=3 ensures at least 3 ticks (never 2).
#     """
#     try:
# 
#         def is_categorical_axis(axis):
#             # Use get_converter() for matplotlib 3.10+ compatibility
#             converter = getattr(axis, 'get_converter', lambda: axis.converter)()
#             if isinstance(converter, StrCategoryConverter):
#                 return True
#             if hasattr(axis, "units") and isinstance(axis.units, UnitData):
#                 return True
#             if isinstance(axis.get_major_locator(), FixedLocator):
#                 return True
#             return False
# 
#         if not is_categorical_axis(ax.xaxis):
#             ax.xaxis.set_major_locator(
#                 MaxNLocator(
#                     nbins=DEFAULT_N_TICKS, min_n_ticks=3, integer=False, prune=None
#                 )
#             )
# 
#         if not is_categorical_axis(ax.yaxis):
#             ax.yaxis.set_major_locator(
#                 MaxNLocator(
#                     nbins=DEFAULT_N_TICKS, min_n_ticks=3, integer=False, prune=None
#                 )
#             )
#     except Exception:
#         pass
# 
# 
# # ============================================================================
# # Method-specific post-processing
# # ============================================================================
# def _postprocess_pie(result):
#     """Apply styling for pie charts."""
#     # pie returns (wedges, texts, autotexts) when autopct is used
#     if len(result) >= 3:
#         autotexts = result[2]
#         for autotext in autotexts:
#             autotext.set_fontsize(6)  # 6pt for inline percentages
# 
# 
# def _postprocess_stem(result):
#     """Apply styling for stem plots."""
#     baseline = result.baseline
#     if baseline is not None:
#         baseline.set_color("black")
#         baseline.set_linestyle("--")
# 
# 
# def _postprocess_errorbar(result):
#     """Apply styling for errorbar plots.
# 
#     Simplifies the legend to show only a line (no caps/bars).
#     """
#     import matplotlib.legend as mlegend
#     from matplotlib.container import ErrorbarContainer
#     from matplotlib.legend_handler import HandlerErrorbar, HandlerLine2D
# 
#     # Custom handler that shows only a simple line for errorbar
#     class SimpleLineHandler(HandlerErrorbar):
#         def create_artists(
#             self,
#             legend,
#             orig_handle,
#             xdescent,
#             ydescent,
#             width,
#             height,
#             fontsize,
#             trans,
#         ):
#             # Use HandlerLine2D to create just a line
#             line_handler = HandlerLine2D()
#             # Get the data line from the ErrorbarContainer
#             data_line = orig_handle[0]
#             if data_line is not None:
#                 return line_handler.create_artists(
#                     legend,
#                     data_line,
#                     xdescent,
#                     ydescent,
#                     width,
#                     height,
#                     fontsize,
#                     trans,
#                 )
#             return []
# 
#     # Register the handler globally for ErrorbarContainer
#     mlegend.Legend.update_default_handler_map({ErrorbarContainer: SimpleLineHandler()})
# 
# 
# def _postprocess_violin(result, ax, kwargs, args):
#     """Apply styling for violin plots with optional boxplot overlay."""
#     # Get scitex palette for coloring
#     from scitex.plt.color._PARAMS import HEX
# 
#     palette = [
#         HEX["blue"],
#         HEX["red"],
#         HEX["green"],
#         HEX["yellow"],
#         HEX["purple"],
#         HEX["orange"],
#         HEX["lightblue"],
#         HEX["pink"],
#     ]
# 
#     if "bodies" in result:
#         for i, body in enumerate(result["bodies"]):
#             body.set_facecolor(palette[i % len(palette)])
#             body.set_edgecolor("black")
#             body.set_linewidth(mm_to_pt(DEFAULT_LINE_WIDTH_MM))
#             body.set_alpha(1.0)
# 
#     # Add boxplot overlay by default (disable with boxplot=False)
#     add_boxplot = kwargs.pop("boxplot", True)
#     if add_boxplot and args:
#         try:
#             # Get data from first positional argument
#             data = args[0]
#             # Get positions if specified, otherwise use default
#             positions = kwargs.get("positions", None)
#             if positions is None:
#                 positions = range(1, len(data) + 1)
# 
#             # Calculate boxplot width dynamically from violin width
#             # Get violin width from kwargs or use matplotlib default (0.5)
#             violin_widths = kwargs.get("widths", 0.5)
#             if hasattr(violin_widths, "__iter__"):
#                 violin_widths = violin_widths[0] if len(violin_widths) > 0 else 0.5
#             # Boxplot width = 20% of violin width
#             boxplot_widths = violin_widths * 0.2
# 
#             # Draw boxplot overlay with styling
#             line_width = mm_to_pt(DEFAULT_LINE_WIDTH_MM)
#             marker_size = mm_to_pt(DEFAULT_MARKER_SIZE_MM)
# 
#             # Call matplotlib's boxplot directly to avoid recursive post-processing
#             # which would override our gray styling with the default blue
#             if hasattr(ax, "_axes_mpl"):
#                 mpl_ax = ax._axes_mpl
#             else:
#                 mpl_ax = ax
#             bp = mpl_ax.boxplot(
#                 data,
#                 positions=list(positions),
#                 widths=boxplot_widths,
#                 patch_artist=True,
#                 manage_ticks=False,  # Don't modify existing ticks
#             )
# 
#             # Style the boxplot: scitex gray fill with black edges for visibility
#             # Set high z-order so boxplot appears on top of violin bodies
#             from scitex.plt.color._PARAMS import HEX
# 
#             boxplot_zorder = 10
#             for box in bp.get("boxes", []):
#                 box.set_facecolor(HEX["gray"])  # Scitex gray fill
#                 box.set_edgecolor("black")
#                 box.set_alpha(1.0)
#                 box.set_linewidth(line_width)
#                 box.set_zorder(boxplot_zorder)
#             for median in bp.get("medians", []):
#                 median.set_color("black")  # Black median line
#                 median.set_linewidth(line_width)  # 0.2mm thickness
#                 median.set_zorder(boxplot_zorder + 1)
#             for whisker in bp.get("whiskers", []):
#                 whisker.set_color("black")
#                 whisker.set_linewidth(line_width)
#                 whisker.set_zorder(boxplot_zorder)
#             for cap in bp.get("caps", []):
#                 cap.set_color("black")
#                 cap.set_linewidth(line_width)
#                 cap.set_zorder(boxplot_zorder)
#             for flier in bp.get("fliers", []):
#                 flier.set_markerfacecolor("none")  # No fill (open circles)
#                 flier.set_markeredgecolor("black")
#                 flier.set_markersize(marker_size)  # 0.8mm
#                 flier.set_markeredgewidth(line_width)  # 0.2mm
#                 flier.set_zorder(boxplot_zorder + 2)
#         except Exception:
#             pass  # Silently continue if boxplot overlay fails
# 
# 
# def _postprocess_boxplot(result, ax):
#     """Apply styling for boxplots (standalone, not violin overlay)."""
#     # Use the centralized style_boxplot function for consistent styling
#     from scitex.plt.ax import style_boxplot
# 
#     style_boxplot(result)
# 
#     # Cap width: 33% of box width
#     if "caps" in result and "boxes" in result and len(result["boxes"]) > 0:
#         try:
#             cap_width_pts = _calculate_cap_width_from_box(result["boxes"][0], ax)
#             for cap in result["caps"]:
#                 cap.set_markersize(cap_width_pts)
#         except Exception:
#             pass
# 
# 
# def _postprocess_scatter(result, kwargs):
#     """Apply styling for scatter plots."""
#     # Apply default 0.8mm marker size if 's' not specified
#     if "s" not in kwargs:
#         size_pt = mm_to_pt(DEFAULT_MARKER_SIZE_MM)
#         marker_area = size_pt**2
#         result.set_sizes([marker_area])
# 
# 
# def _postprocess_hist(result, ax):
#     """Apply styling for histogram plots.
# 
#     Ensures histogram bars have proper edge color and alpha for visibility.
#     """
#     line_width = mm_to_pt(DEFAULT_LINE_WIDTH_MM)
# 
#     # result is (n, bins, patches) tuple
#     if len(result) >= 3:
#         patches = result[2]
#         # Handle both single histogram and stacked histograms
#         if hasattr(patches, "__iter__"):
#             for patch_group in patches:
#                 if hasattr(patch_group, "__iter__"):
#                     for patch in patch_group:
#                         patch.set_edgecolor("black")
#                         patch.set_linewidth(line_width)
#                         # Ensure alpha is at least 0.7 for visibility
#                         if patch.get_alpha() is None or patch.get_alpha() < 0.7:
#                             patch.set_alpha(1.0)
#                 else:
#                     # Single patch
#                     patch_group.set_edgecolor("black")
#                     patch_group.set_linewidth(line_width)
#                     if patch_group.get_alpha() is None or patch_group.get_alpha() < 0.7:
#                         patch_group.set_alpha(1.0)
# 
# 
# def _postprocess_fill_between(result, kwargs):
#     """Apply styling for fill_between plots.
# 
#     Ensures shaded regions have proper alpha for visibility.
#     """
#     # result is a PolyCollection
#     if result is not None:
#         # Set edge color to match face color or black
#         line_width = mm_to_pt(DEFAULT_LINE_WIDTH_MM)
# 
#         # Only set edge if not already specified
#         if "edgecolor" not in kwargs and "ec" not in kwargs:
#             result.set_edgecolor("none")
# 
#         # Ensure alpha is reasonable (default 0.3 is common for fill_between)
#         if "alpha" not in kwargs:
#             result.set_alpha(0.3)
# 
# 
# def _postprocess_bar(result, ax, kwargs):
#     """Apply styling for bar plots with colors and error bars."""
#     # Get scitex palette for coloring (only if color not explicitly set)
#     if "color" not in kwargs and "c" not in kwargs:
#         from scitex.plt.color._PARAMS import HEX
# 
#         palette = [
#             HEX["blue"],
#             HEX["red"],
#             HEX["green"],
#             HEX["yellow"],
#             HEX["purple"],
#             HEX["orange"],
#             HEX["lightblue"],
#             HEX["pink"],
#         ]
# 
#         line_width = mm_to_pt(DEFAULT_LINE_WIDTH_MM)
#         for i, patch in enumerate(result.patches):
#             patch.set_facecolor(palette[i % len(palette)])
#             patch.set_edgecolor("black")
#             patch.set_linewidth(line_width)
# 
#     if "yerr" not in kwargs or kwargs["yerr"] is None:
#         return
# 
#     try:
#         errorbar = result.errorbar
#         if errorbar is None:
#             return
# 
#         lines = errorbar.lines
#         if not lines or len(lines) < 3:
#             return
# 
#         caplines = lines[1]
#         if caplines and len(caplines) >= 2:
#             # Hide lower caps (one-sided error bars)
#             caplines[0].set_visible(False)
# 
#             # Adjust cap width to 33% of bar width
#             if len(result.patches) > 0:
#                 cap_width_pts = _calculate_cap_width_from_bar(
#                     result.patches[0], ax, "width"
#                 )
#                 for cap in caplines[1:]:
#                     cap.set_markersize(cap_width_pts)
# 
#         # Make error bar lines one-sided
#         barlinecols = lines[2]
#         _make_errorbar_one_sided(barlinecols, "vertical")
#     except Exception:
#         pass
# 
# 
# def _postprocess_barh(result, ax, kwargs):
#     """Apply styling for horizontal bar plots with colors and error bars."""
#     # Get scitex palette for coloring (only if color not explicitly set)
#     if "color" not in kwargs and "c" not in kwargs:
#         from scitex.plt.color._PARAMS import HEX
# 
#         palette = [
#             HEX["blue"],
#             HEX["red"],
#             HEX["green"],
#             HEX["yellow"],
#             HEX["purple"],
#             HEX["orange"],
#             HEX["lightblue"],
#             HEX["pink"],
#         ]
# 
#         line_width = mm_to_pt(DEFAULT_LINE_WIDTH_MM)
#         for i, patch in enumerate(result.patches):
#             patch.set_facecolor(palette[i % len(palette)])
#             patch.set_edgecolor("black")
#             patch.set_linewidth(line_width)
# 
#     if "xerr" not in kwargs or kwargs["xerr"] is None:
#         return
# 
#     try:
#         errorbar = result.errorbar
#         if errorbar is None:
#             return
# 
#         lines = errorbar.lines
#         if not lines or len(lines) < 3:
#             return
# 
#         caplines = lines[1]
#         if caplines and len(caplines) >= 2:
#             # Hide left caps (one-sided error bars)
#             caplines[0].set_visible(False)
# 
#             # Adjust cap width to 33% of bar height
#             if len(result.patches) > 0:
#                 cap_width_pts = _calculate_cap_width_from_bar(
#                     result.patches[0], ax, "height"
#                 )
#                 for cap in caplines[1:]:
#                     cap.set_markersize(cap_width_pts)
# 
#         # Make error bar lines one-sided
#         barlinecols = lines[2]
#         _make_errorbar_one_sided(barlinecols, "horizontal")
#     except Exception:
#         pass
# 
# 
# # ============================================================================
# # Helper functions
# # ============================================================================
# def _calculate_cap_width_from_box(box, ax):
#     """Calculate cap width as 33% of box width in points."""
#     # Get box width from path
#     if hasattr(box, "get_path"):
#         path = box.get_path()
#         vertices = path.vertices
#         x_coords = vertices[:, 0]
#         box_width_data = x_coords.max() - x_coords.min()
#     elif hasattr(box, "get_xdata"):
#         x_data = box.get_xdata()
#         box_width_data = max(x_data) - min(x_data)
#     else:
#         box_width_data = 0.5  # Default
# 
#     return _data_width_to_points(box_width_data, ax, "x") * CAP_WIDTH_RATIO
# 
# 
# def _calculate_cap_width_from_bar(patch, ax, dimension):
#     """Calculate cap width as 33% of bar width/height in points."""
#     if dimension == "width":
#         bar_size = patch.get_width()
#         return _data_width_to_points(bar_size, ax, "x") * CAP_WIDTH_RATIO
#     else:  # height
#         bar_size = patch.get_height()
#         return _data_width_to_points(bar_size, ax, "y") * CAP_WIDTH_RATIO
# 
# 
# def _data_width_to_points(data_size, ax, axis="x"):
#     """Convert a data-space size to points."""
#     fig = ax.get_figure()
#     bbox = ax.get_position()
# 
#     if axis == "x":
#         ax_size_inches = bbox.width * fig.get_figwidth()
#         lim = ax.get_xlim()
#     else:
#         ax_size_inches = bbox.height * fig.get_figheight()
#         lim = ax.get_ylim()
# 
#     data_range = lim[1] - lim[0]
#     size_inches = (data_size / data_range) * ax_size_inches
#     return size_inches * 72  # 72 points per inch
# 
# 
# def _make_errorbar_one_sided(barlinecols, direction):
#     """Make error bar line segments one-sided (outward only)."""
#     if not barlinecols or len(barlinecols) == 0:
#         return
# 
#     for lc in barlinecols:
#         if not hasattr(lc, "get_segments"):
#             continue
# 
#         segs = lc.get_segments()
#         new_segs = []
#         for seg in segs:
#             if len(seg) < 2:
#                 continue
# 
#             if direction == "vertical":
#                 # Keep upper half
#                 bottom_y = min(seg[0][1], seg[1][1])
#                 top_y = max(seg[0][1], seg[1][1])
#                 mid_y = (bottom_y + top_y) / 2
#                 new_seg = np.array([[seg[0][0], mid_y], [seg[0][0], top_y]])
#             else:  # horizontal
#                 # Keep right half
#                 left_x = min(seg[0][0], seg[1][0])
#                 right_x = max(seg[0][0], seg[1][0])
#                 mid_x = (left_x + right_x) / 2
#                 new_seg = np.array([[mid_x, seg[0][1]], [right_x, seg[0][1]]])
# 
#             new_segs.append(new_seg)
# 
#         if new_segs:
#             lc.set_segments(new_segs)
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/styles/_plot_postprocess.py
# --------------------------------------------------------------------------------
