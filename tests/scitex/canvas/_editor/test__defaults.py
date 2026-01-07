# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_editor/_defaults.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # File: ./src/scitex/vis/editor/_defaults.py
# """Default style settings for SciTeX visual editor."""
# 
# from scitex.plt.styles import get_default_dpi
# 
# 
# def get_scitex_defaults():
#     """
#     Get default style values based on SciTeX publication standards.
# 
#     Returns
#     -------
#     dict
#         Dictionary of default style settings
#     """
#     return {
#         # Font sizes (from _configure_mpl.py)
#         "fontsize": 7,  # Base font size for publication
#         "title_fontsize": 8,  # Title size
#         "axis_fontsize": 7,  # Axis label size
#         "tick_fontsize": 7,  # Tick label size
#         "legend_fontsize": 6,  # Legend font size
#         # Line settings (in points, 1mm = 2.83465pt)
#         "linewidth": 0.57,  # Default line width (0.2mm)
#         # Tick settings
#         "n_ticks": 4,  # Number of ticks
#         "tick_length": 0.8,  # Tick length in mm
#         "tick_width": 0.2,  # Tick width in mm
#         "tick_direction": "out",  # Tick direction
#         # Axes settings
#         "axis_width": 0.2,  # Spine/axes line width in mm
#         "hide_top_spine": True,  # Hide top spine
#         "hide_right_spine": True,  # Hide right spine
#         # Grid settings
#         "grid": False,  # Grid off by default
#         "grid_linewidth": 0.6,
#         "grid_alpha": 0.3,
#         # Figure settings
#         "dpi": get_default_dpi(),  # From SCITEX_STYLE.yaml
#         "fig_size": [3.15, 2.68],  # Default figure size in inches
#         # Colors (SciTeX defaults)
#         "facecolor": "#ffffff",  # White background
#         "transparent": True,  # Transparent background
#         # Legend
#         "legend_visible": True,
#         "legend_frameon": False,
#         "legend_loc": "best",
#         # Font family
#         "font_family": "Arial",  # Sans-serif for publication
#         # Annotations
#         "annotations": [],
#     }
# 
# 
# def get_scitex_colors():
#     """
#     Get SciTeX color palette.
# 
#     Returns
#     -------
#     dict
#         Dictionary of named colors with hex values
#     """
#     # Based on scitex.plt.color.PARAMS
#     return {
#         "blue": "#1f77b4",
#         "orange": "#ff7f0e",
#         "green": "#2ca02c",
#         "red": "#d62728",
#         "purple": "#9467bd",
#         "brown": "#8c564b",
#         "pink": "#e377c2",
#         "gray": "#7f7f7f",
#         "olive": "#bcbd22",
#         "cyan": "#17becf",
#         # Additional publication colors
#         "black": "#000000",
#         "white": "#ffffff",
#         "dark_gray": "#404040",
#         "light_gray": "#d0d0d0",
#     }
# 
# 
# def extract_defaults_from_metadata(metadata):
#     """
#     Extract style defaults from loaded figure metadata.
# 
#     Parameters
#     ----------
#     metadata : dict
#         Figure metadata loaded from JSON
# 
#     Returns
#     -------
#     dict
#         Dictionary of style values extracted from metadata
#     """
#     defaults = get_scitex_defaults()
# 
#     if not metadata:
#         return defaults
# 
#     # Extract from scitex section (new format)
#     scitex_meta = metadata.get("scitex", {})
#     style_mm = scitex_meta.get("style_mm", {})
# 
#     # Font sizes from style_mm
#     if "axis_font_size_pt" in style_mm:
#         defaults["axis_fontsize"] = style_mm["axis_font_size_pt"]
#     if "tick_font_size_pt" in style_mm:
#         defaults["tick_fontsize"] = style_mm["tick_font_size_pt"]
#     if "title_font_size_pt" in style_mm:
#         defaults["title_fontsize"] = style_mm["title_font_size_pt"]
#     if "legend_font_size_pt" in style_mm:
#         defaults["legend_fontsize"] = style_mm["legend_font_size_pt"]
# 
#     # Line/axis thickness from metadata (in mm)
#     if "trace_thickness_mm" in style_mm:
#         defaults["linewidth"] = style_mm["trace_thickness_mm"] * 2.83465  # mm to pt
#     if "axis_thickness_mm" in style_mm:
#         defaults["axis_width"] = style_mm["axis_thickness_mm"]
#     if "tick_length_mm" in style_mm:
#         defaults["tick_length"] = style_mm["tick_length_mm"]
#     if "tick_thickness_mm" in style_mm:
#         defaults["tick_width"] = style_mm["tick_thickness_mm"]
#     if "n_ticks" in style_mm:
#         defaults["n_ticks"] = style_mm["n_ticks"]
# 
#     # Dimensions from metadata (support both old and new formats)
#     dimensions = metadata.get("dimensions", {})
#     if "dpi" in dimensions:
#         defaults["dpi"] = dimensions["dpi"]
#     if "figure_size_inch" in dimensions:
#         defaults["fig_size"] = dimensions["figure_size_inch"]
# 
#     # New format: size.width_mm, size.height_mm, size.dpi
#     size = metadata.get("size", {})
#     if "dpi" in size:
#         defaults["dpi"] = size["dpi"]
#     if "width_mm" in size and "height_mm" in size:
#         # Convert mm to inches
#         width_inch = size["width_mm"] / 25.4
#         height_inch = size["height_mm"] / 25.4
#         defaults["fig_size"] = [width_inch, height_inch]
# 
#     # Axis labels from metadata
#     if "xlabel" in metadata:
#         defaults["xlabel"] = metadata["xlabel"]
#     if "ylabel" in metadata:
#         defaults["ylabel"] = metadata["ylabel"]
#     if "title" in metadata:
#         defaults["title"] = metadata["title"]
# 
#     # Try to extract from axes if present
#     axes = metadata.get("axes", {})
#     if isinstance(axes, dict):
#         # New format: {'x': {'label': ...}, 'y': {'label': ...}}
#         x_axis = axes.get("x", {})
#         y_axis = axes.get("y", {})
#         if "label" in x_axis:
#             unit = x_axis.get("unit", "")
#             label = x_axis["label"]
#             if unit:
#                 defaults["xlabel"] = f"{label} [{unit}]"
#             else:
#                 defaults["xlabel"] = label
#         if "label" in y_axis:
#             unit = y_axis.get("unit", "")
#             label = y_axis["label"]
#             if unit:
#                 defaults["ylabel"] = f"{label} [{unit}]"
#             else:
#                 defaults["ylabel"] = label
#         # Extract axis limits
#         if "lim" in x_axis:
#             defaults["xlim"] = x_axis["lim"]
#         if "lim" in y_axis:
#             defaults["ylim"] = y_axis["lim"]
#     elif isinstance(axes, list) and len(axes) > 0:
#         # Legacy format: list of axes
#         ax = axes[0]
#         if "xlabel" in ax:
#             defaults["xlabel"] = ax["xlabel"]
#         if "ylabel" in ax:
#             defaults["ylabel"] = ax["ylabel"]
#         if "title" in ax:
#             defaults["title"] = ax["title"]
#         if "grid" in ax:
#             defaults["grid"] = ax["grid"]
#         # Extract axis limits from list format
#         if "xlim" in ax:
#             defaults["xlim"] = ax["xlim"]
#         if "ylim" in ax:
#             defaults["ylim"] = ax["ylim"]
# 
#     # Extract traces information - check multiple possible locations
#     traces = metadata.get("traces", [])
# 
#     # Also check axes[].lines (pltz bundle format)
#     if not traces and isinstance(axes, list) and len(axes) > 0:
#         ax = axes[0]
#         lines = ax.get("lines", [])
#         if lines:
#             traces = lines
# 
#     # Also check hit_regions.color_map for trace info
#     if not traces:
#         hit_regions = metadata.get("hit_regions", {})
#         color_map = hit_regions.get("color_map", {})
#         if color_map:
#             traces = []
#             for trace_id, trace_info in color_map.items():
#                 traces.append(
#                     {
#                         "id": trace_id,
#                         "label": trace_info.get("label", f"Trace {trace_id}"),
#                         "type": trace_info.get("type", "line"),
#                     }
#                 )
# 
#     if traces:
#         defaults["traces"] = traces
# 
#     # Extract legend information from multiple possible locations
#     legend = metadata.get("legend", {})
# 
#     # Also check selectable_regions.axes[0].legend (pltz bundle format)
#     if not legend:
#         selectable = metadata.get("selectable_regions", {})
#         sel_axes = selectable.get("axes", [])
#         if sel_axes and len(sel_axes) > 0:
#             legend = sel_axes[0].get("legend", {})
# 
#     if legend:
#         defaults["legend_visible"] = legend.get("visible", True)
#         defaults["legend_frameon"] = legend.get("frameon", False)
#         # Support both old format (loc) and new format (location)
#         loc = legend.get("location") or legend.get("loc", "best")
#         # Convert numeric legend loc to string (matplotlib accepts both but GUI needs string)
#         defaults["legend_loc"] = _normalize_legend_loc(loc)
#         # Extract fontsize if present
#         if "fontsize" in legend:
#             defaults["legend_fontsize"] = legend["fontsize"]
#         # Extract ncols if present
#         if "ncols" in legend:
#             defaults["legend_ncols"] = legend["ncols"]
#         # Extract title if present
#         if "title" in legend:
#             defaults["legend_title"] = legend["title"]
# 
#         # Extract legend entries if present
#         entries = legend.get("entries", [])
#         if entries and not traces:
#             # Use legend entries as trace labels
#             defaults["traces"] = [
#                 {"label": e.get("text", f"Trace {i}")} for i, e in enumerate(entries)
#             ]
# 
#     return defaults
# 
# 
# def _normalize_legend_loc(loc):
#     """Convert legend location to string format for GUI compatibility.
# 
#     Matplotlib accepts both numeric codes and string locations for legends.
#     This function ensures consistency by converting numeric codes to strings.
# 
#     Parameters
#     ----------
#     loc : int or str
#         Legend location (numeric or string)
# 
#     Returns
#     -------
#     str
#         String representation of legend location
#     """
#     if isinstance(loc, str):
#         return loc
# 
#     # Numeric to string mapping (matplotlib legend location codes)
#     loc_map = {
#         0: "best",
#         1: "upper right",
#         2: "upper left",
#         3: "lower left",
#         4: "lower right",
#         5: "right",
#         6: "center left",
#         7: "center right",
#         8: "lower center",
#         9: "upper center",
#         10: "center",
#     }
# 
#     return loc_map.get(loc, "best")
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_editor/_defaults.py
# --------------------------------------------------------------------------------
