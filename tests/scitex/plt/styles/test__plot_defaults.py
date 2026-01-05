# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/styles/_plot_defaults.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-01 10:00:00 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/styles/_plot_defaults.py
# 
# """Pre-processing default kwargs for plot methods.
# 
# This module centralizes all default styling applied BEFORE matplotlib
# methods are called. Each function modifies kwargs in-place.
# 
# Priority: direct kwarg → env var → YAML config → default
# 
# Style values use the key format from YAML (e.g., 'lines.trace_mm').
# Env vars: SCITEX_PLT_LINES_TRACE_MM (prefix + dots→underscores + uppercase)
# """
# 
# from scitex.plt.utils import mm_to_pt
# from scitex.plt.styles.presets import resolve_style_value
# 
# # Default alpha for fill regions (0.3 = semi-transparent)
# DEFAULT_FILL_ALPHA = 0.3
# 
# 
# # ============================================================================
# # Style helper function
# # ============================================================================
# def _get_style_value(key, default, style_dict=None):
#     """Get style value with priority: style_dict → active_style → env → yaml → default.
# 
#     Args:
#         key: YAML-style key (e.g., 'lines.trace_mm')
#         default: Fallback default value
#         style_dict: Optional user-provided style dict (overrides all)
# 
#     Returns:
#         Resolved style value
#     """
#     flat_key = _yaml_key_to_flat(key)
# 
#     # Priority 1: User passed explicit style dict
#     if style_dict is not None and flat_key in style_dict:
#         return style_dict[flat_key]
# 
#     # Priority 2: Check active style set via set_style()
#     from scitex.plt.styles.presets import _active_style
# 
#     if _active_style is not None and flat_key in _active_style:
#         return _active_style[flat_key]
# 
#     # Priority 3: Use resolve_style_value for: env → yaml → default
#     return resolve_style_value(key, None, default)
# 
# 
# def _yaml_key_to_flat(key):
#     """Convert YAML key to flat SCITEX_STYLE key.
# 
#     Examples:
#         'lines.trace_mm' -> 'trace_thickness_mm'
#         'markers.size_mm' -> 'marker_size_mm'
#     """
#     # Mapping from YAML keys to flat keys used in SCITEX_STYLE
#     mapping = {
#         "lines.trace_mm": "trace_thickness_mm",
#         "lines.errorbar_mm": "errorbar_thickness_mm",
#         "lines.errorbar_cap_mm": "errorbar_cap_width_mm",
#         "markers.size_mm": "marker_size_mm",
#     }
#     return mapping.get(key, key)
# 
# 
# # ============================================================================
# # Pre-processing functions
# # ============================================================================
# def apply_plot_defaults(method_name, kwargs, id_value=None, ax=None):
#     """Apply default kwargs for a plot method before calling matplotlib.
# 
#     Args:
#         method_name: Name of the matplotlib method being called
#         kwargs: Keyword arguments dict (modified in-place)
#         id_value: Optional id passed to the method
#         ax: The matplotlib axes (for methods needing axis setup)
# 
#     Returns:
#         Modified kwargs dict
# 
#     Note:
#         Priority: direct kwarg → style dict → env var → yaml → default
#         Users can pass `style=dict` kwarg to override env/yaml defaults.
#     """
#     # Extract optional style dict (removes 'style' key from kwargs)
#     style_dict = kwargs.pop("style", None)
# 
#     # Dispatch to method-specific defaults
#     if method_name == "plot":
#         _apply_plot_line_defaults(kwargs, id_value, style_dict)
#     elif method_name in ("bar", "barh"):
#         _apply_bar_defaults(kwargs, style_dict)
#     elif method_name == "errorbar":
#         _apply_errorbar_defaults(kwargs, style_dict)
#     elif method_name in ("fill_between", "fill_betweenx"):
#         _apply_fill_defaults(kwargs)
#     elif method_name in ("quiver", "streamplot"):
#         _apply_vector_field_defaults(method_name, kwargs, ax, style_dict)
#     elif method_name == "boxplot":
#         _apply_boxplot_defaults(kwargs)
#     elif method_name == "violinplot":
#         _apply_violinplot_defaults(kwargs)
# 
#     return kwargs
# 
# 
# def _apply_plot_line_defaults(kwargs, id_value=None, style_dict=None):
#     """Apply defaults for ax.plot() method."""
#     line_width_mm = _get_style_value("lines.trace_mm", 0.2, style_dict)
# 
#     # Default line width
#     if "linewidth" not in kwargs and "lw" not in kwargs:
#         kwargs["linewidth"] = mm_to_pt(line_width_mm)
# 
#     # KDE-specific styling when id contains "kde"
#     if id_value and "kde" in str(id_value).lower():
#         if "linestyle" not in kwargs and "ls" not in kwargs:
#             kwargs["linestyle"] = "--"
#         if "color" not in kwargs and "c" not in kwargs:
#             kwargs["color"] = "black"
# 
# 
# def _apply_bar_defaults(kwargs, style_dict=None):
#     """Apply defaults for ax.bar() and ax.barh() methods."""
#     line_width_mm = _get_style_value("lines.trace_mm", 0.2, style_dict)
# 
#     # Set error bar line thickness
#     if "error_kw" not in kwargs:
#         kwargs["error_kw"] = {}
#     if "elinewidth" not in kwargs.get("error_kw", {}):
#         kwargs["error_kw"]["elinewidth"] = mm_to_pt(line_width_mm)
#     if "capthick" not in kwargs.get("error_kw", {}):
#         kwargs["error_kw"]["capthick"] = mm_to_pt(line_width_mm)
#     # Set a temporary capsize that will be adjusted in post-processing
#     if "capsize" not in kwargs:
#         kwargs["capsize"] = 5  # Placeholder, adjusted later to 33% of bar width
# 
# 
# def _apply_errorbar_defaults(kwargs, style_dict=None):
#     """Apply defaults for ax.errorbar() method."""
#     line_width_mm = _get_style_value("lines.trace_mm", 0.2, style_dict)
#     cap_size_mm = _get_style_value("lines.errorbar_cap_mm", 0.8, style_dict)
# 
#     if "capsize" not in kwargs:
#         kwargs["capsize"] = mm_to_pt(cap_size_mm)
#     if "capthick" not in kwargs:
#         kwargs["capthick"] = mm_to_pt(line_width_mm)
#     if "elinewidth" not in kwargs:
#         kwargs["elinewidth"] = mm_to_pt(line_width_mm)
# 
# 
# def _apply_fill_defaults(kwargs):
#     """Apply defaults for ax.fill_between() and ax.fill_betweenx() methods."""
#     if "alpha" not in kwargs:
#         kwargs["alpha"] = DEFAULT_FILL_ALPHA  # Transparent to see overlapping data
# 
# 
# def _apply_vector_field_defaults(method_name, kwargs, ax, style_dict=None):
#     """Apply defaults for ax.quiver() and ax.streamplot() methods."""
#     line_width_mm = _get_style_value("lines.trace_mm", 0.2, style_dict)
#     marker_size_mm = _get_style_value("markers.size_mm", 0.8, style_dict)
# 
#     # Set equal aspect ratio for proper vector display
#     if ax is not None:
#         ax.set_aspect("equal", adjustable="datalim")
# 
#     if method_name == "streamplot":
#         if "arrowsize" not in kwargs:
#             # arrowsize is a scaling factor; scale relative to default
#             kwargs["arrowsize"] = mm_to_pt(marker_size_mm) / 3
#         if "linewidth" not in kwargs:
#             kwargs["linewidth"] = mm_to_pt(line_width_mm)
# 
#     elif method_name == "quiver":
#         if "width" not in kwargs:
#             kwargs["width"] = 0.003  # Narrow arrow shaft (axes fraction)
#         if "headwidth" not in kwargs:
#             kwargs["headwidth"] = 3  # Head width relative to shaft
#         if "headlength" not in kwargs:
#             kwargs["headlength"] = 4
#         if "headaxislength" not in kwargs:
#             kwargs["headaxislength"] = 3.5
# 
# 
# def _apply_boxplot_defaults(kwargs):
#     """Apply defaults for ax.boxplot() method."""
#     # Enable patch_artist for fillable boxes
#     if "patch_artist" not in kwargs:
#         kwargs["patch_artist"] = True
# 
# 
# def _apply_violinplot_defaults(kwargs):
#     """Apply defaults for ax.violinplot() method."""
#     # Default to showing boxplot overlay (can be disabled with boxplot=False)
#     # Store the boxplot setting for post-processing, then remove from kwargs
#     # so it doesn't get passed to matplotlib's violinplot
#     if "boxplot" not in kwargs:
#         kwargs["boxplot"] = True  # Default: add boxplot overlay
# 
#     # Default to hiding extrema (min/max bars) when boxplot is shown
#     if "showextrema" not in kwargs:
#         kwargs["showextrema"] = False
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/styles/_plot_defaults.py
# --------------------------------------------------------------------------------
