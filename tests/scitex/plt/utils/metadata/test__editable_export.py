# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/metadata/_editable_export.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-10 (ywatanabe)"
# # File: scitex/plt/utils/metadata/_editable_export.py
# 
# """
# Main export function for Schema v0.3 (scitex.plt.figure.editable).
# 
# This module provides the top-level export_editable_figure() function that
# orchestrates geometry extraction for all elements in a matplotlib figure,
# producing a JSON-serializable dictionary suitable for interactive editing.
# """
# 
# from typing import Dict, Any, Optional, List
# import numpy as np
# from datetime import datetime
# 
# from ._geometry_extraction import (
#     extract_axes_bbox_px,
#     extract_line_geometry,
#     extract_scatter_geometry,
#     extract_polygon_geometry,
#     extract_rectangle_geometry,
#     extract_bar_group_geometry,
#     extract_text_geometry,
#     extract_image_geometry,
# )
# 
# 
# def export_editable_figure(
#     fig,
#     title: str = "",
#     description: str = "",
#     include_full_paths: bool = False,
#     simplify_threshold: float = 0.5,
# ) -> Dict[str, Any]:
#     """
#     Export a matplotlib figure with geometry data for interactive editing.
# 
#     This produces a Schema v0.3 compliant dictionary with axes-local pixel
#     coordinates for all visual elements, suitable for shape-based hit testing.
# 
#     Parameters
#     ----------
#     fig : matplotlib.figure.Figure or scitex.plt.FigureWrapper
#         The figure to export
#     title : str
#         Optional title for the figure metadata
#     description : str
#         Optional description for the figure metadata
#     include_full_paths : bool
#         If True, include full (unsimplified) paths for lines. Default False.
#     simplify_threshold : float
#         Maximum pixel error for path simplification. Default 0.5.
# 
#     Returns
#     -------
#     dict
#         Schema v0.3 compliant dictionary with structure:
#         - scitex_schema: "scitex.plt.figure.editable"
#         - scitex_schema_version: "0.3.0"
#         - figure: {size_px, dpi, ...}
#         - axes: {ax_id: {bbox_px, ...}, ...}
#         - elements: {elem_id: {geometry_px, style, ...}, ...}
#         - statistics: {} (placeholder for stats integration)
#         - data: {} (placeholder for linked data)
#         - output: {} (rendering metadata)
#     """
#     # Handle scitex FigureWrapper
#     mpl_fig = fig.figure if hasattr(fig, 'figure') else fig
# 
#     # Ensure we have a renderer
#     try:
#         renderer = mpl_fig.canvas.get_renderer()
#     except Exception:
#         mpl_fig.canvas.draw()
#         renderer = mpl_fig.canvas.get_renderer()
# 
#     # Figure dimensions
#     fig_width_px = int(mpl_fig.get_figwidth() * mpl_fig.dpi)
#     fig_height_px = int(mpl_fig.get_figheight() * mpl_fig.dpi)
# 
#     # Build output structure
#     output = {
#         "scitex_schema": "scitex.plt.figure.editable",
#         "scitex_schema_version": "0.3.0",
#         "meta": {
#             "title": title,
#             "description": description,
#             "exported_at": datetime.now().isoformat(),
#         },
#         "figure": {
#             "size_px": [fig_width_px, fig_height_px],
#             "dpi": mpl_fig.dpi,
#             "figsize_inches": [mpl_fig.get_figwidth(), mpl_fig.get_figheight()],
#         },
#         "axes": {},
#         "elements": {},
#         "statistics": {},
#         "data": {},
#         "output": {
#             "format": "png",
#             "width_px": fig_width_px,
#             "height_px": fig_height_px,
#         },
#     }
# 
#     # Get all axes
#     axes_list = mpl_fig.get_axes()
# 
#     for ax_idx, ax in enumerate(axes_list):
#         # Handle scitex AxisWrapper
#         mpl_ax = ax._axis_mpl if hasattr(ax, '_axis_mpl') else ax
# 
#         ax_id = f"ax_{ax_idx:02d}"
# 
#         # Extract axes bbox
#         axes_bbox = extract_axes_bbox_px(mpl_ax, mpl_fig)
#         output["axes"][ax_id] = {
#             "bbox_px": axes_bbox,
#             "position": list(mpl_ax.get_position().bounds),
#             "xlim": list(mpl_ax.get_xlim()),
#             "ylim": list(mpl_ax.get_ylim()),
#         }
# 
#         # Extract elements from this axes
#         elements = _extract_axes_elements(
#             mpl_ax, mpl_fig, ax_id, renderer,
#             include_full_paths=include_full_paths,
#             simplify_threshold=simplify_threshold,
#         )
#         output["elements"].update(elements)
# 
#     return output
# 
# 
# def _extract_axes_elements(
#     ax, fig, ax_id: str, renderer,
#     include_full_paths: bool = False,
#     simplify_threshold: float = 0.5,
# ) -> Dict[str, Any]:
#     """
#     Extract all visual elements from a single axes.
# 
#     Parameters
#     ----------
#     ax : matplotlib.axes.Axes
#         The axes object
#     fig : matplotlib.figure.Figure
#         The figure object
#     ax_id : str
#         Identifier for the axes (e.g., "ax_00")
#     renderer : RendererBase
#         Matplotlib renderer for text bbox extraction
#     include_full_paths : bool
#         Include unsimplified paths
#     simplify_threshold : float
#         Path simplification threshold in pixels
# 
#     Returns
#     -------
#     dict
#         Dictionary of {element_id: element_data}
#     """
#     elements = {}
#     zorder_counter = 0
# 
#     # Extract lines (Line2D)
#     for idx, line in enumerate(ax.lines):
#         elem_id = f"{ax_id}_line_{idx:02d}"
#         label = line.get_label()
#         if label is None or label.startswith('_'):
#             label = f"line_{idx}"
# 
#         geom = extract_line_geometry(line, ax, fig, simplify_threshold)
# 
#         elements[elem_id] = {
#             "id": elem_id,
#             "axes_id": ax_id,
#             "element_type": "line",
#             "label": label,
#             "geometry_px": geom,
#             "style": _extract_line_style(line),
#             "editable_styles": ["color", "linewidth", "linestyle", "alpha", "marker"],
#             "zorder": line.get_zorder(),
#             "visible": line.get_visible(),
#         }
#         zorder_counter += 1
# 
#     # Extract collections (scatter, polygon fills, etc.)
#     for idx, coll in enumerate(ax.collections):
#         coll_type = type(coll).__name__
# 
#         if coll_type == "PathCollection":
#             # Scatter plot
#             elem_id = f"{ax_id}_scatter_{idx:02d}"
#             label = coll.get_label()
#             if label is None or label.startswith('_'):
#                 label = f"scatter_{idx}"
# 
#             geom = extract_scatter_geometry(coll, ax, fig)
# 
#             elements[elem_id] = {
#                 "id": elem_id,
#                 "axes_id": ax_id,
#                 "element_type": "scatter",
#                 "label": label,
#                 "geometry_px": geom,
#                 "style": _extract_scatter_style(coll),
#                 "editable_styles": ["facecolor", "edgecolor", "s", "alpha", "marker"],
#                 "zorder": coll.get_zorder(),
#                 "visible": coll.get_visible(),
#             }
# 
#         elif coll_type in ("PolyCollection", "FillBetweenPolyCollection"):
#             # Fill_between, violin, etc.
#             elem_id = f"{ax_id}_fill_{idx:02d}"
#             label = coll.get_label()
#             if label is None or label.startswith('_'):
#                 label = f"fill_{idx}"
# 
#             geom = extract_polygon_geometry(coll, ax, fig)
# 
#             elements[elem_id] = {
#                 "id": elem_id,
#                 "axes_id": ax_id,
#                 "element_type": "fill",
#                 "label": label,
#                 "geometry_px": geom,
#                 "style": _extract_polygon_style(coll),
#                 "editable_styles": ["facecolor", "edgecolor", "alpha"],
#                 "zorder": coll.get_zorder(),
#                 "visible": coll.get_visible(),
#             }
# 
#         elif coll_type == "LineCollection":
#             # Multiple lines (e.g., errorbar caps)
#             elem_id = f"{ax_id}_linecoll_{idx:02d}"
#             elements[elem_id] = {
#                 "id": elem_id,
#                 "axes_id": ax_id,
#                 "element_type": "line_collection",
#                 "label": f"linecoll_{idx}",
#                 "geometry_px": {"coord_space": "axes", "bbox": None},
#                 "style": {},
#                 "editable_styles": ["color", "linewidth", "alpha"],
#                 "zorder": coll.get_zorder(),
#                 "visible": coll.get_visible(),
#             }
# 
#     # Extract patches (bars, rectangles)
#     bar_patches = []
#     other_patches = []
# 
#     for patch in ax.patches:
#         patch_type = type(patch).__name__
#         if patch_type == "Rectangle":
#             # Check if it's part of a bar plot (has nonzero width and height)
#             if patch.get_width() != 0 and patch.get_height() != 0:
#                 bar_patches.append(patch)
#             else:
#                 other_patches.append(patch)
#         else:
#             other_patches.append(patch)
# 
#     # Group bar patches by similar x position (bars) or y position (horizontal bars)
#     if bar_patches:
#         elem_id = f"{ax_id}_bars"
#         geom = extract_bar_group_geometry(bar_patches, ax, fig)
# 
#         elements[elem_id] = {
#             "id": elem_id,
#             "axes_id": ax_id,
#             "element_type": "bar",
#             "label": "bars",
#             "geometry_px": geom,
#             "style": _extract_bar_style(bar_patches[0] if bar_patches else None),
#             "editable_styles": ["facecolor", "edgecolor", "alpha", "linewidth"],
#             "zorder": bar_patches[0].get_zorder() if bar_patches else 0,
#             "visible": True,
#         }
# 
#     # Extract images
#     for idx, img in enumerate(ax.images):
#         elem_id = f"{ax_id}_image_{idx:02d}"
#         geom = extract_image_geometry(img, ax, fig)
# 
#         elements[elem_id] = {
#             "id": elem_id,
#             "axes_id": ax_id,
#             "element_type": "image",
#             "label": f"image_{idx}",
#             "geometry_px": geom,
#             "style": {"cmap": str(img.get_cmap().name), "alpha": img.get_alpha()},
#             "editable_styles": ["cmap", "alpha", "clim"],
#             "zorder": img.get_zorder(),
#             "visible": img.get_visible(),
#         }
# 
#     # Extract texts (excluding axis labels and title which are handled separately)
#     for idx, text in enumerate(ax.texts):
#         if not text.get_text():
#             continue
# 
#         elem_id = f"{ax_id}_text_{idx:02d}"
#         geom = extract_text_geometry(text, ax, fig)
# 
#         elements[elem_id] = {
#             "id": elem_id,
#             "axes_id": ax_id,
#             "element_type": "text",
#             "label": text.get_text()[:20],
#             "text": text.get_text(),
#             "geometry_px": geom,
#             "style": _extract_text_style(text),
#             "editable_styles": ["fontsize", "color", "fontweight", "alpha"],
#             "zorder": text.get_zorder(),
#             "visible": text.get_visible(),
#         }
# 
#     return elements
# 
# 
# def _extract_line_style(line) -> Dict[str, Any]:
#     """Extract style properties from a Line2D."""
#     color = line.get_color()
#     if isinstance(color, np.ndarray):
#         color = color.tolist()
# 
#     return {
#         "color": color,
#         "linewidth": line.get_linewidth(),
#         "linestyle": line.get_linestyle(),
#         "alpha": line.get_alpha(),
#         "marker": line.get_marker(),
#         "markersize": line.get_markersize(),
#     }
# 
# 
# def _extract_scatter_style(coll) -> Dict[str, Any]:
#     """Extract style properties from a PathCollection (scatter)."""
#     facecolors = coll.get_facecolors()
#     edgecolors = coll.get_edgecolors()
# 
#     fc = facecolors[0].tolist() if len(facecolors) > 0 else [0, 0, 0, 1]
#     ec = edgecolors[0].tolist() if len(edgecolors) > 0 else [0, 0, 0, 1]
# 
#     sizes = coll.get_sizes()
#     s = float(sizes[0]) if len(sizes) > 0 else 36
# 
#     return {
#         "facecolor": fc,
#         "edgecolor": ec,
#         "s": s,
#         "alpha": coll.get_alpha(),
#     }
# 
# 
# def _extract_polygon_style(coll) -> Dict[str, Any]:
#     """Extract style properties from a PolyCollection."""
#     facecolors = coll.get_facecolors()
#     edgecolors = coll.get_edgecolors()
# 
#     fc = facecolors[0].tolist() if len(facecolors) > 0 else [0, 0, 0, 0.3]
#     ec = edgecolors[0].tolist() if len(edgecolors) > 0 else [0, 0, 0, 1]
# 
#     return {
#         "facecolor": fc,
#         "edgecolor": ec,
#         "alpha": coll.get_alpha(),
#     }
# 
# 
# def _extract_bar_style(patch) -> Dict[str, Any]:
#     """Extract style properties from a Rectangle patch."""
#     if patch is None:
#         return {}
# 
#     fc = patch.get_facecolor()
#     ec = patch.get_edgecolor()
# 
#     return {
#         "facecolor": list(fc) if hasattr(fc, '__iter__') else fc,
#         "edgecolor": list(ec) if hasattr(ec, '__iter__') else ec,
#         "linewidth": patch.get_linewidth(),
#         "alpha": patch.get_alpha(),
#     }
# 
# 
# def _extract_text_style(text) -> Dict[str, Any]:
#     """Extract style properties from a Text object."""
#     return {
#         "fontsize": text.get_fontsize(),
#         "color": text.get_color(),
#         "fontweight": text.get_fontweight(),
#         "fontstyle": text.get_fontstyle(),
#         "fontfamily": text.get_fontfamily(),
#         "alpha": text.get_alpha(),
#         "ha": text.get_ha(),
#         "va": text.get_va(),
#         "rotation": text.get_rotation(),
#     }
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/metadata/_editable_export.py
# --------------------------------------------------------------------------------
