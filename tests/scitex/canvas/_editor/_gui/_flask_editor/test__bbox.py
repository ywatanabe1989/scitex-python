# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_editor/_gui/_flask_editor/_bbox.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # File: ./src/scitex/vis/editor/flask_editor/bbox.py
# """Bounding box extraction for figure elements.
# 
# Updated to integrate Schema v0.3 geometry extraction for shape-based hit testing.
# """
# 
# from typing import Any, Dict
# 
# # Try to import schema v0.3 geometry extraction
# try:
#     from scitex.plt.utils.metadata._geometry_extraction import (
#         extract_axes_bbox_px,
#         extract_bar_group_geometry,
#         extract_line_geometry,
#         extract_polygon_geometry,
#         extract_scatter_geometry,
#     )
# 
#     GEOMETRY_V03_AVAILABLE = True
# except ImportError:
#     GEOMETRY_V03_AVAILABLE = False
# 
# 
# def extract_bboxes(
#     fig, ax, renderer, img_width: int, img_height: int
# ) -> Dict[str, Any]:
#     """Extract bounding boxes for all figure elements (single-axis)."""
#     from matplotlib.transforms import Bbox
# 
#     # Get figure tight bbox in inches
#     fig_bbox = fig.get_tightbbox(renderer)
#     tight_x0 = fig_bbox.x0
#     tight_y0 = fig_bbox.y0
#     tight_width = fig_bbox.width
#     tight_height = fig_bbox.height
# 
#     # bbox_inches='tight' adds pad_inches (default 0.1) around the tight bbox
#     pad_inches = 0.1
#     saved_width_inches = tight_width + 2 * pad_inches
#     saved_height_inches = tight_height + 2 * pad_inches
# 
#     # Scale factors for converting inches to pixels
#     scale_x = img_width / saved_width_inches
#     scale_y = img_height / saved_height_inches
# 
#     bboxes = {}
# 
#     def get_element_bbox(element, name):
#         """Get element bbox in image pixel coordinates."""
#         try:
#             bbox = element.get_window_extent(renderer)
# 
#             elem_x0_inches = bbox.x0 / fig.dpi
#             elem_x1_inches = bbox.x1 / fig.dpi
#             elem_y0_inches = bbox.y0 / fig.dpi
#             elem_y1_inches = bbox.y1 / fig.dpi
# 
#             x0_rel = elem_x0_inches - tight_x0 + pad_inches
#             x1_rel = elem_x1_inches - tight_x0 + pad_inches
#             y0_rel = saved_height_inches - (elem_y1_inches - tight_y0 + pad_inches)
#             y1_rel = saved_height_inches - (elem_y0_inches - tight_y0 + pad_inches)
# 
#             bboxes[name] = {
#                 "x0": max(0, int(x0_rel * scale_x)),
#                 "y0": max(0, int(y0_rel * scale_y)),
#                 "x1": min(img_width, int(x1_rel * scale_x)),
#                 "y1": min(img_height, int(y1_rel * scale_y)),
#                 "label": name.replace("_", " ").title(),
#             }
#         except Exception as e:
#             print(f"Error getting bbox for {name}: {e}")
# 
#     def bbox_to_img_coords(bbox):
#         """Convert matplotlib bbox to image pixel coordinates."""
#         x0_inches = bbox.x0 / fig.dpi
#         y0_inches = bbox.y0 / fig.dpi
#         x1_inches = bbox.x1 / fig.dpi
#         y1_inches = bbox.y1 / fig.dpi
#         x0_rel = x0_inches - tight_x0 + pad_inches
#         y0_rel = y0_inches - tight_y0 + pad_inches
#         x1_rel = x1_inches - tight_x0 + pad_inches
#         y1_rel = y1_inches - tight_y0 + pad_inches
#         return {
#             "x0": int(x0_rel * scale_x),
#             "y0": int((saved_height_inches - y1_rel) * scale_y),
#             "x1": int(x1_rel * scale_x),
#             "y1": int((saved_height_inches - y0_rel) * scale_y),
#         }
# 
#     # Get bboxes for title, labels
#     # Use ax_00_ prefix for consistency with geometry_px.json format
#     ax_prefix = "ax_00_"
#     if ax.title.get_text():
#         get_element_bbox(ax.title, f"{ax_prefix}title")
#     if ax.xaxis.label.get_text():
#         get_element_bbox(ax.xaxis.label, f"{ax_prefix}xlabel")
#     if ax.yaxis.label.get_text():
#         get_element_bbox(ax.yaxis.label, f"{ax_prefix}ylabel")
# 
#     # Get axis bboxes
#     _extract_axis_bboxes(ax, renderer, bboxes, bbox_to_img_coords, Bbox, ax_prefix)
# 
#     # Get legend bbox
#     legend = ax.get_legend()
#     if legend:
#         get_element_bbox(legend, f"{ax_prefix}legend")
# 
#     # Get caption bbox (figure-level text)
#     for text_artist in fig.texts:
#         # Check if this is a caption (positioned below figure)
#         text_content = text_artist.get_text()
#         if text_content:
#             pos = text_artist.get_position()
#             # Caption is typically centered horizontally (x ~= 0.5) and below figure (y < 0.1)
#             if pos[1] < 0.1:
#                 get_element_bbox(text_artist, f"{ax_prefix}caption")
#                 break  # Only one caption expected
# 
#     # Get trace (line) bboxes
#     _extract_trace_bboxes(
#         ax,
#         fig,
#         renderer,
#         bboxes,
#         get_element_bbox,
#         tight_x0,
#         tight_y0,
#         saved_height_inches,
#         scale_x,
#         scale_y,
#         pad_inches,
#     )
# 
#     # Add schema v0.3 metadata if available
#     if GEOMETRY_V03_AVAILABLE:
#         axes_bbox = extract_axes_bbox_px(ax, fig)
#         bboxes["_meta"] = {
#             "schema_version": "0.3.0",
#             "axes_bbox_px": axes_bbox,
#             "geometry_available": True,
#         }
# 
#     return bboxes
# 
# 
# def extract_bboxes_multi(
#     fig, axes_map: Dict[str, Any], renderer, img_width: int, img_height: int
# ) -> Dict[str, Any]:
#     """Extract bounding boxes for all elements in a multi-axis figure.
# 
#     Args:
#         fig: Matplotlib figure
#         axes_map: Dict mapping axis IDs (e.g., 'ax_00') to matplotlib Axes objects
#         renderer: Matplotlib renderer
#         img_width: Image width in pixels
#         img_height: Image height in pixels
# 
#     Returns:
#         Dict with bboxes keyed by "{ax_id}_{element_type}" (e.g., "ax_00_xlabel")
#     """
#     from matplotlib.transforms import Bbox
# 
#     # Get figure tight bbox in inches
#     fig_bbox = fig.get_tightbbox(renderer)
#     tight_x0 = fig_bbox.x0
#     tight_y0 = fig_bbox.y0
#     tight_width = fig_bbox.width
#     tight_height = fig_bbox.height
# 
#     # bbox_inches='tight' adds pad_inches (default 0.1) around the tight bbox
#     pad_inches = 0.1
#     saved_width_inches = tight_width + 2 * pad_inches
#     saved_height_inches = tight_height + 2 * pad_inches
# 
#     # Scale factors for converting inches to pixels
#     scale_x = img_width / saved_width_inches
#     scale_y = img_height / saved_height_inches
# 
#     bboxes = {}
# 
#     def get_element_bbox(element, name, ax_id, current_ax=None):
#         """Get element bbox in image pixel coordinates."""
#         import numpy as np
# 
#         full_name = f"{ax_id}_{name}"
#         try:
#             bbox = element.get_window_extent(renderer)
# 
#             # Check for invalid bbox (infinity or NaN)
#             if not (
#                 np.isfinite(bbox.x0)
#                 and np.isfinite(bbox.x1)
#                 and np.isfinite(bbox.y0)
#                 and np.isfinite(bbox.y1)
#             ):
#                 # Try to get bbox from data for scatter/collection elements
#                 if hasattr(element, "get_offsets") and current_ax is not None:
#                     offsets = element.get_offsets()
#                     if len(offsets) > 0 and np.isfinite(offsets).all():
#                         # Use axis transform to get display coordinates
#                         display_coords = current_ax.transData.transform(offsets)
#                         x0 = display_coords[:, 0].min()
#                         x1 = display_coords[:, 0].max()
#                         y0 = display_coords[:, 1].min()
#                         y1 = display_coords[:, 1].max()
#                         if np.isfinite([x0, x1, y0, y1]).all():
#                             from matplotlib.transforms import Bbox
# 
#                             bbox = Bbox.from_extents(x0, y0, x1, y1)
#                         else:
#                             return  # Skip this element
#                     else:
#                         return  # Skip this element
#                 else:
#                     return  # Skip this element
# 
#             elem_x0_inches = bbox.x0 / fig.dpi
#             elem_x1_inches = bbox.x1 / fig.dpi
#             elem_y0_inches = bbox.y0 / fig.dpi
#             elem_y1_inches = bbox.y1 / fig.dpi
# 
#             x0_rel = elem_x0_inches - tight_x0 + pad_inches
#             x1_rel = elem_x1_inches - tight_x0 + pad_inches
#             y0_rel = saved_height_inches - (elem_y1_inches - tight_y0 + pad_inches)
#             y1_rel = saved_height_inches - (elem_y0_inches - tight_y0 + pad_inches)
# 
#             # Clamp values to avoid overflow
#             x0_px = max(0, min(img_width, int(x0_rel * scale_x)))
#             y0_px = max(0, min(img_height, int(y0_rel * scale_y)))
#             x1_px = max(0, min(img_width, int(x1_rel * scale_x)))
#             y1_px = max(0, min(img_height, int(y1_rel * scale_y)))
# 
#             bboxes[full_name] = {
#                 "x0": x0_px,
#                 "y0": y0_px,
#                 "x1": x1_px,
#                 "y1": y1_px,
#                 "label": f"{ax_id}: {name.replace('_', ' ').title()}",
#                 "ax_id": ax_id,
#             }
#         except Exception as e:
#             print(f"Error getting bbox for {full_name}: {e}")
# 
#     def bbox_to_img_coords(bbox):
#         """Convert matplotlib bbox to image pixel coordinates."""
#         x0_inches = bbox.x0 / fig.dpi
#         y0_inches = bbox.y0 / fig.dpi
#         x1_inches = bbox.x1 / fig.dpi
#         y1_inches = bbox.y1 / fig.dpi
#         x0_rel = x0_inches - tight_x0 + pad_inches
#         y0_rel = y0_inches - tight_y0 + pad_inches
#         x1_rel = x1_inches - tight_x0 + pad_inches
#         y1_rel = y1_inches - tight_y0 + pad_inches
#         return {
#             "x0": int(x0_rel * scale_x),
#             "y0": int((saved_height_inches - y1_rel) * scale_y),
#             "x1": int(x1_rel * scale_x),
#             "y1": int((saved_height_inches - y0_rel) * scale_y),
#         }
# 
#     # Extract bboxes for each axis
#     for ax_id, ax in axes_map.items():
#         # Get axes bounding box (the entire panel area)
#         try:
#             ax_bbox = ax.get_window_extent(renderer)
#             coords = bbox_to_img_coords(ax_bbox)
#             # Extract actual title/labels from the axes
#             title_text = ax.title.get_text() if ax.title else ""
#             xlabel_text = ax.xaxis.label.get_text() if ax.xaxis.label else ""
#             ylabel_text = ax.yaxis.label.get_text() if ax.yaxis.label else ""
#             bboxes[f"{ax_id}_panel"] = {
#                 **coords,
#                 "label": f"Panel {ax_id}",
#                 "ax_id": ax_id,
#                 "is_panel": True,
#                 "title": title_text,
#                 "xlabel": xlabel_text,
#                 "ylabel": ylabel_text,
#             }
#         except Exception as e:
#             print(f"Error getting panel bbox for {ax_id}: {e}")
# 
#         # Get bboxes for title, labels
#         if ax.title.get_text():
#             get_element_bbox(ax.title, "title", ax_id, ax)
#         if ax.xaxis.label.get_text():
#             get_element_bbox(ax.xaxis.label, "xlabel", ax_id, ax)
#         if ax.yaxis.label.get_text():
#             get_element_bbox(ax.yaxis.label, "ylabel", ax_id, ax)
# 
#         # Get X-axis bbox (spine + ticks + ticklabels)
#         _extract_axis_bboxes_for_axis(
#             ax, ax_id, renderer, bboxes, bbox_to_img_coords, Bbox
#         )
# 
#         # Get legend bbox
#         legend = ax.get_legend()
#         if legend:
#             get_element_bbox(legend, "legend", ax_id, ax)
#             # Add element_type for drag detection
#             if f"{ax_id}_legend" in bboxes:
#                 bboxes[f"{ax_id}_legend"]["element_type"] = "legend"
#                 bboxes[f"{ax_id}_legend"]["draggable"] = True
# 
#         # Get panel letter (text annotations like A, B, C)
#         import re
# 
#         panel_letter_pattern = re.compile(r"^[A-Z]\.?$|^\([A-Za-z]\)$")
#         for idx, text_artist in enumerate(ax.texts):
#             text_content = text_artist.get_text().strip()
#             if text_content and panel_letter_pattern.match(text_content):
#                 name = f"panel_letter_{text_content.replace('.', '').replace('(', '').replace(')', '')}"
#                 get_element_bbox(text_artist, name, ax_id, ax)
#                 full_name = f"{ax_id}_{name}"
#                 if full_name in bboxes:
#                     bboxes[full_name]["element_type"] = "panel_letter"
#                     bboxes[full_name]["draggable"] = True
#                     bboxes[full_name]["text"] = text_content
#                     # Get position in axes coordinates (0-1)
#                     pos = text_artist.get_position()
#                     transform = text_artist.get_transform()
#                     if transform == ax.transAxes:
#                         bboxes[full_name]["axes_position"] = {"x": pos[0], "y": pos[1]}
# 
#         # Get trace (line) bboxes
#         _extract_trace_bboxes_for_axis(
#             ax,
#             ax_id,
#             fig,
#             renderer,
#             bboxes,
#             get_element_bbox,
#             tight_x0,
#             tight_y0,
#             saved_height_inches,
#             scale_x,
#             scale_y,
#             pad_inches,
#         )
# 
#     # Get caption bbox (figure-level text)
#     # This is outside the per-axis loop since caption is a figure-level element
#     for text_artist in fig.texts:
#         text_content = text_artist.get_text()
#         if text_content:
#             pos = text_artist.get_position()
#             # Caption is typically centered horizontally (x ~= 0.5) and below figure (y < 0.1)
#             if pos[1] < 0.1:
#                 try:
#                     bbox = text_artist.get_window_extent(renderer)
#                     x0_inches = bbox.x0 / fig.dpi
#                     x1_inches = bbox.x1 / fig.dpi
#                     y0_inches = bbox.y0 / fig.dpi
#                     y1_inches = bbox.y1 / fig.dpi
#                     x0_rel = x0_inches - tight_x0 + pad_inches
#                     x1_rel = x1_inches - tight_x0 + pad_inches
#                     y0_rel = saved_height_inches - (y1_inches - tight_y0 + pad_inches)
#                     y1_rel = saved_height_inches - (y0_inches - tight_y0 + pad_inches)
#                     bboxes["caption"] = {
#                         "x0": max(0, int(x0_rel * scale_x)),
#                         "y0": max(0, int(y0_rel * scale_y)),
#                         "x1": min(img_width, int(x1_rel * scale_x)),
#                         "y1": min(img_height, int(y1_rel * scale_y)),
#                         "label": "Caption",
#                     }
#                 except Exception as e:
#                     print(f"Error getting caption bbox: {e}")
#                 break  # Only one caption expected
# 
#     # Add schema v0.3 metadata if available
#     if GEOMETRY_V03_AVAILABLE:
#         axes_bboxes = {}
#         for ax_id, ax in axes_map.items():
#             axes_bboxes[ax_id] = extract_axes_bbox_px(ax, fig)
#         bboxes["_meta"] = {
#             "schema_version": "0.3.0",
#             "axes": axes_bboxes,
#             "geometry_available": True,
#         }
# 
#     return bboxes
# 
# 
# def _extract_trace_bboxes_for_axis(
#     ax,
#     ax_id,
#     fig,
#     renderer,
#     bboxes,
#     get_element_bbox,
#     tight_x0,
#     tight_y0,
#     saved_height_inches,
#     scale_x,
#     scale_y,
#     pad_inches,
# ):
#     """Extract bboxes for all data elements in a specific axis.
# 
#     Handles:
#     - Lines (plot, errorbar lines)
#     - Scatter points (PathCollection)
#     - Fill areas (PolyCollection from fill_between)
#     - Bars (Rectangle patches)
#     """
#     import numpy as np
# 
#     def coords_to_img_points(data_coords):
#         """Convert data coordinates to image pixel coordinates."""
#         if len(data_coords) == 0:
#             return []
#         transform = ax.transData
#         points_display = transform.transform(data_coords)
#         points_img = []
#         for px, py in points_display:
#             # Skip invalid points (NaN, infinity)
#             if not np.isfinite(px) or not np.isfinite(py):
#                 continue
#             px_inches = px / fig.dpi
#             py_inches = py / fig.dpi
#             x_rel = px_inches - tight_x0 + pad_inches
#             y_rel = saved_height_inches - (py_inches - tight_y0 + pad_inches)
#             # Clamp to reasonable bounds to avoid overflow
#             x_img = max(-10000, min(10000, int(x_rel * scale_x)))
#             y_img = max(-10000, min(10000, int(y_rel * scale_y)))
#             points_img.append([x_img, y_img])
#         # Downsample if too many
#         if len(points_img) > 100:
#             step = len(points_img) // 100
#             points_img = points_img[::step]
#         return points_img
# 
#     # 1. Extract lines (plot, errorbar lines, etc.)
#     line_idx = 0
#     for line in ax.get_lines():
#         try:
#             label = line.get_label()
#             # Include unlabeled lines but mark them appropriately
#             if label is None or label.startswith("_"):
#                 label = None  # Will use generic name
# 
#             trace_name = f"trace_{line_idx}"
#             full_name = f"{ax_id}_{trace_name}"
#             get_element_bbox(line, trace_name, ax_id, ax)
# 
#             if full_name in bboxes:
#                 bboxes[full_name]["label"] = f"{ax_id}: {label or f'Line {line_idx}'}"
#                 bboxes[full_name]["trace_idx"] = line_idx
#                 bboxes[full_name]["element_type"] = "line"
# 
#                 xdata, ydata = line.get_xdata(), line.get_ydata()
#                 if len(xdata) > 0:
#                     bboxes[full_name]["points"] = coords_to_img_points(
#                         list(zip(xdata, ydata))
#                     )
# 
#                 # Add schema v0.3 geometry_px if available
#                 if GEOMETRY_V03_AVAILABLE:
#                     try:
#                         geom = extract_line_geometry(line, ax, fig)
#                         bboxes[full_name]["geometry_px"] = geom
#                     except Exception:
#                         pass  # Fall back to legacy points
# 
#             line_idx += 1
#         except Exception as e:
#             print(f"Error getting line bbox for {ax_id}: {e}")
# 
#     # 2. Extract collections (scatter, fill_between, etc.)
#     coll_idx = 0
#     for coll in ax.collections:
#         try:
#             label = coll.get_label()
#             if label is None or label.startswith("_"):
#                 # Still extract unlabeled collections but with generic name
#                 label = None
# 
#             coll_type = type(coll).__name__
#             if coll_type == "PathCollection":
#                 # Scatter points
#                 element_name = f"scatter_{coll_idx}"
#                 full_name = f"{ax_id}_{element_name}"
#                 get_element_bbox(coll, element_name, ax_id, ax)
# 
#                 if full_name in bboxes:
#                     bboxes[full_name][
#                         "label"
#                     ] = f"{ax_id}: {label or f'Scatter {coll_idx}'}"
#                     bboxes[full_name]["element_type"] = "scatter"
# 
#                     # Get scatter point positions
#                     offsets = coll.get_offsets()
#                     if len(offsets) > 0:
#                         bboxes[full_name]["points"] = coords_to_img_points(offsets)
# 
#                     # Add schema v0.3 geometry_px if available
#                     if GEOMETRY_V03_AVAILABLE:
#                         try:
#                             geom = extract_scatter_geometry(coll, ax, fig)
#                             bboxes[full_name]["geometry_px"] = geom
#                         except Exception:
#                             pass  # Fall back to legacy points
# 
#             elif coll_type in ("PolyCollection", "FillBetweenPolyCollection"):
#                 # Fill areas (fill_between, etc.)
#                 element_name = f"fill_{coll_idx}"
#                 full_name = f"{ax_id}_{element_name}"
#                 get_element_bbox(coll, element_name, ax_id, ax)
# 
#                 if full_name in bboxes:
#                     bboxes[full_name][
#                         "label"
#                     ] = f"{ax_id}: {label or f'Fill {coll_idx}'}"
#                     bboxes[full_name]["element_type"] = "fill"
# 
#                     # Add schema v0.3 geometry_px if available
#                     if GEOMETRY_V03_AVAILABLE:
#                         try:
#                             geom = extract_polygon_geometry(coll, ax, fig)
#                             bboxes[full_name]["geometry_px"] = geom
#                         except Exception:
#                             pass
# 
#             coll_idx += 1
#         except Exception as e:
#             print(f"Error getting collection bbox for {ax_id}: {e}")
# 
#     # 3. Extract patches (bars, rectangles, etc.)
#     patch_idx = 0
#     for patch in ax.patches:
#         try:
#             label = patch.get_label()
#             patch_type = type(patch).__name__
# 
#             if patch_type == "Rectangle":
#                 # Bar chart bars
#                 element_name = f"bar_{patch_idx}"
#                 full_name = f"{ax_id}_{element_name}"
#                 get_element_bbox(patch, element_name, ax_id, ax)
# 
#                 if full_name in bboxes:
#                     bboxes[full_name][
#                         "label"
#                     ] = f"{ax_id}: {label or f'Bar {patch_idx}'}"
#                     bboxes[full_name]["element_type"] = "bar"
# 
#             patch_idx += 1
#         except Exception as e:
#             print(f"Error getting patch bbox for {ax_id}: {e}")
# 
# 
# def _extract_axis_bboxes_for_axis(
#     ax, ax_id, renderer, bboxes, bbox_to_img_coords, Bbox
# ):
#     """Extract X and Y axis bboxes for a specific axis (multi-axis version)."""
#     try:
#         # X-axis: combine spine and tick labels into one bbox
#         x_axis_bboxes = []
#         for ticklabel in ax.xaxis.get_ticklabels():
#             if ticklabel.get_visible():
#                 try:
#                     tb = ticklabel.get_window_extent(renderer)
#                     if tb.width > 0:
#                         x_axis_bboxes.append(tb)
#                 except Exception:
#                     pass
#         for tick in ax.xaxis.get_major_ticks():
#             if tick.tick1line.get_visible():
#                 try:
#                     tb = tick.tick1line.get_window_extent(renderer)
#                     if tb.width > 0 or tb.height > 0:
#                         x_axis_bboxes.append(tb)
#                 except Exception:
#                     pass
#         spine_bbox = ax.spines["bottom"].get_window_extent(renderer)
#         if spine_bbox.width > 0:
#             if x_axis_bboxes:
#                 tick_union = Bbox.union(x_axis_bboxes)
#                 constrained_spine = Bbox.from_extents(
#                     tick_union.x0, spine_bbox.y0, tick_union.x1, spine_bbox.y1
#                 )
#                 x_axis_bboxes.append(constrained_spine)
#             else:
#                 x_axis_bboxes.append(spine_bbox)
#         if x_axis_bboxes:
#             combined = Bbox.union(x_axis_bboxes)
#             bboxes[f"{ax_id}_xaxis"] = bbox_to_img_coords(combined)
#             bboxes[f"{ax_id}_xaxis"]["label"] = f"{ax_id}: X Axis"
#             bboxes[f"{ax_id}_xaxis"]["ax_id"] = ax_id
#             bboxes[f"{ax_id}_xaxis"]["element_type"] = "xaxis"
# 
#         # Y-axis: combine spine and tick labels into one bbox
#         y_axis_bboxes = []
#         for ticklabel in ax.yaxis.get_ticklabels():
#             if ticklabel.get_visible():
#                 try:
#                     tb = ticklabel.get_window_extent(renderer)
#                     if tb.width > 0:
#                         y_axis_bboxes.append(tb)
#                 except Exception:
#                     pass
#         for tick in ax.yaxis.get_major_ticks():
#             if tick.tick1line.get_visible():
#                 try:
#                     tb = tick.tick1line.get_window_extent(renderer)
#                     if tb.width > 0 or tb.height > 0:
#                         y_axis_bboxes.append(tb)
#                 except Exception:
#                     pass
#         spine_bbox = ax.spines["left"].get_window_extent(renderer)
#         if spine_bbox.height > 0:
#             if y_axis_bboxes:
#                 tick_union = Bbox.union(y_axis_bboxes)
#                 constrained_spine = Bbox.from_extents(
#                     spine_bbox.x0, tick_union.y0, spine_bbox.x1, tick_union.y1
#                 )
#                 y_axis_bboxes.append(constrained_spine)
#             else:
#                 y_axis_bboxes.append(spine_bbox)
#         if y_axis_bboxes:
#             combined = Bbox.union(y_axis_bboxes)
#             padded = Bbox.from_extents(
#                 combined.x0 - 10, combined.y0 - 5, combined.x1 + 5, combined.y1 + 5
#             )
#             bboxes[f"{ax_id}_yaxis"] = bbox_to_img_coords(padded)
#             bboxes[f"{ax_id}_yaxis"]["label"] = f"{ax_id}: Y Axis"
#             bboxes[f"{ax_id}_yaxis"]["ax_id"] = ax_id
#             bboxes[f"{ax_id}_yaxis"]["element_type"] = "yaxis"
# 
#     except Exception as e:
#         print(f"Error getting axis bboxes for {ax_id}: {e}")
# 
# 
# def _extract_axis_bboxes(ax, renderer, bboxes, bbox_to_img_coords, Bbox, ax_prefix=""):
#     """Extract bboxes for X and Y axis elements.
# 
#     Args:
#         ax: Matplotlib axis.
#         renderer: Figure renderer.
#         bboxes: Dict to store bboxes.
#         bbox_to_img_coords: Coordinate conversion function.
#         Bbox: Matplotlib Bbox class.
#         ax_prefix: Prefix for bbox names (e.g., "ax_00_").
#     """
#     try:
#         # X-axis: combine spine and tick labels into one bbox
#         x_axis_bboxes = []
#         for ticklabel in ax.xaxis.get_ticklabels():
#             if ticklabel.get_visible():
#                 try:
#                     tb = ticklabel.get_window_extent(renderer)
#                     if tb.width > 0:
#                         x_axis_bboxes.append(tb)
#                 except Exception:
#                     pass
#         for tick in ax.xaxis.get_major_ticks():
#             if tick.tick1line.get_visible():
#                 try:
#                     tb = tick.tick1line.get_window_extent(renderer)
#                     if tb.width > 0 or tb.height > 0:
#                         x_axis_bboxes.append(tb)
#                 except Exception:
#                     pass
#         spine_bbox = ax.spines["bottom"].get_window_extent(renderer)
#         if spine_bbox.width > 0:
#             if x_axis_bboxes:
#                 tick_union = Bbox.union(x_axis_bboxes)
#                 constrained_spine = Bbox.from_extents(
#                     tick_union.x0, spine_bbox.y0, tick_union.x1, spine_bbox.y1
#                 )
#                 x_axis_bboxes.append(constrained_spine)
#             else:
#                 x_axis_bboxes.append(spine_bbox)
#         if x_axis_bboxes:
#             combined = Bbox.union(x_axis_bboxes)
#             bboxes[f"{ax_prefix}xaxis_spine"] = bbox_to_img_coords(combined)
#             bboxes[f"{ax_prefix}xaxis_spine"]["label"] = "X Spine & Ticks"
# 
#         # Y-axis: combine spine and tick labels into one bbox
#         y_axis_bboxes = []
#         for ticklabel in ax.yaxis.get_ticklabels():
#             if ticklabel.get_visible():
#                 try:
#                     tb = ticklabel.get_window_extent(renderer)
#                     if tb.width > 0:
#                         y_axis_bboxes.append(tb)
#                 except Exception:
#                     pass
#         for tick in ax.yaxis.get_major_ticks():
#             if tick.tick1line.get_visible():
#                 try:
#                     tb = tick.tick1line.get_window_extent(renderer)
#                     if tb.width > 0 or tb.height > 0:
#                         y_axis_bboxes.append(tb)
#                 except Exception:
#                     pass
#         spine_bbox = ax.spines["left"].get_window_extent(renderer)
#         if spine_bbox.height > 0:
#             if y_axis_bboxes:
#                 tick_union = Bbox.union(y_axis_bboxes)
#                 constrained_spine = Bbox.from_extents(
#                     spine_bbox.x0, tick_union.y0, spine_bbox.x1, tick_union.y1
#                 )
#                 y_axis_bboxes.append(constrained_spine)
#             else:
#                 y_axis_bboxes.append(spine_bbox)
#         if y_axis_bboxes:
#             combined = Bbox.union(y_axis_bboxes)
#             padded = Bbox.from_extents(
#                 combined.x0 - 10, combined.y0 - 5, combined.x1 + 5, combined.y1 + 5
#             )
#             bboxes[f"{ax_prefix}yaxis_spine"] = bbox_to_img_coords(padded)
#             bboxes[f"{ax_prefix}yaxis_spine"]["label"] = "Y Spine & Ticks"
# 
#     except Exception as e:
#         print(f"Error getting axis bboxes: {e}")
# 
# 
# def _extract_trace_bboxes(
#     ax,
#     fig,
#     renderer,
#     bboxes,
#     get_element_bbox,
#     tight_x0,
#     tight_y0,
#     saved_height_inches,
#     scale_x,
#     scale_y,
#     pad_inches,
# ):
#     """Extract bboxes for all data elements (lines, scatter, fill) with proximity detection."""
#     import numpy as np
# 
#     def coords_to_img_points(data_coords):
#         """Convert data coordinates to image pixel coordinates."""
#         if len(data_coords) == 0:
#             return []
#         transform = ax.transData
#         points_display = transform.transform(data_coords)
#         points_img = []
#         for px, py in points_display:
#             if not np.isfinite(px) or not np.isfinite(py):
#                 continue
#             px_inches = px / fig.dpi
#             py_inches = py / fig.dpi
#             x_rel = px_inches - tight_x0 + pad_inches
#             y_rel = saved_height_inches - (py_inches - tight_y0 + pad_inches)
#             x_img = max(-10000, min(10000, int(x_rel * scale_x)))
#             y_img = max(-10000, min(10000, int(y_rel * scale_y)))
#             points_img.append([x_img, y_img])
#         if len(points_img) > 100:
#             step = len(points_img) // 100
#             points_img = points_img[::step]
#         return points_img
# 
#     # 1. Extract lines
#     for idx, line in enumerate(ax.get_lines()):
#         try:
#             label = line.get_label()
#             # Include unlabeled lines but mark them appropriately
#             if label is None or label.startswith("_"):
#                 label = None  # Will use generic name
#             get_element_bbox(line, f"trace_{idx}")
#             if f"trace_{idx}" in bboxes:
#                 bboxes[f"trace_{idx}"]["label"] = label or f"Trace {idx}"
#                 bboxes[f"trace_{idx}"]["trace_idx"] = idx
#                 bboxes[f"trace_{idx}"]["element_type"] = "line"
# 
#                 xdata, ydata = line.get_xdata(), line.get_ydata()
#                 if len(xdata) > 0:
#                     bboxes[f"trace_{idx}"]["points"] = coords_to_img_points(
#                         list(zip(xdata, ydata))
#                     )
# 
#                 # Add schema v0.3 geometry_px if available
#                 if GEOMETRY_V03_AVAILABLE:
#                     try:
#                         geom = extract_line_geometry(line, ax, fig)
#                         bboxes[f"trace_{idx}"]["geometry_px"] = geom
#                     except Exception:
#                         pass
#         except Exception as e:
#             print(f"Error getting trace bbox: {e}")
# 
#     # 2. Extract collections (scatter, fill_between)
#     coll_idx = 0
#     for coll in ax.collections:
#         try:
#             label = coll.get_label()
#             if label is None or label.startswith("_"):
#                 label = None
# 
#             coll_type = type(coll).__name__
#             if coll_type == "PathCollection":
#                 # Scatter points
#                 elem_key = f"scatter_{coll_idx}"
#                 get_element_bbox(coll, elem_key)
# 
#                 # Initialize entry if bbox extraction failed but we have data
#                 offsets = coll.get_offsets()
#                 if elem_key not in bboxes and len(offsets) > 0:
#                     # Create bbox from data coordinates as fallback
#                     points_img = coords_to_img_points(offsets)
#                     if points_img:
#                         xs = [p[0] for p in points_img]
#                         ys = [p[1] for p in points_img]
#                         bboxes[elem_key] = {
#                             "x0": min(xs) - 10,
#                             "y0": min(ys) - 10,
#                             "x1": max(xs) + 10,
#                             "y1": max(ys) + 10,
#                         }
# 
#                 if elem_key in bboxes:
#                     bboxes[elem_key]["label"] = label or f"Scatter {coll_idx}"
#                     bboxes[elem_key]["element_type"] = "scatter"
# 
#                     if len(offsets) > 0:
#                         bboxes[elem_key]["points"] = coords_to_img_points(offsets)
# 
#                     # Add schema v0.3 geometry_px if available
#                     if GEOMETRY_V03_AVAILABLE:
#                         try:
#                             geom = extract_scatter_geometry(coll, ax, fig)
#                             bboxes[elem_key]["geometry_px"] = geom
#                         except Exception:
#                             pass
# 
#             elif coll_type in ("PolyCollection", "FillBetweenPolyCollection"):
#                 # Fill areas
#                 get_element_bbox(coll, f"fill_{coll_idx}")
#                 if f"fill_{coll_idx}" in bboxes:
#                     bboxes[f"fill_{coll_idx}"]["label"] = label or f"Fill {coll_idx}"
#                     bboxes[f"fill_{coll_idx}"]["element_type"] = "fill"
# 
#                     # Add schema v0.3 geometry_px if available
#                     if GEOMETRY_V03_AVAILABLE:
#                         try:
#                             geom = extract_polygon_geometry(coll, ax, fig)
#                             bboxes[f"fill_{coll_idx}"]["geometry_px"] = geom
#                         except Exception:
#                             pass
# 
#             coll_idx += 1
#         except Exception as e:
#             print(f"Error getting collection bbox: {e}")
# 
# 
# def extract_bboxes_from_metadata(
#     metadata: Dict[str, Any],
#     img_width: int,
#     img_height: int,
# ) -> Dict[str, Any]:
#     """Extract bounding boxes from pre-computed metadata (without re-rendering).
# 
#     This is used when loading actual PNGs from bundles instead of re-rendering.
#     Extracts bbox info from:
#     - hit_regions (if available from v0.3 schema)
#     - elements dict
#     - axes positions
# 
#     Args:
#         metadata: JSON metadata from spec.json or panel JSON
#         img_width: Image width in pixels
#         img_height: Image height in pixels
# 
#     Returns:
#         Dict with bboxes keyed by element name
#     """
#     bboxes = {}
# 
#     # Check for pre-computed hit_regions (v0.3 schema)
#     hit_regions = metadata.get("hit_regions", {})
#     if hit_regions:
#         color_map = hit_regions.get("color_map", {})
#         for element_name, color in color_map.items():
#             # We don't have exact coords from color map, but we can create placeholder
#             bboxes[element_name] = {
#                 "label": element_name.replace("_", " ").title(),
#                 "element_type": _guess_element_type(element_name),
#             }
# 
#     # Check for geometry_px in cache (v0.3 layered bundle)
#     geometry_px = metadata.get("geometry_px", {})
#     if geometry_px:
#         for element_name, geom in geometry_px.items():
#             if isinstance(geom, dict) and "bbox" in geom:
#                 bbox = geom["bbox"]
#                 bboxes[element_name] = {
#                     "x0": bbox.get("x0", 0),
#                     "y0": bbox.get("y0", 0),
#                     "x1": bbox.get("x1", img_width),
#                     "y1": bbox.get("y1", img_height),
#                     "label": element_name.replace("_", " ").title(),
#                     "element_type": _guess_element_type(element_name),
#                 }
#                 if "points" in geom:
#                     bboxes[element_name]["points"] = geom["points"]
# 
#     # Extract from elements dict if present
#     elements = metadata.get("elements", {})
#     if not isinstance(elements, dict):
#         elements = {}
#     for element_name, element_info in elements.items():
#         if not isinstance(element_info, dict):
#             continue
#         if element_name not in bboxes:
#             bboxes[element_name] = {
#                 "label": element_info.get(
#                     "label", element_name.replace("_", " ").title()
#                 ),
#                 "element_type": element_info.get(
#                     "type", _guess_element_type(element_name)
#                 ),
#             }
# 
#     # Extract from axes (handle both dict and list formats)
#     axes = metadata.get("axes", [])
#     if isinstance(axes, list):
#         axes_list = axes
#     elif isinstance(axes, dict):
#         axes_list = list(axes.values())
#     else:
#         axes_list = []
# 
#     for i, ax_spec in enumerate(axes_list):
#         if not isinstance(ax_spec, dict):
#             continue
# 
#         ax_id = ax_spec.get("id", f"ax{i}")
# 
#         # Panel bbox - check for "bbox" field (new format) or "position" (old format)
#         bbox_spec = ax_spec.get("bbox", {})
#         pos = ax_spec.get("position", [])
# 
#         if bbox_spec and isinstance(bbox_spec, dict):
#             # New format: bbox with x0, y0, width, height in panel fraction
#             x0_frac = bbox_spec.get("x0", 0)
#             y0_frac = bbox_spec.get("y0", 0)
#             w_frac = bbox_spec.get("width", 1)
#             h_frac = bbox_spec.get("height", 1)
#             x0 = int(x0_frac * img_width)
#             y0 = int(y0_frac * img_height)
#             x1 = int((x0_frac + w_frac) * img_width)
#             y1 = int((y0_frac + h_frac) * img_height)
#             bboxes[f"{ax_id}_panel"] = {
#                 "x0": x0,
#                 "y0": y0,
#                 "x1": x1,
#                 "y1": y1,
#                 "label": f"Panel {ax_id}",
#                 "ax_id": ax_id,
#                 "is_panel": True,
#             }
#         elif len(pos) >= 4:
#             # Old format: position is in figure fraction [x0, y0, width, height]
#             x0 = int(pos[0] * img_width)
#             y0 = int((1 - pos[1] - pos[3]) * img_height)  # Flip Y
#             x1 = int((pos[0] + pos[2]) * img_width)
#             y1 = int((1 - pos[1]) * img_height)  # Flip Y
#             bboxes[f"{ax_id}_panel"] = {
#                 "x0": x0,
#                 "y0": y0,
#                 "x1": x1,
#                 "y1": y1,
#                 "label": f"Panel {ax_id}",
#                 "ax_id": ax_id,
#                 "is_panel": True,
#             }
# 
#         # Title/labels from labels dict (new format) or xaxis/yaxis (old format)
#         labels = ax_spec.get("labels", {})
#         xaxis = ax_spec.get("xaxis", {})
#         yaxis = ax_spec.get("yaxis", {})
# 
#         xlabel = labels.get("xlabel") or (
#             xaxis.get("label") if isinstance(xaxis, dict) else None
#         )
#         ylabel = labels.get("ylabel") or (
#             yaxis.get("label") if isinstance(yaxis, dict) else None
#         )
#         title = labels.get("title")
# 
#         if xlabel:
#             bboxes[f"{ax_id}_xlabel"] = {
#                 "label": f"{ax_id}: {xlabel}",
#                 "element_type": "xlabel",
#                 "ax_id": ax_id,
#             }
#         if ylabel:
#             bboxes[f"{ax_id}_ylabel"] = {
#                 "label": f"{ax_id}: {ylabel}",
#                 "element_type": "ylabel",
#                 "ax_id": ax_id,
#             }
#         if title:
#             bboxes[f"{ax_id}_title"] = {
#                 "label": title,
#                 "element_type": "title",
#                 "ax_id": ax_id,
#             }
# 
#     # Extract from traces array (pltz spec format)
#     traces = metadata.get("traces", [])
#     if isinstance(traces, list):
#         for i, trace in enumerate(traces):
#             if not isinstance(trace, dict):
#                 continue
#             trace_id = trace.get("id", f"trace_{i}")
#             trace_type = trace.get("type", "line")
#             trace_label = trace.get("label", f"Trace {i}")
#             ax_idx = trace.get("axes_index", 0)
# 
#             # Use axes bbox as fallback for trace bbox
#             ax_panel_key = None
#             for key in bboxes:
#                 if key.endswith("_panel") and bboxes[key].get("ax_id", "").endswith(
#                     str(ax_idx)
#                 ):
#                     ax_panel_key = key
#                     break
#             if not ax_panel_key:
#                 # Find any panel bbox
#                 for key in bboxes:
#                     if key.endswith("_panel"):
#                         ax_panel_key = key
#                         break
# 
#             trace_bbox = {
#                 "label": trace_label,
#                 "element_type": trace_type,
#                 "trace_idx": i,
#                 "ax_id": f"ax{ax_idx}",
#             }
# 
#             # Copy panel bbox coordinates if available
#             if ax_panel_key and ax_panel_key in bboxes:
#                 panel = bboxes[ax_panel_key]
#                 trace_bbox["x0"] = panel.get("x0", 0)
#                 trace_bbox["y0"] = panel.get("y0", 0)
#                 trace_bbox["x1"] = panel.get("x1", img_width)
#                 trace_bbox["y1"] = panel.get("y1", img_height)
# 
#             bboxes[f"trace_{i}"] = trace_bbox
# 
#     # If no bboxes found, return minimal set
#     if not bboxes:
#         bboxes["panel"] = {
#             "x0": 0,
#             "y0": 0,
#             "x1": img_width,
#             "y1": img_height,
#             "label": "Panel",
#             "is_panel": True,
#         }
# 
#     return bboxes
# 
# 
# def _guess_element_type(name: str) -> str:
#     """Guess element type from element name."""
#     name_lower = name.lower()
#     if "line" in name_lower or "trace" in name_lower:
#         return "line"
#     elif "scatter" in name_lower:
#         return "scatter"
#     elif "bar" in name_lower:
#         return "bar"
#     elif "fill" in name_lower:
#         return "fill"
#     elif "xlabel" in name_lower:
#         return "xlabel"
#     elif "ylabel" in name_lower:
#         return "ylabel"
#     elif "title" in name_lower:
#         return "title"
#     elif "legend" in name_lower:
#         return "legend"
#     elif "xaxis" in name_lower:
#         return "xaxis"
#     elif "yaxis" in name_lower:
#         return "yaxis"
#     elif "panel" in name_lower:
#         return "panel"
#     return "unknown"
# 
# 
# def extract_bboxes_from_geometry_px(
#     geometry_data: Dict[str, Any],
#     img_width: int,
#     img_height: int,
# ) -> Dict[str, Any]:
#     """Extract bounding boxes from geometry_px.json (cached pixel coordinates).
# 
#     This provides precise pixel coordinates for interactive element selection.
# 
#     Args:
#         geometry_data: JSON data from geometry_px.json
#         img_width: Actual image width in pixels
#         img_height: Actual image height in pixels
# 
#     Returns:
#         Dict with bboxes keyed by element name
#     """
#     bboxes = {}
# 
#     # Get figure dimensions from geometry to calculate scale
#     figure_px = geometry_data.get("figure_px", [img_width, img_height])
#     if isinstance(figure_px, list) and len(figure_px) >= 2:
#         geom_width, geom_height = figure_px[0], figure_px[1]
#     else:
#         geom_width, geom_height = img_width, img_height
# 
#     # Scale factor if image size differs from geometry
#     scale_x = img_width / geom_width if geom_width > 0 else 1
#     scale_y = img_height / geom_height if geom_height > 0 else 1
# 
#     # Extract axes bboxes
#     axes = geometry_data.get("axes", [])
#     for i, ax in enumerate(axes):
#         if not isinstance(ax, dict):
#             continue
#         ax_id = ax.get("id", f"ax{i}")
#         bbox_px = ax.get("bbox_px", {})
#         if bbox_px:
#             x0 = float(bbox_px.get("x0", 0)) * scale_x
#             y0 = float(bbox_px.get("y0", 0)) * scale_y
#             w = float(bbox_px.get("width", 0)) * scale_x
#             h = float(bbox_px.get("height", 0)) * scale_y
#             bboxes[f"{ax_id}_panel"] = {
#                 "x0": int(x0),
#                 "y0": int(y0),
#                 "x1": int(x0 + w),
#                 "y1": int(y0 + h),
#                 "label": f"Axes {ax_id}",
#                 "ax_id": ax_id,
#                 "is_panel": True,
#             }
# 
#     # Helper to safely convert to int (handle inf/nan)
#     def safe_int(val, default=0, max_val=10000):
#         import math
# 
#         if val is None or math.isinf(val) or math.isnan(val):
#             return default
#         return max(0, min(int(val), max_val))
# 
#     # Extract artists (lines, scatter, bars, etc.)
#     artists = geometry_data.get("artists", [])
#     for i, artist in enumerate(artists):
#         if not isinstance(artist, dict):
#             continue
# 
#         artist_id = artist.get("id", str(i))
#         artist_type = artist.get("type", "unknown")
#         artist_label = artist.get("label") or f"{artist_type}_{i}"
#         axes_index = artist.get("axes_index", 0)
# 
#         # Get bbox_px
#         bbox_px = artist.get("bbox_px", {})
#         if bbox_px:
#             x0 = float(bbox_px.get("x0", 0)) * scale_x
#             y0 = float(bbox_px.get("y0", 0)) * scale_y
#             w = float(bbox_px.get("width", 0)) * scale_x
#             h = float(bbox_px.get("height", 0)) * scale_y
# 
#             artist_bbox = {
#                 "x0": safe_int(x0, 0, img_width),
#                 "y0": safe_int(y0, 0, img_height),
#                 "x1": safe_int(x0 + w, img_width, img_width),
#                 "y1": safe_int(y0 + h, img_height, img_height),
#                 "label": artist_label,
#                 "element_type": artist_type,
#                 "trace_idx": i,
#                 "ax_id": f"ax{axes_index}",
#             }
# 
#             # Get path_px for lines (for precise hover detection)
#             path_px = artist.get("path_px", [])
#             if path_px and len(path_px) > 0:
#                 import math
# 
#                 # Scale points to actual image coordinates, filter out inf/nan
#                 scaled_points = []
#                 for pt in path_px:
#                     if isinstance(pt, (list, tuple)) and len(pt) >= 2:
#                         px, py = pt[0] * scale_x, pt[1] * scale_y
#                         if not (
#                             math.isinf(px)
#                             or math.isinf(py)
#                             or math.isnan(px)
#                             or math.isnan(py)
#                         ):
#                             scaled_points.append([px, py])
#                 if scaled_points:
#                     artist_bbox["points"] = scaled_points
# 
#             # Get scatter points if available
#             scatter_px = artist.get("scatter_px", [])
#             if scatter_px and len(scatter_px) > 0:
#                 import math
# 
#                 scaled_points = []
#                 for pt in scatter_px:
#                     if isinstance(pt, (list, tuple)) and len(pt) >= 2:
#                         px, py = pt[0] * scale_x, pt[1] * scale_y
#                         if not (
#                             math.isinf(px)
#                             or math.isinf(py)
#                             or math.isnan(px)
#                             or math.isnan(py)
#                         ):
#                             scaled_points.append([px, py])
#                 if scaled_points:
#                     artist_bbox["points"] = scaled_points
#                     artist_bbox["element_type"] = "scatter"
# 
#             bboxes[f"trace_{i}"] = artist_bbox
# 
#     # Extract from selectable_regions (title, xlabel, ylabel, xaxis, yaxis)
#     selectable = geometry_data.get("selectable_regions", {})
#     sel_axes = selectable.get("axes", [])
#     for ax_data in sel_axes:
#         if not isinstance(ax_data, dict):
#             continue
#         ax_idx = ax_data.get("index", 0)
#         ax_id = f"ax{ax_idx}"
# 
#         # Title
#         title_data = ax_data.get("title", {})
#         if title_data and "bbox_px" in title_data:
#             bbox = title_data["bbox_px"]
#             if isinstance(bbox, list) and len(bbox) >= 4:
#                 bboxes[f"{ax_id}_title"] = {
#                     "x0": safe_int(bbox[0] * scale_x, 0, img_width),
#                     "y0": safe_int(bbox[1] * scale_y, 0, img_height),
#                     "x1": safe_int(bbox[2] * scale_x, img_width, img_width),
#                     "y1": safe_int(bbox[3] * scale_y, img_height, img_height),
#                     "label": title_data.get("text", "Title"),
#                     "element_type": "title",
#                     "ax_id": ax_id,
#                 }
# 
#         # X Label
#         xlabel_data = ax_data.get("xlabel", {})
#         if xlabel_data and "bbox_px" in xlabel_data:
#             bbox = xlabel_data["bbox_px"]
#             if isinstance(bbox, list) and len(bbox) >= 4:
#                 bboxes[f"{ax_id}_xlabel"] = {
#                     "x0": safe_int(bbox[0] * scale_x, 0, img_width),
#                     "y0": safe_int(bbox[1] * scale_y, 0, img_height),
#                     "x1": safe_int(bbox[2] * scale_x, img_width, img_width),
#                     "y1": safe_int(bbox[3] * scale_y, img_height, img_height),
#                     "label": xlabel_data.get("text", "X Label"),
#                     "element_type": "xlabel",
#                     "ax_id": ax_id,
#                 }
# 
#         # Y Label
#         ylabel_data = ax_data.get("ylabel", {})
#         if ylabel_data and "bbox_px" in ylabel_data:
#             bbox = ylabel_data["bbox_px"]
#             if isinstance(bbox, list) and len(bbox) >= 4:
#                 bboxes[f"{ax_id}_ylabel"] = {
#                     "x0": safe_int(bbox[0] * scale_x, 0, img_width),
#                     "y0": safe_int(bbox[1] * scale_y, 0, img_height),
#                     "x1": safe_int(bbox[2] * scale_x, img_width, img_width),
#                     "y1": safe_int(bbox[3] * scale_y, img_height, img_height),
#                     "label": ylabel_data.get("text", "Y Label"),
#                     "element_type": "ylabel",
#                     "ax_id": ax_id,
#                 }
# 
#         # X Axis spine
#         xaxis_data = ax_data.get("xaxis", {})
#         if xaxis_data:
#             spine = xaxis_data.get("spine", {})
#             if spine and "bbox_px" in spine:
#                 bbox = spine["bbox_px"]
#                 if isinstance(bbox, list) and len(bbox) >= 4:
#                     bboxes[f"{ax_id}_xaxis"] = {
#                         "x0": safe_int(bbox[0] * scale_x, 0, img_width),
#                         "y0": safe_int(bbox[1] * scale_y, 0, img_height),
#                         "x1": safe_int(bbox[2] * scale_x, img_width, img_width),
#                         "y1": safe_int(bbox[3] * scale_y, img_height, img_height),
#                         "label": "X Axis",
#                         "element_type": "xaxis",
#                         "ax_id": ax_id,
#                     }
# 
#         # Y Axis spine
#         yaxis_data = ax_data.get("yaxis", {})
#         if yaxis_data:
#             spine = yaxis_data.get("spine", {})
#             if spine and "bbox_px" in spine:
#                 bbox = spine["bbox_px"]
#                 if isinstance(bbox, list) and len(bbox) >= 4:
#                     bboxes[f"{ax_id}_yaxis"] = {
#                         "x0": safe_int(bbox[0] * scale_x, 0, img_width),
#                         "y0": safe_int(bbox[1] * scale_y, 0, img_height),
#                         "x1": safe_int(bbox[2] * scale_x, img_width, img_width),
#                         "y1": safe_int(bbox[3] * scale_y, img_height, img_height),
#                         "label": "Y Axis",
#                         "element_type": "yaxis",
#                         "ax_id": ax_id,
#                     }
# 
#         # Legend
#         legend_data = ax_data.get("legend", {})
#         if legend_data and "bbox_px" in legend_data:
#             bbox = legend_data["bbox_px"]
#             if isinstance(bbox, list) and len(bbox) >= 4:
#                 bboxes[f"{ax_id}_legend"] = {
#                     "x0": safe_int(bbox[0] * scale_x, 0, img_width),
#                     "y0": safe_int(bbox[1] * scale_y, 0, img_height),
#                     "x1": safe_int(bbox[2] * scale_x, img_width, img_width),
#                     "y1": safe_int(bbox[3] * scale_y, img_height, img_height),
#                     "label": "Legend",
#                     "element_type": "legend",
#                     "ax_id": ax_id,
#                 }
# 
#     # If no bboxes found, return minimal set
#     if not bboxes:
#         bboxes["panel"] = {
#             "x0": 0,
#             "y0": 0,
#             "x1": img_width,
#             "y1": img_height,
#             "label": "Panel",
#             "is_panel": True,
#         }
# 
#     return bboxes
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_editor/_gui/_flask_editor/_bbox.py
# --------------------------------------------------------------------------------
