#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/editor/flask_editor/bbox.py
"""Bounding box extraction for figure elements.

Updated to integrate Schema v0.3 geometry extraction for shape-based hit testing.
"""

from typing import Dict, Any

# Try to import schema v0.3 geometry extraction
try:
    from scitex.plt.utils.metadata._geometry_extraction import (
        extract_axes_bbox_px,
        extract_line_geometry,
        extract_scatter_geometry,
        extract_polygon_geometry,
        extract_bar_group_geometry,
    )
    GEOMETRY_V03_AVAILABLE = True
except ImportError:
    GEOMETRY_V03_AVAILABLE = False


def extract_bboxes(
    fig, ax, renderer, img_width: int, img_height: int
) -> Dict[str, Any]:
    """Extract bounding boxes for all figure elements (single-axis)."""
    from matplotlib.transforms import Bbox

    # Get figure tight bbox in inches
    fig_bbox = fig.get_tightbbox(renderer)
    tight_x0 = fig_bbox.x0
    tight_y0 = fig_bbox.y0
    tight_width = fig_bbox.width
    tight_height = fig_bbox.height

    # bbox_inches='tight' adds pad_inches (default 0.1) around the tight bbox
    pad_inches = 0.1
    saved_width_inches = tight_width + 2 * pad_inches
    saved_height_inches = tight_height + 2 * pad_inches

    # Scale factors for converting inches to pixels
    scale_x = img_width / saved_width_inches
    scale_y = img_height / saved_height_inches

    bboxes = {}

    def get_element_bbox(element, name):
        """Get element bbox in image pixel coordinates."""
        try:
            bbox = element.get_window_extent(renderer)

            elem_x0_inches = bbox.x0 / fig.dpi
            elem_x1_inches = bbox.x1 / fig.dpi
            elem_y0_inches = bbox.y0 / fig.dpi
            elem_y1_inches = bbox.y1 / fig.dpi

            x0_rel = elem_x0_inches - tight_x0 + pad_inches
            x1_rel = elem_x1_inches - tight_x0 + pad_inches
            y0_rel = saved_height_inches - (elem_y1_inches - tight_y0 + pad_inches)
            y1_rel = saved_height_inches - (elem_y0_inches - tight_y0 + pad_inches)

            bboxes[name] = {
                "x0": max(0, int(x0_rel * scale_x)),
                "y0": max(0, int(y0_rel * scale_y)),
                "x1": min(img_width, int(x1_rel * scale_x)),
                "y1": min(img_height, int(y1_rel * scale_y)),
                "label": name.replace("_", " ").title(),
            }
        except Exception as e:
            print(f"Error getting bbox for {name}: {e}")

    def bbox_to_img_coords(bbox):
        """Convert matplotlib bbox to image pixel coordinates."""
        x0_inches = bbox.x0 / fig.dpi
        y0_inches = bbox.y0 / fig.dpi
        x1_inches = bbox.x1 / fig.dpi
        y1_inches = bbox.y1 / fig.dpi
        x0_rel = x0_inches - tight_x0 + pad_inches
        y0_rel = y0_inches - tight_y0 + pad_inches
        x1_rel = x1_inches - tight_x0 + pad_inches
        y1_rel = y1_inches - tight_y0 + pad_inches
        return {
            "x0": int(x0_rel * scale_x),
            "y0": int((saved_height_inches - y1_rel) * scale_y),
            "x1": int(x1_rel * scale_x),
            "y1": int((saved_height_inches - y0_rel) * scale_y),
        }

    # Get bboxes for title, labels
    if ax.title.get_text():
        get_element_bbox(ax.title, "title")
    if ax.xaxis.label.get_text():
        get_element_bbox(ax.xaxis.label, "xlabel")
    if ax.yaxis.label.get_text():
        get_element_bbox(ax.yaxis.label, "ylabel")

    # Get axis bboxes
    _extract_axis_bboxes(ax, renderer, bboxes, bbox_to_img_coords, Bbox)

    # Get legend bbox
    legend = ax.get_legend()
    if legend:
        get_element_bbox(legend, "legend")

    # Get trace (line) bboxes
    _extract_trace_bboxes(
        ax,
        fig,
        renderer,
        bboxes,
        get_element_bbox,
        tight_x0,
        tight_y0,
        saved_height_inches,
        scale_x,
        scale_y,
        pad_inches,
    )

    # Add schema v0.3 metadata if available
    if GEOMETRY_V03_AVAILABLE:
        axes_bbox = extract_axes_bbox_px(ax, fig)
        bboxes["_meta"] = {
            "schema_version": "0.3.0",
            "axes_bbox_px": axes_bbox,
            "geometry_available": True,
        }

    return bboxes


def extract_bboxes_multi(
    fig, axes_map: Dict[str, Any], renderer, img_width: int, img_height: int
) -> Dict[str, Any]:
    """Extract bounding boxes for all elements in a multi-axis figure.

    Args:
        fig: Matplotlib figure
        axes_map: Dict mapping axis IDs (e.g., 'ax_00') to matplotlib Axes objects
        renderer: Matplotlib renderer
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Dict with bboxes keyed by "{ax_id}_{element_type}" (e.g., "ax_00_xlabel")
    """
    from matplotlib.transforms import Bbox

    # Get figure tight bbox in inches
    fig_bbox = fig.get_tightbbox(renderer)
    tight_x0 = fig_bbox.x0
    tight_y0 = fig_bbox.y0
    tight_width = fig_bbox.width
    tight_height = fig_bbox.height

    # bbox_inches='tight' adds pad_inches (default 0.1) around the tight bbox
    pad_inches = 0.1
    saved_width_inches = tight_width + 2 * pad_inches
    saved_height_inches = tight_height + 2 * pad_inches

    # Scale factors for converting inches to pixels
    scale_x = img_width / saved_width_inches
    scale_y = img_height / saved_height_inches

    bboxes = {}

    def get_element_bbox(element, name, ax_id, current_ax=None):
        """Get element bbox in image pixel coordinates."""
        import numpy as np
        full_name = f"{ax_id}_{name}"
        try:
            bbox = element.get_window_extent(renderer)

            # Check for invalid bbox (infinity or NaN)
            if not (np.isfinite(bbox.x0) and np.isfinite(bbox.x1) and
                    np.isfinite(bbox.y0) and np.isfinite(bbox.y1)):
                # Try to get bbox from data for scatter/collection elements
                if hasattr(element, 'get_offsets') and current_ax is not None:
                    offsets = element.get_offsets()
                    if len(offsets) > 0 and np.isfinite(offsets).all():
                        # Use axis transform to get display coordinates
                        display_coords = current_ax.transData.transform(offsets)
                        x0 = display_coords[:, 0].min()
                        x1 = display_coords[:, 0].max()
                        y0 = display_coords[:, 1].min()
                        y1 = display_coords[:, 1].max()
                        if np.isfinite([x0, x1, y0, y1]).all():
                            from matplotlib.transforms import Bbox
                            bbox = Bbox.from_extents(x0, y0, x1, y1)
                        else:
                            return  # Skip this element
                    else:
                        return  # Skip this element
                else:
                    return  # Skip this element

            elem_x0_inches = bbox.x0 / fig.dpi
            elem_x1_inches = bbox.x1 / fig.dpi
            elem_y0_inches = bbox.y0 / fig.dpi
            elem_y1_inches = bbox.y1 / fig.dpi

            x0_rel = elem_x0_inches - tight_x0 + pad_inches
            x1_rel = elem_x1_inches - tight_x0 + pad_inches
            y0_rel = saved_height_inches - (elem_y1_inches - tight_y0 + pad_inches)
            y1_rel = saved_height_inches - (elem_y0_inches - tight_y0 + pad_inches)

            # Clamp values to avoid overflow
            x0_px = max(0, min(img_width, int(x0_rel * scale_x)))
            y0_px = max(0, min(img_height, int(y0_rel * scale_y)))
            x1_px = max(0, min(img_width, int(x1_rel * scale_x)))
            y1_px = max(0, min(img_height, int(y1_rel * scale_y)))

            bboxes[full_name] = {
                "x0": x0_px,
                "y0": y0_px,
                "x1": x1_px,
                "y1": y1_px,
                "label": f"{ax_id}: {name.replace('_', ' ').title()}",
                "ax_id": ax_id,
            }
        except Exception as e:
            print(f"Error getting bbox for {full_name}: {e}")

    def bbox_to_img_coords(bbox):
        """Convert matplotlib bbox to image pixel coordinates."""
        x0_inches = bbox.x0 / fig.dpi
        y0_inches = bbox.y0 / fig.dpi
        x1_inches = bbox.x1 / fig.dpi
        y1_inches = bbox.y1 / fig.dpi
        x0_rel = x0_inches - tight_x0 + pad_inches
        y0_rel = y0_inches - tight_y0 + pad_inches
        x1_rel = x1_inches - tight_x0 + pad_inches
        y1_rel = y1_inches - tight_y0 + pad_inches
        return {
            "x0": int(x0_rel * scale_x),
            "y0": int((saved_height_inches - y1_rel) * scale_y),
            "x1": int(x1_rel * scale_x),
            "y1": int((saved_height_inches - y0_rel) * scale_y),
        }

    # Extract bboxes for each axis
    for ax_id, ax in axes_map.items():
        # Get axes bounding box (the entire panel area)
        try:
            ax_bbox = ax.get_window_extent(renderer)
            coords = bbox_to_img_coords(ax_bbox)
            # Extract actual title/labels from the axes
            title_text = ax.title.get_text() if ax.title else ""
            xlabel_text = ax.xaxis.label.get_text() if ax.xaxis.label else ""
            ylabel_text = ax.yaxis.label.get_text() if ax.yaxis.label else ""
            bboxes[f"{ax_id}_panel"] = {
                **coords,
                "label": f"Panel {ax_id}",
                "ax_id": ax_id,
                "is_panel": True,
                "title": title_text,
                "xlabel": xlabel_text,
                "ylabel": ylabel_text,
            }
        except Exception as e:
            print(f"Error getting panel bbox for {ax_id}: {e}")

        # Get bboxes for title, labels
        if ax.title.get_text():
            get_element_bbox(ax.title, "title", ax_id, ax)
        if ax.xaxis.label.get_text():
            get_element_bbox(ax.xaxis.label, "xlabel", ax_id, ax)
        if ax.yaxis.label.get_text():
            get_element_bbox(ax.yaxis.label, "ylabel", ax_id, ax)

        # Get X-axis bbox (spine + ticks + ticklabels)
        _extract_axis_bboxes_for_axis(ax, ax_id, renderer, bboxes, bbox_to_img_coords, Bbox)

        # Get legend bbox
        legend = ax.get_legend()
        if legend:
            get_element_bbox(legend, "legend", ax_id, ax)

        # Get trace (line) bboxes
        _extract_trace_bboxes_for_axis(
            ax,
            ax_id,
            fig,
            renderer,
            bboxes,
            get_element_bbox,
            tight_x0,
            tight_y0,
            saved_height_inches,
            scale_x,
            scale_y,
            pad_inches,
        )

    # Add schema v0.3 metadata if available
    if GEOMETRY_V03_AVAILABLE:
        axes_bboxes = {}
        for ax_id, ax in axes_map.items():
            axes_bboxes[ax_id] = extract_axes_bbox_px(ax, fig)
        bboxes["_meta"] = {
            "schema_version": "0.3.0",
            "axes": axes_bboxes,
            "geometry_available": True,
        }

    return bboxes


def _extract_trace_bboxes_for_axis(
    ax,
    ax_id,
    fig,
    renderer,
    bboxes,
    get_element_bbox,
    tight_x0,
    tight_y0,
    saved_height_inches,
    scale_x,
    scale_y,
    pad_inches,
):
    """Extract bboxes for all data elements in a specific axis.

    Handles:
    - Lines (plot, errorbar lines)
    - Scatter points (PathCollection)
    - Fill areas (PolyCollection from fill_between)
    - Bars (Rectangle patches)
    """
    import numpy as np

    def coords_to_img_points(data_coords):
        """Convert data coordinates to image pixel coordinates."""
        if len(data_coords) == 0:
            return []
        transform = ax.transData
        points_display = transform.transform(data_coords)
        points_img = []
        for px, py in points_display:
            # Skip invalid points (NaN, infinity)
            if not np.isfinite(px) or not np.isfinite(py):
                continue
            px_inches = px / fig.dpi
            py_inches = py / fig.dpi
            x_rel = px_inches - tight_x0 + pad_inches
            y_rel = saved_height_inches - (py_inches - tight_y0 + pad_inches)
            # Clamp to reasonable bounds to avoid overflow
            x_img = max(-10000, min(10000, int(x_rel * scale_x)))
            y_img = max(-10000, min(10000, int(y_rel * scale_y)))
            points_img.append([x_img, y_img])
        # Downsample if too many
        if len(points_img) > 100:
            step = len(points_img) // 100
            points_img = points_img[::step]
        return points_img

    # 1. Extract lines (plot, errorbar lines, etc.)
    line_idx = 0
    for line in ax.get_lines():
        try:
            label = line.get_label()
            # Include unlabeled lines but mark them appropriately
            if label is None or label.startswith("_"):
                label = None  # Will use generic name

            trace_name = f"trace_{line_idx}"
            full_name = f"{ax_id}_{trace_name}"
            get_element_bbox(line, trace_name, ax_id, ax)

            if full_name in bboxes:
                bboxes[full_name]["label"] = f"{ax_id}: {label or f'Line {line_idx}'}"
                bboxes[full_name]["trace_idx"] = line_idx
                bboxes[full_name]["element_type"] = "line"

                xdata, ydata = line.get_xdata(), line.get_ydata()
                if len(xdata) > 0:
                    bboxes[full_name]["points"] = coords_to_img_points(
                        list(zip(xdata, ydata))
                    )

                # Add schema v0.3 geometry_px if available
                if GEOMETRY_V03_AVAILABLE:
                    try:
                        geom = extract_line_geometry(line, ax, fig)
                        bboxes[full_name]["geometry_px"] = geom
                    except Exception:
                        pass  # Fall back to legacy points

            line_idx += 1
        except Exception as e:
            print(f"Error getting line bbox for {ax_id}: {e}")

    # 2. Extract collections (scatter, fill_between, etc.)
    coll_idx = 0
    for coll in ax.collections:
        try:
            label = coll.get_label()
            if label is None or label.startswith("_"):
                # Still extract unlabeled collections but with generic name
                label = None

            coll_type = type(coll).__name__
            if coll_type == "PathCollection":
                # Scatter points
                element_name = f"scatter_{coll_idx}"
                full_name = f"{ax_id}_{element_name}"
                get_element_bbox(coll, element_name, ax_id, ax)

                if full_name in bboxes:
                    bboxes[full_name]["label"] = f"{ax_id}: {label or f'Scatter {coll_idx}'}"
                    bboxes[full_name]["element_type"] = "scatter"

                    # Get scatter point positions
                    offsets = coll.get_offsets()
                    if len(offsets) > 0:
                        bboxes[full_name]["points"] = coords_to_img_points(offsets)

                    # Add schema v0.3 geometry_px if available
                    if GEOMETRY_V03_AVAILABLE:
                        try:
                            geom = extract_scatter_geometry(coll, ax, fig)
                            bboxes[full_name]["geometry_px"] = geom
                        except Exception:
                            pass  # Fall back to legacy points

            elif coll_type in ("PolyCollection", "FillBetweenPolyCollection"):
                # Fill areas (fill_between, etc.)
                element_name = f"fill_{coll_idx}"
                full_name = f"{ax_id}_{element_name}"
                get_element_bbox(coll, element_name, ax_id, ax)

                if full_name in bboxes:
                    bboxes[full_name]["label"] = f"{ax_id}: {label or f'Fill {coll_idx}'}"
                    bboxes[full_name]["element_type"] = "fill"

                    # Add schema v0.3 geometry_px if available
                    if GEOMETRY_V03_AVAILABLE:
                        try:
                            geom = extract_polygon_geometry(coll, ax, fig)
                            bboxes[full_name]["geometry_px"] = geom
                        except Exception:
                            pass

            coll_idx += 1
        except Exception as e:
            print(f"Error getting collection bbox for {ax_id}: {e}")

    # 3. Extract patches (bars, rectangles, etc.)
    patch_idx = 0
    for patch in ax.patches:
        try:
            label = patch.get_label()
            patch_type = type(patch).__name__

            if patch_type == "Rectangle":
                # Bar chart bars
                element_name = f"bar_{patch_idx}"
                full_name = f"{ax_id}_{element_name}"
                get_element_bbox(patch, element_name, ax_id, ax)

                if full_name in bboxes:
                    bboxes[full_name]["label"] = f"{ax_id}: {label or f'Bar {patch_idx}'}"
                    bboxes[full_name]["element_type"] = "bar"

            patch_idx += 1
        except Exception as e:
            print(f"Error getting patch bbox for {ax_id}: {e}")


def _extract_axis_bboxes_for_axis(ax, ax_id, renderer, bboxes, bbox_to_img_coords, Bbox):
    """Extract X and Y axis bboxes for a specific axis (multi-axis version)."""
    try:
        # X-axis: combine spine and tick labels into one bbox
        x_axis_bboxes = []
        for ticklabel in ax.xaxis.get_ticklabels():
            if ticklabel.get_visible():
                try:
                    tb = ticklabel.get_window_extent(renderer)
                    if tb.width > 0:
                        x_axis_bboxes.append(tb)
                except Exception:
                    pass
        for tick in ax.xaxis.get_major_ticks():
            if tick.tick1line.get_visible():
                try:
                    tb = tick.tick1line.get_window_extent(renderer)
                    if tb.width > 0 or tb.height > 0:
                        x_axis_bboxes.append(tb)
                except Exception:
                    pass
        spine_bbox = ax.spines["bottom"].get_window_extent(renderer)
        if spine_bbox.width > 0:
            if x_axis_bboxes:
                tick_union = Bbox.union(x_axis_bboxes)
                constrained_spine = Bbox.from_extents(
                    tick_union.x0, spine_bbox.y0, tick_union.x1, spine_bbox.y1
                )
                x_axis_bboxes.append(constrained_spine)
            else:
                x_axis_bboxes.append(spine_bbox)
        if x_axis_bboxes:
            combined = Bbox.union(x_axis_bboxes)
            bboxes[f"{ax_id}_xaxis"] = bbox_to_img_coords(combined)
            bboxes[f"{ax_id}_xaxis"]["label"] = f"{ax_id}: X Axis"
            bboxes[f"{ax_id}_xaxis"]["ax_id"] = ax_id
            bboxes[f"{ax_id}_xaxis"]["element_type"] = "xaxis"

        # Y-axis: combine spine and tick labels into one bbox
        y_axis_bboxes = []
        for ticklabel in ax.yaxis.get_ticklabels():
            if ticklabel.get_visible():
                try:
                    tb = ticklabel.get_window_extent(renderer)
                    if tb.width > 0:
                        y_axis_bboxes.append(tb)
                except Exception:
                    pass
        for tick in ax.yaxis.get_major_ticks():
            if tick.tick1line.get_visible():
                try:
                    tb = tick.tick1line.get_window_extent(renderer)
                    if tb.width > 0 or tb.height > 0:
                        y_axis_bboxes.append(tb)
                except Exception:
                    pass
        spine_bbox = ax.spines["left"].get_window_extent(renderer)
        if spine_bbox.height > 0:
            if y_axis_bboxes:
                tick_union = Bbox.union(y_axis_bboxes)
                constrained_spine = Bbox.from_extents(
                    spine_bbox.x0, tick_union.y0, spine_bbox.x1, tick_union.y1
                )
                y_axis_bboxes.append(constrained_spine)
            else:
                y_axis_bboxes.append(spine_bbox)
        if y_axis_bboxes:
            combined = Bbox.union(y_axis_bboxes)
            padded = Bbox.from_extents(
                combined.x0 - 10, combined.y0 - 5, combined.x1 + 5, combined.y1 + 5
            )
            bboxes[f"{ax_id}_yaxis"] = bbox_to_img_coords(padded)
            bboxes[f"{ax_id}_yaxis"]["label"] = f"{ax_id}: Y Axis"
            bboxes[f"{ax_id}_yaxis"]["ax_id"] = ax_id
            bboxes[f"{ax_id}_yaxis"]["element_type"] = "yaxis"

    except Exception as e:
        print(f"Error getting axis bboxes for {ax_id}: {e}")


def _extract_axis_bboxes(ax, renderer, bboxes, bbox_to_img_coords, Bbox):
    """Extract bboxes for X and Y axis elements."""
    try:
        # X-axis: combine spine and tick labels into one bbox
        x_axis_bboxes = []
        for ticklabel in ax.xaxis.get_ticklabels():
            if ticklabel.get_visible():
                try:
                    tb = ticklabel.get_window_extent(renderer)
                    if tb.width > 0:
                        x_axis_bboxes.append(tb)
                except Exception:
                    pass
        for tick in ax.xaxis.get_major_ticks():
            if tick.tick1line.get_visible():
                try:
                    tb = tick.tick1line.get_window_extent(renderer)
                    if tb.width > 0 or tb.height > 0:
                        x_axis_bboxes.append(tb)
                except Exception:
                    pass
        spine_bbox = ax.spines["bottom"].get_window_extent(renderer)
        if spine_bbox.width > 0:
            if x_axis_bboxes:
                tick_union = Bbox.union(x_axis_bboxes)
                constrained_spine = Bbox.from_extents(
                    tick_union.x0, spine_bbox.y0, tick_union.x1, spine_bbox.y1
                )
                x_axis_bboxes.append(constrained_spine)
            else:
                x_axis_bboxes.append(spine_bbox)
        if x_axis_bboxes:
            combined = Bbox.union(x_axis_bboxes)
            bboxes["xaxis_ticks"] = bbox_to_img_coords(combined)
            bboxes["xaxis_ticks"]["label"] = "X Spine & Ticks"

        # Y-axis: combine spine and tick labels into one bbox
        y_axis_bboxes = []
        for ticklabel in ax.yaxis.get_ticklabels():
            if ticklabel.get_visible():
                try:
                    tb = ticklabel.get_window_extent(renderer)
                    if tb.width > 0:
                        y_axis_bboxes.append(tb)
                except Exception:
                    pass
        for tick in ax.yaxis.get_major_ticks():
            if tick.tick1line.get_visible():
                try:
                    tb = tick.tick1line.get_window_extent(renderer)
                    if tb.width > 0 or tb.height > 0:
                        y_axis_bboxes.append(tb)
                except Exception:
                    pass
        spine_bbox = ax.spines["left"].get_window_extent(renderer)
        if spine_bbox.height > 0:
            if y_axis_bboxes:
                tick_union = Bbox.union(y_axis_bboxes)
                constrained_spine = Bbox.from_extents(
                    spine_bbox.x0, tick_union.y0, spine_bbox.x1, tick_union.y1
                )
                y_axis_bboxes.append(constrained_spine)
            else:
                y_axis_bboxes.append(spine_bbox)
        if y_axis_bboxes:
            combined = Bbox.union(y_axis_bboxes)
            padded = Bbox.from_extents(
                combined.x0 - 10, combined.y0 - 5, combined.x1 + 5, combined.y1 + 5
            )
            bboxes["yaxis_ticks"] = bbox_to_img_coords(padded)
            bboxes["yaxis_ticks"]["label"] = "Y Spine & Ticks"

    except Exception as e:
        print(f"Error getting axis bboxes: {e}")


def _extract_trace_bboxes(
    ax,
    fig,
    renderer,
    bboxes,
    get_element_bbox,
    tight_x0,
    tight_y0,
    saved_height_inches,
    scale_x,
    scale_y,
    pad_inches,
):
    """Extract bboxes for all data elements (lines, scatter, fill) with proximity detection."""
    import numpy as np

    def coords_to_img_points(data_coords):
        """Convert data coordinates to image pixel coordinates."""
        if len(data_coords) == 0:
            return []
        transform = ax.transData
        points_display = transform.transform(data_coords)
        points_img = []
        for px, py in points_display:
            if not np.isfinite(px) or not np.isfinite(py):
                continue
            px_inches = px / fig.dpi
            py_inches = py / fig.dpi
            x_rel = px_inches - tight_x0 + pad_inches
            y_rel = saved_height_inches - (py_inches - tight_y0 + pad_inches)
            x_img = max(-10000, min(10000, int(x_rel * scale_x)))
            y_img = max(-10000, min(10000, int(y_rel * scale_y)))
            points_img.append([x_img, y_img])
        if len(points_img) > 100:
            step = len(points_img) // 100
            points_img = points_img[::step]
        return points_img

    # 1. Extract lines
    for idx, line in enumerate(ax.get_lines()):
        try:
            label = line.get_label()
            # Include unlabeled lines but mark them appropriately
            if label is None or label.startswith("_"):
                label = None  # Will use generic name
            get_element_bbox(line, f"trace_{idx}")
            if f"trace_{idx}" in bboxes:
                bboxes[f"trace_{idx}"]["label"] = label or f"Trace {idx}"
                bboxes[f"trace_{idx}"]["trace_idx"] = idx
                bboxes[f"trace_{idx}"]["element_type"] = "line"

                xdata, ydata = line.get_xdata(), line.get_ydata()
                if len(xdata) > 0:
                    bboxes[f"trace_{idx}"]["points"] = coords_to_img_points(
                        list(zip(xdata, ydata))
                    )

                # Add schema v0.3 geometry_px if available
                if GEOMETRY_V03_AVAILABLE:
                    try:
                        geom = extract_line_geometry(line, ax, fig)
                        bboxes[f"trace_{idx}"]["geometry_px"] = geom
                    except Exception:
                        pass
        except Exception as e:
            print(f"Error getting trace bbox: {e}")

    # 2. Extract collections (scatter, fill_between)
    coll_idx = 0
    for coll in ax.collections:
        try:
            label = coll.get_label()
            if label is None or label.startswith("_"):
                label = None

            coll_type = type(coll).__name__
            if coll_type == "PathCollection":
                # Scatter points
                elem_key = f"scatter_{coll_idx}"
                get_element_bbox(coll, elem_key)

                # Initialize entry if bbox extraction failed but we have data
                offsets = coll.get_offsets()
                if elem_key not in bboxes and len(offsets) > 0:
                    # Create bbox from data coordinates as fallback
                    points_img = coords_to_img_points(offsets)
                    if points_img:
                        xs = [p[0] for p in points_img]
                        ys = [p[1] for p in points_img]
                        bboxes[elem_key] = {
                            "x0": min(xs) - 10,
                            "y0": min(ys) - 10,
                            "x1": max(xs) + 10,
                            "y1": max(ys) + 10,
                        }

                if elem_key in bboxes:
                    bboxes[elem_key]["label"] = label or f"Scatter {coll_idx}"
                    bboxes[elem_key]["element_type"] = "scatter"

                    if len(offsets) > 0:
                        bboxes[elem_key]["points"] = coords_to_img_points(offsets)

                    # Add schema v0.3 geometry_px if available
                    if GEOMETRY_V03_AVAILABLE:
                        try:
                            geom = extract_scatter_geometry(coll, ax, fig)
                            bboxes[elem_key]["geometry_px"] = geom
                        except Exception:
                            pass

            elif coll_type in ("PolyCollection", "FillBetweenPolyCollection"):
                # Fill areas
                get_element_bbox(coll, f"fill_{coll_idx}")
                if f"fill_{coll_idx}" in bboxes:
                    bboxes[f"fill_{coll_idx}"]["label"] = label or f"Fill {coll_idx}"
                    bboxes[f"fill_{coll_idx}"]["element_type"] = "fill"

                    # Add schema v0.3 geometry_px if available
                    if GEOMETRY_V03_AVAILABLE:
                        try:
                            geom = extract_polygon_geometry(coll, ax, fig)
                            bboxes[f"fill_{coll_idx}"]["geometry_px"] = geom
                        except Exception:
                            pass

            coll_idx += 1
        except Exception as e:
            print(f"Error getting collection bbox: {e}")


# EOF
